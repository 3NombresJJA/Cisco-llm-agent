from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
import re

# ============================================================
# 1. Cargar LLM
# ============================================================

model_base = "./models/llama32-3b"
model_path = "./models/llama32-router-lora"

tokenizer = AutoTokenizer.from_pretrained(model_base)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_base,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.01,
    return_full_text=False,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id

)

llm = HuggingFacePipeline(pipeline=pipe)


# ============================================================
# 2. Templates
# ============================================================

def interface_template(interface, ip, mask, description, lan):
    return f"""
interface {interface}
 description {description}
 ip address {ip} {mask}
 switchport access vlan {lan}
 no shutdown
"""


def dhcp_template(pool, network, mask, router, dns=None):
    text = f"""
ip dhcp pool {pool}
 network {network} {mask}
 default-router {router}
"""
    if dns:
        text += f" dns-server {dns}\n"

    return text


def vlan_template(vlan_id, vlan_name, interface):
    return f"""
vlan {vlan_id}
 name {vlan_name}
exit
interface {interface}
 switchport access vlan {vlan_id}
 no shutdown
"""


# ============================================================
# 3. Tools sin inventar datos
# ============================================================

def parse_field(text, field):
    m = re.search(rf"{field}\s*[:=]\s*([\w\.\-/]+)", text, re.I)
    return m.group(1) if m else None


def interface_tool(params: str):
    interface = parse_field(params, "interface")
    ip = parse_field(params, "ip")
    mask = parse_field(params, "mask")
    description = parse_field(params, "description")
    lan = parse_field(params, "lan")

    missing = []
    if not interface: missing.append("interface")
    if not ip: missing.append("ip")
    if not mask: missing.append("mask")
    if not description: missing.append("description")
    if not lan: missing.append("lan")

    if missing:
        return f"Datos faltantes para interfaz: {', '.join(missing)}"

    return interface_template(interface, ip, mask, description, lan)


def dhcp_tool(params: str):
    pool = parse_field(params, "pool")
    network = parse_field(params, "network")
    mask = parse_field(params, "mask")
    router = parse_field(params, "router")
    dns = parse_field(params, "dns")  # ahora opcional

    missing = []
    if not pool: missing.append("pool")
    if not network: missing.append("network")
    if not mask: missing.append("mask")
    if not router: missing.append("router")

    if missing:
        return f"Datos faltantes DHCP: {', '.join(missing)}"

    # dns es opcional, no se inventa
    return dhcp_template(pool, network, mask, router, dns)


def vlan_tool(params: str):
    vlan_id = parse_field(params, "vlan")
    name = parse_field(params, "name")
    interface = parse_field(params, "interface")

    missing = []
    if not vlan_id: missing.append("vlan")
    if not name: missing.append("name")
    if not interface: missing.append("interface")

    if missing:
        return f"Datos faltantes VLAN: {', '.join(missing)}"

    return vlan_template(vlan_id, name, interface)


tools = [
    Tool(name="configure_interface", func=interface_tool,
         description="Configura interfaces Cisco."),
    Tool(name="configure_dhcp", func=dhcp_tool,
         description="Configura DHCP."),
    Tool(name="configure_vlan", func=vlan_tool,
         description="Configura VLAN.")
]


# ============================================================
# 4. Estado
# ============================================================

class AgentState(dict):
    input: str
    messages: list
    tool_call: str | None
    tool_input: str | None


# ============================================================
# 5. LLM Node SIN system prompt
# ============================================================

def llm_node(state: AgentState):

    system_message = """
Eres un asistente experto en configuración Cisco IOS.

SOLO puedes usar estas herramientas:
- configure_interface
- configure_dhcp
- configure_vlan

REGLAS:
- SOLO usa tools si el usuario pide interfaz, DHCP o VLAN
- Para OSPF, BGP, ACL, NAT → SOLO TEXTO
- NO inventes datos
- NO inventes conversaciones
- NO repitas la pregunta
- NO escribas como usuario
- Responde UNA sola vez

FORMATO TOOL:
TOOL=configure_interface
INPUT=interface:Gi0/0, ip:10.0.0.1, mask:255.255.255.0, description:WAN, lan:100
"""

    user_prompt = f"""{system_message}

ENTRADA_USUARIO:
{state['input']}

SALIDA:
"""

    response = llm.pipeline(
        user_prompt,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.01
    )

    # ✅ Extraer generación correctamente
    text = response[0]["generated_text"].strip()

    # ✅ CORTES ANTI-ALUCINACIÓN (ESTO ES CLAVE)
    cortes = ["Usuario:", "usuario:", "User:", "ENTRADA_USUARIO:", "SALIDA:"]
    for c in cortes:
        if c in text:
            text = text.split(c)[0].strip()

    # ✅ Detectar llamada a tool
    tool_match = re.search(r"TOOL=(\w+)", text)
    input_match = re.search(r"INPUT=(.+)", text, re.S)

    if tool_match:
        return {
            "messages": [AIMessage(content=text)],
            "tool_call": tool_match.group(1),
            "tool_input": input_match.group(1).strip() if input_match else None,
        }

    return {
        "messages": [AIMessage(content=text)],
        "tool_call": None,
        "tool_input": None
    }


# ============================================================
# 6. Tools Node
# ============================================================

def tool_node(state: AgentState):
    if not state["tool_call"]:
        return {"messages": state["messages"]}

    tool_name = state["tool_call"]
    tool = next((t for t in tools if t.name == tool_name), None)

    if not tool:
        return {
            "messages": state["messages"] + [
                AIMessage(content=f"Error: Tool {tool_name} no existe.")
            ]
        }

    result = tool.run(state["tool_input"] or "")

    # Si faltan datos, NO continuar el flujo
    if "Datos faltantes" in result:
        return {
            "messages": state["messages"] + [
                AIMessage(content=result)
            ]
        }

    return {
        "messages": state["messages"] + [
            AIMessage(content=result)
        ]
    }


# ============================================================
# 7. Grafo
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")

graph.add_edge("llm", "tools")
graph.add_edge("tools", END)

agent = graph.compile()


# ============================================================
# 8. Ejecución
# ============================================================

if __name__ == "__main__":
    while True:
        q = input("\nPregunta → ")

        resp = agent.invoke({"input": q, "messages": []})

        print("\n========= RESPUESTA =========\n")
        for msg in resp["messages"]:
            print(msg.content)
