from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
    temperature=0.05,
    return_full_text=False,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id
)

raw_llm = HuggingFacePipeline(pipeline=pipe)


# ============================================================
# *** AGREGADO *** — Convertir modelo TEXT → CHAT
# ============================================================

class CleanChatModel:
    """
    Convierte un modelo de texto en un modelo de chat.
    Evita:
      - Autocompletar la entrada del usuario
      - Inventar "Usuario:" o "Asistente:"
      - Crear conversaciones falsas
      - Repetir system prompt
    """

    def __init__(self, base_llm):
        self.base_llm = base_llm

    def invoke(self, messages):
        system_text = ""
        conversation = ""

        for m in messages:
            if isinstance(m, SystemMessage):
                system_text += m.content.strip() + "\n\n"
            elif isinstance(m, HumanMessage):
                conversation += f"Usuario: {m.content.strip()}\nAsistente: "

        final_prompt = system_text + conversation

        # Llamamos al modelo TEXT
        out = self.base_llm.invoke(final_prompt)

        # Limpiar respuesta
        text = out.content if hasattr(out, "content") else out
        text = text.strip()

        # Evitar que se meta en turnos falsos
        if "Usuario:" in text:
            text = text.split("Usuario:")[0].strip()

        return AIMessage(content=text)


# Crear LLM DE CHAT
llm = CleanChatModel(raw_llm)


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
    
    m = re.search(rf"(?i){re.escape(field)}\s*[:=]\s*([\w\.\-/]+)", text)
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
    dns = parse_field(params, "dns")  # opcional

    missing = []
    if not pool: missing.append("pool")
    if not network: missing.append("network")
    if not mask: missing.append("mask")
    if not router: missing.append("router")

    if missing:
        return f"Datos faltantes DHCP: {', '.join(missing)}"

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

    system_message = SystemMessage(content="""


Eres un asistente especializado en configuración de redes Cisco.

Tienes disponibles SOLO estas herramientas:

1) configure_interface
2) configure_dhcp
3) configure_vlan

REGLAS OBLIGATORIAS:

1) Si el usuario pide configurar una interfaz → usa configure_interface
2) Si pide DHCP → usa configure_dhcp
3) Si pide VLAN → usa configure_vlan
4) NUNCA inventes nombres de tools
5) SIEMPRE usa este formato EXACTO cuando llames tools:
Si es para VLAN usa:
TOOL=configure_vlan
INPUT=vlan:<ID>, name:<NAME>, interface:<INTERFACE>

Si es para DHCP usa:
TOOL=configure_dhcp
INPUT=pool:<poolNAME>, network:<Network_IP>, mask:<MASK>, router:<Router_IP>, dns:<DNS_IP>

Si es para configurar interfaces usa únicamente:
TOOL=configure_interface
INPUT= interface:<Interface>, ip:<InterfaceIP>, mask:<MASK>, description:<TEXT>, lan:<ID>

6) Si no aplica ninguna tool → responde SOLO con texto normal.
7) No expliques tu razonamiento.
8) No repitas la pregunta.
9) Analiza los mensajes paso a paso.
10) Si al momento de llamar una tool faltan datos, pidele los datos faltantes al usuario.
11) NO TE INVENTES DATOS.
12) SIEMPRE PERO SIEMPRE ANALIZA SI DEBES USAR UNA TOOL O NO
13) NUNCA pero nunca agregues informacion a los mensajes de entrada
14) Nunca inventes turnos “Usuario:” o “Asistente:”.
15) Si el usuario pide OSPF, RIP, BGP, NAT, ACL, DNS u otros protocolos avanzados: 
- Responde con comandos reales de Cisco 
La respuesta debe: 
- Tener título
- Tener bloques de configuración 
- No ser genérica 
- No inventar datos si no se dieron

Ejemplo valido:
TOOL=configure_interface
INPUT=interface:Gi0/0, ip:192.168.1.1, mask:255.255.255.0, description:WAN, lan:10


""")

    # *** ÚNICO CAMBIO ***
    # En vez de pasar STRING, pasamos MENSAJES al modelo de chat
    user_msg = HumanMessage(content=state["input"])

    response = llm.invoke([system_message, user_msg])

    text = response.content

    tool_match = re.search(r"(?i)\bTOOL\s*[:=]\s*([A-Za-z0-9_\-]+)", text)
    input_match = re.search(r"(?i)\bINPUT\s*[:=]\s*(.+)", text, re.S)



    if tool_match:
        # normalizar nombre de tool (coincida con tus Tool.name)
        tool_name = tool_match.group(1).strip().lower()

        # limpiar input: quitar comas extra, saltos y espacios al inicio/fin
        raw_input = input_match.group(1).strip() if input_match else ""
        # opcional: reemplazar múltiples espacios y convertir comas a separadores estándar
        tool_input = re.sub(r"\s*,\s*", ", ", raw_input).strip(", ").strip()

        return {
            "messages": [AIMessage(content=text)],
            "tool_call": tool_name,
            "tool_input": tool_input,
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

    if "Datos faltantes" in result:
        return {
            "messages": state["messages"] + [
                AIMessage(content=result)
            ]
        }

    return {
        "messages": state["messages"] + [AIMessage(content=result)]
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
