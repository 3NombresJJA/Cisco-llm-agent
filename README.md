# Cisco-llm-agent
Cisco LLM Agent – Proyecto de Automatización de Redes con IA

Este repositorio contiene todo el flujo completo para entrenar, configurar y desplegar un agente LLM especializado en configuraciones de red Cisco, incluyendo:

-Descarga del modelo base LLaMA 3.2

-Fine-tuning con LoRA

-Dataset con conocimiento de redes

-Scripts para pruebas

-Arquitectura del agente con tools

Guía completa de instalación (ubicada en /SetupConfig)

*El modelo final está publicado en Hugging Face aquí:*

https://huggingface.co/Awakate/Cisco-Configuration-agent

La estructura del repositorio es la siguiente:

**Cisco-llm-agent**
 ├── SetupConfig/          # Scripts para montar el entorno y reproducir el proyecto

 │    ├── agenteP.py       # Versión 1 del agente (LangGraph)
 
 │    ├── download_model.py# Descarga del modelo base LLaMA 3.2 3B
 
 │    ├── finetune_model.py# Script para hacer fine-tuning LoRA
 
 │    └── test.py          # Test rápido del modelo base sin agente
 
 │
 
 ├── Knowledge/            # Dataset utilizado para el fine-tuning
 
 │    └── dataset.jsonl    # Datos en formato instruction-input-output
 
 │
 
 ├── Models/               # Opcional – puedes guardar aquí modelos locales
 
 │                         # (No se suben al repo por tamaño)
 
 
 │
 
 ├── agenteJ.py            # Version Final del agente
 
 ├── .gitignore            # Ignora archivos pesados y entornos
 
 └── README.md             # Este documento


 **¿Qué hace este proyecto?**

Este repositorio permite:

1. Descargar un modelo base (LLaMA 3.2 – 3B)
  Usando download_model.py se obtiene el modelo base sobre el cual se hace fine-tuning.

2. Entrenar un modelo especializado en Cisco (LoRA)
    El script finetune_model.py:
 
      -Aplica LoRA en 4-bit (optimizado para 6GB VRAM)

      -Entrena con un dataset especializado

      -Produce un adaptador LoRA publicable 


4. Ejecutar un agente inteligente con tools (LangGraph)
  El agente viene configurado con las herramientas:

    -configure_interface

    -configure_dhcp

    -configure_vlan


  Además:
  
   -Analiza prompts del usuario
    
   -Decide si usar una tool
 
  O responde directamente con comandos Cisco IOS

4. Usar el modelo en local o con HF
  El repositorio incluye un archivo test.py para probar inferencia local.

**Arquitectura del Agente (Resumen)**
Usuario → LLM → Parser de Tools → Nodo de Tools → Respuesta Final
  	
  -El usuario envía una instrucción
  
  -El LLM decide si debe llamar una tool
  
  -Un parser detecta: TOOL=... INPUT=...
  
  -La tool se ejecuta y devuelve configuración real
  
  -El agente responde

**¿Quieres reproducir todo el sistema?**
La guía completa está en:

/SetupConfig/README.md

Ahí encontrarás:
  
  -Instalación del entorno
  
  -Descarga del modelo base
  
  -Cómo hacer fine-tuning LoRA
  
  -Cómo probar el modelo final
  
  -Cómo ejecutar el agente
  
**Dataset usado para entrenar el modelo**

El dataset está en:

  Knowledge/dataset.jsonl

Con estructura:

{
  "instruction": "Configura DHCP",
  
  "input": "Pool red1 192.168.1.0/24 router 192.168.1.1",
  
  "output": "ip dhcp pool red1..."
}

Incluye configuraciones de:

  -Interfaces
  
  -VLAN
  
  -DHCP
  
  -OSPF
  
  -NAT
  
  -ACL
  
  -DNS

Escenarios enseñados manualmente

**Tecnologías Usadas**

  -Python 3.12
  
  -Transformers
  
  -PEFT / LoRA
  
  -BitsAndBytes
  
  -LangGraph
  
  -Hugging Face Hub

Contacto

Modelo en Hugging Face:
https://huggingface.co/Awakate/Cisco-Configuration-agent

Repositorio GitHub:
https://github.com/3NombresJJA/Cisco-llm-agent
