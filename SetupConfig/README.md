# SetupConfig – Guía de Instalación y Reproducción del Proyecto

Este directorio contiene todos los scripts necesarios para reconstruir desde cero el LLM especializado en configuraciones Cisco.

Incluye:

- Descarga del modelo base.

- Fine-tuning LoRA.

- Pruebas de inferencia.

- Ejemplo de agente.

# Contenido

    SetupConfig/
     ├── agenteP.py          # Agente funcional con LangGraph
     ├── download_model.py   # Descarga del LLaMA 3.2 3B base
     ├── finetune_model.py   # Script completo de fine-tuning LoRA
     └── test.py             # Prueba de inferencia del modelo base

# Guia de Uso:

# **1. Instalar entorno virtual:**

 Se recomienda trabajar dentro de un entorno virtual para ejecturar este proyecto, e instalar las librerias necesarias en ese entorno
 
    python3 -m venv venv
    source venv/bin/activate
    pip install transformers peft bitsandbytes accelerate datasets langchain langgraph torch

# **2. Descargar el modelo base:**

Para el correcto funcionamiento de este modelo se necesita el agente base Llama3.2-3B, para ello se tiene que pedir la licencia de uso, 
para la descarga de este se uso HugginFace, se recomienda guardar el modelo en la carpeta de Models, 
una vez se tenga la licencia de Huggin Face se tendra que autenticar para poder descargarlo y luego ejecutar el siguiente script:

    python3 download_model.py

Esto guardara el modelo en la carpeta en:

    /models/llama32-3b/

# **3. Hacerle Fintune Al modelo:**

Una vez descargado el agente se puede hacer un test sencillo ejecutando:

    python3 test.py

Para proceder a hacer un FineTune se ejecuta:

    python3 finetune_model.py

Esto ejecuta un FineTune con LoRA a 4 bits, el cual se puede hacer con GPUS de bajo costo, en este caso una de 6GB VRAM,
se entrena con el data set de la carpeta Knowledge y genera un adaptador lora en la carpeta de Models.

# **4. Ejecución del agente** 

Una vez que se ha completado el fine tune del modelo base, se debio crear un archivo con los pesos LoRA dentro de la carpeta Models, esta carpeta viene con una version del agente implementado, este agente cuenta con: 

- Detecta herramientas necesarias
- Ejecuta configure_interface, configure_dhcp, configure_vlan
- Responde con configuraciones Cisco reales

Para ejecutar este agente corre:

        python3 agenteP.py

En caso tal de que quieras correr la versión final del agente, es el archivo agenteJ.py, en la raiz del proyecto.



