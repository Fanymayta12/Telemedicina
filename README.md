# **Plataforma de Telemedicina con IA**

Una aplicación web desarrollada con Flask que utiliza inteligencia artificial para diagnosticar enfermedades basadas en imágenes médicas, como radiografías. Esta herramienta está diseñada para facilitar el acceso a diagnósticos médicos en zonas rurales o aisladas.

---

## **Características**
- Interfaz web intuitiva para cargar imágenes médicas.
- Diagnóstico automatizado utilizando un modelo de Machine Learning entrenado con imágenes de radiografías.
- Resultados mostrados de manera clara junto con la imagen cargada.
- Diseño responsivo y amigable.

---

## **Requisitos Previos**

1. **Python 3.10 o superior**
2. **Librerías necesarias**: Flask, TensorFlow, y otras especificadas en el archivo `requirements.txt`.

---

## **Instalación y Ejecución**

### **1. Crea y activa un entorno virtual**
Crear entorno virtual
```bash
python -m venv venv
```
Activar entorno virtual
```bash
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### **3. Instala las dependencias**
```bash
pip install -r requirements.txt
```

### **4. Asegúrate de tener el modelo entrenado**
Guarda el archivo `model.h5` (preentrenado para clasificar imágenes) en la raíz del proyecto. Si no tienes el modelo, puedes entrenarlo ejecutando:
```bash
python train.py
```

### **5. Ejecuta la aplicación**
```bash
python app.py
```

La aplicación estará disponible en [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## **Estructura del Proyecto**
```
.
├── static/
│   └── css/
│       └── index.css        # Archivo de estilos para el diseño de la página
├── templates/
│   └── index.html           # Página principal de la aplicación
├── train.py                 # Script para entrenar el modelo
├── app.py                   # Archivo principal de la aplicación Flask
├── requirements.txt         # Librerías necesarias
├── model.h5                 # Modelo entrenado de Machine Learning
└── README.md                # Este archivo
```

---

## **Cómo Usar**
1. Abre la página principal de la aplicación.
2. Sube una imagen de radiografía en formato JPG o PNG.
3. Haz clic en "Diagnosticar".
4. El resultado del diagnóstico y la imagen cargada se mostrarán en pantalla.

---
