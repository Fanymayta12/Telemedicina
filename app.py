from flask import Flask, render_template, request, redirect, url_for, flash
import os
import csv
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import csv
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import openai
from PIL import Image

# Configuración inicial
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['SECRET_KEY'] = 'your_secret_key'  # Para usar flash mensajes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Cargar el modelo de IA para diagnóstico de imagen
model = load_model('./model/best_model.keras')  # Asegúrate de que el modelo esté en el directorio correcto

# Configuración de OpenAI
openai.api_key = "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"  # Reemplaza con tu clave API de OpenAI

# Historial de conversaciones para el chatbot
conversation_history = []  # Comienza vacío

# Función para verificar extensiones permitidas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para verificar si el archivo es una imagen válida
def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # Verifica si es una imagen válida
        return True
    except (IOError, SyntaxError):
        return False

# Función para registrar el diagnóstico
def log_prediction(filename, result):
    log_path = 'diagnostics_log.csv'
    # Crear archivo si no existe
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['Filename', 'Result'])  # Cabeceras
    # Registrar diagnóstico
    with open(log_path, 'a', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow([filename, result])

# Función para obtener la respuesta de ChatGPT y mantener el historial
def get_chatgpt_response(user_input):
    try:
        # Agregar el mensaje del usuario al historial
        conversation_history.append({"role": "user", "content": user_input})

        # Llamada a la API de OpenAI para obtener la respuesta
        response = openai.ChatCompletion.create(
            model="gpt-4",  # O el modelo que estés utilizando
            messages=conversation_history,
            max_tokens=150
        )

        # Obtener la respuesta del modelo
        bot_response = response['choices'][0]['message']['content'].strip()

        # Agregar la respuesta del bot al historial
        conversation_history.append({"role": "assistant", "content": bot_response})

        return bot_response
    except Exception as e:
        return f"Error al obtener respuesta de la API: {e}"

# Función para decodificar predicción (ejemplo simple)
def decode_prediction(prediction):
    if prediction[0][0] > 0.6:
        return "Neumonía detectada"
    else:
        return "Pulmones saludables"

# Página principal
@app.route('/', methods=['GET', 'POST'])
def index():
    global conversation_history  # Asegúrate de modificar la variable global

    response = ""
    filename = None
    result = None

    # Si es una nueva sesión (GET) y el historial está vacío, agregamos un mensaje de bienvenida del bot.
    if len(conversation_history) == 0:
        # Aquí agregamos un mensaje de bienvenida para el chat
        conversation_history.append({"role": "assistant", "content": "Hola, soy tu asistente médico virtual. ¿En qué puedo ayudarte hoy?"})

    # Manejo de solicitud POST para el chatbot o diagnóstico
    if request.method == 'POST':
        if 'user_input' in request.form:
            user_input = request.form['user_input']
            response = get_chatgpt_response(user_input)

        elif 'file' in request.files:
            # Manejo de solicitud POST para subir una imagen y realizar diagnóstico
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Verificar si la imagen es válida
                if not is_valid_image(filepath):
                    os.remove(filepath)  # Elimina archivos no válidos
                    flash("El archivo no es una imagen válida.", "error")
                    return redirect(request.url)

                try:
                    # Procesar imagen con el modelo
                    img = image.load_img(filepath, target_size=(224, 224))  # Ajustar tamaño según el modelo
                    img_array = image.img_to_array(img) / 255.0  # Normalizar
                    img_array = np.expand_dims(img_array, axis=0)  # Agregar batch
                    prediction = model.predict(img_array)
                    result = decode_prediction(prediction)

                    # Registrar el diagnóstico
                    log_prediction(filename, result)
                except Exception as e:
                    flash(f"Ocurrió un error durante la predicción: {e}", "error")
                    return redirect(request.url)

    # Pasar tanto la respuesta del chatbot como el historial a la plantilla
    return render_template('index.html', response=response, filename=filename, result=result, conversation_history=conversation_history)
# En tu archivo de Flask, agrega lo siguiente:

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No se ha seleccionado un archivo.', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No se ha seleccionado un archivo.', 'error')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Procesar la imagen y realizar la predicción (ajustar según tu modelo)
        if not is_valid_image(filepath):
            os.remove(filepath)  # Eliminar archivo no válido
            flash("El archivo no es una imagen válida.", "error")
            return redirect(request.url)

        try:
            img = image.load_img(filepath, target_size=(224, 224))  # Ajusta según el modelo
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)

            # Decodificar la predicción
            result = decode_prediction(prediction)

            # Registrar el diagnóstico en el archivo CSV
            log_prediction(filename, result)

            return render_template('index.html', filename=filename, result=result)

        except Exception as e:
            flash(f"Ocurrió un error durante la predicción: {e}", "error")
            return redirect(request.url)

    else:
        flash('Tipo de archivo no permitido. Solo JPG, PNG.', 'error')
        return redirect(request.url)

# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

# Ruta para resetear el historial de conversación
@app.route('/reset', methods=['POST'])
def reset_chat():
    global conversation_history
    conversation_history = []  # Borrar el historial completamente
    return redirect(url_for('index'))  # Redirigir a la página principal después de resetear el historial

# Página de error 404 personalizada (si es necesario)
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
