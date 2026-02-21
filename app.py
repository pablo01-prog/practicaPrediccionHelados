from flask import Flask,request,jsonify
import pickle
import os

# importar el modelo
with open ("datos_helados.pkl","rb") as f:
    modelo = pickle.load(f)


app = Flask(__name__)

# definir los endpoints

# endpoint Inicio
@app.route("/", methods = ["GET"])
def inicio():
    return jsonify({
        "modelo": "Modelo de prediccion de helados y temperatura",
        "metodo": "post",
        "ejemplo de entrada": {
            "temperatura":28
        }
    })

# endpoint Health
@app.route("/predict", methods = ["GET"])
def health():
    return jsonify({
        "status":"ok"
    })

# endpoint Prediccion

@app.route("/predict", methods = ["POST"])
def predict():
    # extraer datos
    datos = request.get_json()
    # validar los datos
    if not datos or "temperatura" not in datos:
        return jsonify({
            "mensaje":"Error, el formato temperatura debe ser correcto"
        }),400 # tipo de error

    # la temperatura esta en texto, hay que convertira a numero
    temperatura = float(datos["temperatura"])

    # prediccion, solo hay una, por eso es [0]
    prediccion = modelo.predict(temperatura)[0]
    
    # devolver la prediccion
    return jsonify({
        "temperatura":temperatura,
        "prediccion":prediccion
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT","5000"))
    app.run(host="0.0.0.0", port = port)