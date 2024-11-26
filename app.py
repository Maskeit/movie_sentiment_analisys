from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import joblib

# Inicializar aplicación Flask
app = Flask(__name__)

# Carga de modelo y vectorizador
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

class ReviewForm(Form):
    review = TextAreaField('', [validators.DataRequired()])

# Define ruta raíz o página de inicio
@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('form.html', form=form)

@app.route('/result', methods=['POST'])
def result():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        # obtener texto ingresado por el usuario
        user_review = request.form['review']

        # Preprocesar texto y hacer prediccion
        review_tfidf = tfidf.transform([user_review]) # vectorizar texto
        prediction = model.predict(review_tfidf)[0] # predecir sentimiento
        sentiment = 'Positive' if prediction == 1 else 'Negative'

        # Pasar resultado a la plantilla
        return render_template('result.html', review=user_review, sentiment=sentiment)
    return render_template('form.html', form=form)

# Ejecuta la aplicación en modo depuración para ver errores con detalle
if __name__ == '__main__':
    app.run(debug=True)
