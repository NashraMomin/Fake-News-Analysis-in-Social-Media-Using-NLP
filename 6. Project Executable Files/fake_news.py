import pickle
import flask
from flask import Flask, render_template, request
from flask_cors import CORS
import os
from newspaper import Article
import urllib

app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__,template_folder='templates')
with open('fake_news_model.pkl', 'rb') as handle:
    model = pickle.load(handle)
with open('vectorizer.pkl', 'rb') as handle:
    vectorizer = pickle.load(handle)

@app.route('/')
def main():
    return render_template('main.html')

import logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('url')
        app.logger.debug(f'Text received: {text}')
        
        if not text:
            return render_template('main.html', prediction_text='Invalid text input')
        
        try:
            # Directly use the text as input for prediction
            text_vectorized = vectorizer.transform([text])
            pred = model.predict(text_vectorized)
            app.logger.debug(f'Prediction: {pred[0]}')
            return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))
        except Exception as e:
            app.logger.error(f'Error processing the text: {text} - {e}')
            return render_template('main.html', prediction_text='Failed to process the text')
    return render_template('main.html')




if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
