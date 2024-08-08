from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import re
from flask import Flask, render_template, request, jsonify, json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import pickle
import langdetect
from werkzeug.utils import secure_filename
import os
import docx
import fitz  # PyMuPDF
# import PyPDF2

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_name = "surajsahu/My_AI_model"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
model_bert = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Load the Bidirectional LSTM model for Hindi
bi_lstm_model = tf.keras.models.load_model('bidirectional_lstm_model.h5')
tfidf = joblib.load('tfidf_transformer.pkl')

# Load tokenizer from file for Hindi
tokenizer_file = 'tokenizer.pkl'
with open(tokenizer_file, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = 5000
embedding_dim = 128
max_length = 200
trunc_type = 'post'
padding_type = 'post'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    """Cleans English text for input to the BERT model."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()  
    return text

def clean_text_hindi(text):
    """Cleans Hindi text by removing non-Hindi characters and extra spaces."""
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def preprocess_input_hindi(text):
    """Preprocesses Hindi text for input to the Bidirectional LSTM."""
    text = clean_text_hindi(text)
    text_tfidf = tfidf.transform([text])
    return text, text_tfidf

def classify_text_english(text):
    """Predicts whether English text is written by AI or human."""
  
    cleaned_text = clean_text(text)
    
   
    inputs = tokenizer_bert(
        cleaned_text,  
        return_tensors="tf",
        truncation=True,
        padding=True
    )
    
   
    outputs = model_bert(inputs)
    logits = outputs.logits
    
    
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
    confidence = probabilities[0][predicted_class].numpy()
    
    return "Human" if predicted_class == 0 and confidence > 0.6 else "AI"

def predict_text_class(text):
    
    cleaned_text, text_tfidf = preprocess_input_hindi(text)

    text_sequence = tokenizer.texts_to_sequences([cleaned_text])
    text_padded = pad_sequences(text_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    bi_lstm_pred = (bi_lstm_model.predict(text_padded) > 0.5).astype("int32")

    return 'AI' if bi_lstm_pred[0] == 1 else 'Human'

def is_hindi_text(text):
    """Detects if the text is in Hindi."""
    try:
        lang = langdetect.detect(text)
        return lang == 'hi'
    except Exception as e:
        print(f"Language detection error: {e}")
        return False


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

def extract_text_from_word(docx_file):
    """Extracts text from a Word document."""
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def classify_text_deep(text):
    """Predicts whether English text is written by AI or human."""
  
    cleaned_text = clean_text(text)
    
   
    inputs = tokenizer_bert(
        cleaned_text,  
        return_tensors="tf",
        truncation=True,
        padding=True
    )
    
   
    outputs = model_bert(inputs)
    logits = outputs.logits
    
    
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
    confidence = probabilities[0][predicted_class].numpy()
    
    return "AI" if predicted_class == 0 and confidence > 0.6 else "Human"



@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('Index.html', result=None, error=None)


@app.route('/scan', methods=['POST'])
def scan():
    text = request.form.get('textInput', '').strip()
    language = request.form.get('languageSelect')
    file = request.files.get('uploadFile')

    if file and file.filename != '':
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_word(file)
        else:
            return "Unsupported file format", 400

    if not text:
        error = "Please provide some text or upload a file."
        return render_template('Index.html', result=None, error=error)

    if language == 'hindi':
        if is_hindi_text(text):
            result = predict_text_class(text)
        else:
            error = "The provided text is not in Hindi. Please enter Hindi text."
            return render_template('Index.html', result=None, error=error)
    elif language == 'english':
        if not is_hindi_text(text):
            result = classify_text_english(text)
        else:
            error = "The provided text is not in English. Please enter English text."
            return render_template('Index.html', result=None, error=error)
    else:
        error = "Unsupported language selected."
        return render_template('Index.html', result=None, error=error)

    return render_template('Index.html', result=result, error=None)



def analyze_text(text):
    """
    Analyzes the input text and calculates the percentage of AI-generated and human-generated content.
    """
    
    segments = text.split("\n") 
    
    
    ai_count = 0
    human_count = 0
    highlighted_text = []
    
    
    for segment in segments:
        if is_hindi_text(segment):
            prediction = predict_text_class(segment)
        else:
            prediction = classify_text_deep(segment)
        
        if prediction == "AI":
            highlighted_text.append(f'<span style="color: green;">{segment}</span>')
            ai_count += 1
        else:
            highlighted_text.append(f'<span style="color: red;">{segment}</span>')
            human_count += 1
            
    highlighted_text = ' '.join(highlighted_text)
    
    
    total_segments = len(segments)
    ai_percentage = (ai_count / total_segments) * 100
    human_percentage = (human_count / total_segments) * 100
    
    
    chart_data = {
        "labels": ["AI-generated", "Human-written"],
        "datasets": [{
            "data": [ai_percentage, human_percentage],
            "backgroundColor": ["#FF6384", "#36A2EB"]
        }]
    }
    
    return ai_percentage, human_percentage, highlighted_text, chart_data



@app.route('/deep_scan', methods=['POST'])
def deep_scan():
    text = request.form.get('textInput', '').strip()
    language = request.form.get('languageSelect')
    file = request.files.get('uploadFile')

    if file and file.filename != '':
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_word(file)
        else:
            return "Unsupported file format", 400

    if not text:
        error = "Please provide some text or upload a file."
        return render_template('Index.html', result=None, error=error)

    ai_percentage, human_percentage, highlighted_text, chart_data = analyze_text(text)

    return render_template('result.html', 
                           ai_percentage=ai_percentage, 
                           human_percentage=human_percentage, 
                           highlighted_text=highlighted_text,
                           chart_data=json.dumps(chart_data))

if __name__ == '__main__':
    app.run(port=8080)
