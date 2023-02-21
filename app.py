import time
from flask import Flask,flash,session, request,redirect,url_for,render_template, send_file, redirect,jsonify
from werkzeug.utils import secure_filename
from flask_session import Session
import os,glob
import chat
from chat import BERT
import sys
sys.path.insert(1, './TTS')


from flask_cors import CORS, cross_origin
# from flask_ngrok import run_with_ngrok
#from flask.ext.cors import CORS, cross_origin

#UPLOAD_FOLDER = 'uploads/audios/'
UPLOAD_CSV_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
# run_with_ngrok(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_CSV_FOLDER'] = UPLOAD_CSV_FOLDER

@app.route('/',methods=["GET"])
#@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def home():
    return render_template('index.html')


@app.route("/chat", methods=["GET","POST"])
def index():
    
    text = request.form['chat_text']
    text = [text]
    text = chat.generate_embeddings(text,tokenizer,model)
    # text = best_model.embed(text)
    # out = best_model(text, 1)
    # print(out)
    CO = loaded_ML_model.predict(text)[0]
    # print(CO)
    allout ={
            "1Chat Opening":int(CO),
            "2Chat Closing":-1,
            "3Appropriate Probing":-1,
            "4Empathy & Apology":-1,
            "5Writing Skills":-1,
            "6Intermediate Response":-1,
            "7Rude & Sarcastic":-1,
            "8Correct Tagging":-1,

        }
    if True:
        return jsonify({'output': allout})
    return 'success'
#This is the main function which creates the flask's HTTPS server
#and runs it on the specified port number: 8888
if __name__ == "__main__":
    sess = Session()
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    cors = CORS(app, resources={r"*": {"origins": "*"}})
    CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['Access-Control-Allow-Origin'] = '*'
    model , tokenizer ,loaded_ML_model = chat.load_opening_w_LR()
    # best_model = BERT().to('cpu')
    # chat.load_opening_w_bert('model.pt', best_model)
    

    sess.init_app(app)
    context = ('certificate.pem', 'privateKey.pem')
    app.run(host="0.0.0.0",debug=True,port=8887, ssl_context='adhoc')
    # app.run()