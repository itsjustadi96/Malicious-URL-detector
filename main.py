from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))
    if 'com' in total_Tokens:
        total_Tokens.remove('com')
    return total_Tokens

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    logit, vectorizer = pickle.load(model_file)

# Function to make predictions
def make_predictions(url):
    X_predict = [url]
    X_predict = vectorizer.transform(X_predict)
    return logit.predict(X_predict)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check')
def check():
    return render_template('index.html')

@app.route('/pred')
def pred():
  return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    prediction = make_predictions(url)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)




    