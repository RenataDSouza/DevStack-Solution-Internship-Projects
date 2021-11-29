from flask import Flask, render_template, request, jsonify
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle

app = Flask(__name__, template_folder='Templates')
ps = PorterStemmer()
stop = set(stopwords.words('english'))

# Load model and vectorizer
pac_model = pickle.load(open('model_p1.pkl', 'rb'))
tf_vect = pickle.load(open('tfidf_vec.pkl', 'rb'))


# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# @app.route('/')
# def home():
#     return render_template('index.html')


def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop]
    review = ' '.join(review)
    review_vect = tf_vect.transform([review]).toarray()
    if pac_model.predict(review_vect)[0] == "FAKE":
        prediction = 'This is a fake news'
    else:
        prediction = 'This is a real news'
    return prediction


@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)


if __name__ == "__main__":
    app.debug = True
    app.run()
