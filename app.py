from flask import Flask, render_template, request
import pickle
cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    # if request.method == "POST":
    email = request.form.get("content")
    tokenized_email = cv.transform([email])
    predictions = clf.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return render_template("index.html", predictions = predictions, email = email)


if __name__ == "__main__":
    app.run(debug=True)
