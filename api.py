from flask import Flask, request, render_template
from application import messageClassification

tc = messageClassification()
app = Flask(__name__)


@app.route("/")
def predict_html():
    return render_template("predict.html")


@app.route("/predict")
def predict():
    message = request.args["message"]

    label = tc.test(message)
    return render_template("result.html", message=label)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
