from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    # Example placeholder for prediction logic
    symptoms = request.form.get("symptoms")
    return f"You entered the following symptoms: {symptoms}"


if __name__ == "__main__":
    app.run(debug=True)
