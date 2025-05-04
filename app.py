from flask import Flask, request, render_template
import pandas as pd
from main import recommend  # Ensure this is the correct path to your function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    query = ""
    model = "gemini"  # ✅ initialize model outside POST block

    if request.method == "POST":
        query = request.form.get("query", "")
        model = request.form.get("model", "gemini")
        results_df = recommend(query, engine=model)
        if not results_df.empty:
            results = results_df.to_dict(orient="records")

    return render_template(
        "index.html",
        query=query,
        model=model,
        results=results,
        is_loading=False
    )

if __name__ == "__main__":
    app.run(debug=True)
