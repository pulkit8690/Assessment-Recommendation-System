from flask import Flask, request, render_template
import pandas as pd
from main import recommend  
import os
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    model = "gemini"
    results = None
    is_loading = False

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        model = request.form.get("model", "gemini").lower()
        if query:
            is_loading = True
            results_df = recommend(query, engine=model)
            if not results_df.empty:
                results = results_df.to_dict(orient="records")
            is_loading = False

    return render_template(
        "index.html",
        query=query,
        model=model,
        results=results,
        is_loading=is_loading
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

