import google.generativeai as genai

# Set API key
genai.configure(api_key="AIzaSyC0HS2orQLKlwgWCG3jUnC5scfHwuJVoBg")

# List available models
models = genai.list_models()

for model in models:
    print(f"\n🔹 Model ID: {model.name}")
    print(f" - Description: {model.description}")
    print(f" - Input Token Limit: {model.input_token_limit}")
    print(f" - Output Token Limit: {model.output_token_limit}")
    print(f" - Supported Generation Methods: {model.supported_generation_methods}")


# .\.venv\Scripts\Activate.ps1
