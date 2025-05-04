import os
import logging
import math
from logging.handlers import RotatingFileHandler
from flask import Flask, request, render_template, jsonify, current_app, make_response
from flask_cors import CORS
from main import recommend, get_keys_and_configure

# ——— Flask App Setup ———
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load configuration from environment
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1')
app.config['HOST'] = os.getenv('FLASK_HOST', '0.0.0.0')
app.config['PORT'] = int(os.getenv('FLASK_PORT', '5000'))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))

# Enable CORS if needed
CORS(app)

# ——— Logging ———
log_level = logging.DEBUG if app.config['DEBUG'] else logging.INFO
logging.basicConfig(level=log_level)
app.logger.setLevel(log_level)
if not app.config['DEBUG']:
    os.makedirs('logs', exist_ok=True)
    handler = RotatingFileHandler('logs/app.log', maxBytes=100_000, backupCount=10)
    handler.setLevel(log_level)
    fmt = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')
    handler.setFormatter(fmt)
    app.logger.addHandler(handler)

# ——— Load API keys ———
try:
    GEMINI_KEY, TOGETHER_KEY = get_keys_and_configure()
    app.logger.info("Successfully loaded API keys")
except Exception as e:
    app.logger.error(f"Failed to load API keys: {e}")

# Supported engines
VALID_ENGINES = {'gemini', 'together'}

# ——— Health & Error Handlers ———
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='ok'), 200

@app.errorhandler(404)
def handle_404(err):
    return render_template('404.html'), 404

@app.errorhandler(500)
def handle_500(err):
    app.logger.exception('Internal server error')
    return jsonify({'error': 'An internal error occurred.'}), 500

# ——— Routes ———
@app.route('/', methods=['GET'])
def home():
    dark_mode = request.cookies.get('dark_mode', 'false') == 'true'
    return render_template(
        'index.html',
        query='', model='gemini', dark_mode=dark_mode
    )

@app.route('/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    engine = (data.get('model') or 'gemini').lower()

    if not query:
        return jsonify({'results': [], 'error': 'Provide a non-empty query.'}), 400
    if engine not in VALID_ENGINES:
        return jsonify({'results': [], 'error': f"Unsupported engine '{engine}'"}), 400

    try:
        df = recommend(query, engine=engine)
        # Convert DataFrame to JSON-friendly records, replacing NaN with None and adding adaptive_support
        records = []
        for rec in df.to_dict(orient='records'):
            # Replace NaN
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
            # Include adaptive support flag
            rec['adaptive_support'] = rec.get('adaptive')
            records.append(rec)
        return jsonify({'results': records, 'error': None}), 200
    except Exception:
        app.logger.exception('Recommendation failure')
        return jsonify({'results': [], 'error': 'Internal processing error.'}), 500

@app.route('/toggle-dark', methods=['POST'])
def toggle_dark():
    data = request.get_json() or {}
    mode = 'true' if data.get('dark_mode') else 'false'
    resp = make_response(jsonify({'dark_mode': mode}))
    resp.set_cookie('dark_mode', mode, max_age=30*24*3600)
    return resp

# ——— Run ———
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))  # Replit expects port 3000
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
