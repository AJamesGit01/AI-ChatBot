import os, time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask import Response
from dotenv import load_dotenv

# --- Load env ---
load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# Common knobs
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
MAX_TOKENS  = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

app = Flask(__name__, static_folder="static")
CORS(app)

# --- Provider init ---
if PROVIDER == "openai":
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider")
    oai = OpenAI(api_key=OPENAI_API_KEY)

elif PROVIDER == "gemini":
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY for Gemini provider")
    genai.configure(api_key=GEMINI_API_KEY)
    g_model = genai.GenerativeModel(GEMINI_MODEL)
else:
    raise RuntimeError(f"Unknown PROVIDER: {PROVIDER}")

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    def generate():
        try:
            if PROVIDER == "openai":
                # STREAMING MODE
                stream = oai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=True
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta.get("content")
                    if delta:
                        yield delta

            else:  # GEMINI STREAMING
                response = g_model.generate_content(
                    user_message,
                    generation_config={
                        "temperature": TEMPERATURE,
                        "max_output_tokens": MAX_TOKENS
                    },
                    stream=True
                )

                for chunk in response:
                    if chunk.text:
                        yield chunk.text

        except Exception as e:
            yield f"[ERROR]: {str(e)}"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    import socket
    port = int(os.getenv("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
