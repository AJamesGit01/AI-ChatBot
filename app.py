import os, time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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
    if len(user_message) > 4000:
        return jsonify({"error": "message too long"}), 413

    # Small retry loop for transient 429s (rate limits)
    attempts = 0
    while True:
        try:
            if PROVIDER == "openai":
                resp = oai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful, concise assistant."},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                reply = resp.choices[0].message.content

            else:  # gemini
                # Simple single-turn prompt; for multi-turn use ChatSession
                prompt = f"User: {user_message}\nAssistant:"
                out = g_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": TEMPERATURE,
                        "max_output_tokens": MAX_TOKENS,
                    }
                )
                reply = (out.text or "").strip()

            if not reply:
                reply = "I couldn't generate a response."
            return jsonify({"reply": reply})

        except Exception as e:
            err = str(e)

            # Quota/rate-limit handling (Gemini/OpenAI)
            # Gemini: 429 RESOURCE_EXHAUSTED; OpenAI: 429 rate limit or insufficient_quota
            if "429" in err or "Rate limit" in err or "RESOURCE_EXHAUSTED" in err:
                attempts += 1
                if attempts > 3:
                    return jsonify({"error": "Rate limit hit repeatedly. Please try again shortly."}), 429
                time.sleep(0.6 * (2 ** attempts))
                continue

            if "insufficient_quota" in err or "quota" in err.lower():
                return jsonify({"error": "Quota exhausted for this API key. Add billing or wait for reset."}), 402

            return jsonify({"error": "Upstream API error. Please try again."}), 502

if __name__ == "__main__":
    import socket
    port = int(os.getenv("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
