# ...existing code...
import os
from flask import Flask, render_template, request, url_for, session, redirect, flash
from werkzeug.utils import secure_filename

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from configparser import ConfigParser
import time

# Config Parser
config = ConfigParser()
config.read("config.ini")
genai.configure(api_key=config["Gemini"]["API_KEY"])

UPLOAD_FOLDER = "static/data"
ALLOWED_EXTENSIONS = set(["mp4","mov","avi","webm","wmv","3gp","flv","mpg","mpeg"])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 新增，session 需要
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB 檔案大小限制

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    },
    system_instruction="請用繁體中文回答以下問題。",
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    print("Submit!")
    if "file1" not in request.files:
        print("No file part")
        return render_template("index.html", prediction="請選擇檔案")
    file = request.files["file1"]
    if file.filename == "":
        print("No selected file")
        return render_template("index.html", prediction="請選擇檔案")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        print(filename)
        try:
            video_file_gemini = upload_to_gemini(filename)
            session["video_file_gemini"] = video_file_gemini.name  # 只存檔名
            result = "檔案已上傳成功! 並提供給Gemini處理完畢. 可以開始問問題囉!"
        except Exception as e:
            print(f"上傳或處理失敗: {e}")
            return render_template("index.html", prediction="檔案處理失敗，請稍後再試")
        return render_template(
            "index.html",
            prediction=result,
            filename=filename,
        )
    else:
        return render_template("index.html", prediction="檔案格式不支援")

@app.route("/call_gemini", methods=["POST"])
def call_gemini():
    if "video_file_gemini" not in session:
        return "請先上傳影片檔案", 400
    video_file_name = session["video_file_gemini"]
    try:
        video_file_gemini = genai.get_file(video_file_name)
    except Exception as e:
        print(f"取得 Gemini 檔案失敗: {e}")
        return "影片檔案取得失敗，請重新上傳", 400
    prompt = request.form.get("message", "")
    if not prompt:
        return "請輸入問題", 400
    try:
        response = gemini_model.generate_content(
            [prompt, video_file_gemini], request_options={"timeout": 600}
        )
        print(response)
        return response.text
    except Exception as e:
        print(f"Gemini 回應失敗: {e}")
        return "Gemini 回應失敗，請稍後再試", 500

@app.errorhandler(413)
def file_too_large(e):
    return render_template("index.html", prediction="檔案太大，請上傳小於200MB的檔案"), 413

def upload_to_gemini(filename):
    print(f"Uploading file...")
    video_file = genai.upload_file(path=f"static/data/{filename}")
    print(f"Completed upload: {video_file}")
    while video_file.state.name == "PROCESSING":
        print("Waiting for video to be processed.")
        time.sleep(1)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    print(f"Video processing complete: " + video_file.uri)
    return video_file

if __name__ == "__main__":
    app.run()