import os
import gdown
import torch
from flask import Flask, request, jsonify
from model.u2net import U2NET  # تأكد من أن هذا هو المسار الصحيح للملف

app = Flask(__name__)

# مسار حفظ النموذج
model_dir = "./saved_models/u2net"
model_path = os.path.join(model_dir, "u2net.pth")

# رابط Google Drive الكامل للنموذج
gdrive_url = "https://drive.google.com/uc?id=1TkSXyOSimHA4uMBP_vJWG_T7wdUgKaeV"
def download_model_if_needed():
    if not os.path.exists(model_path):
        print("[INFO] Model not found. Downloading from Google Drive...")
        os.makedirs(model_dir, exist_ok=True)
        gdown.download(gdrive_url, output=model_path, quiet=False)
    else:
        print("[INFO] Model already exists. Skipping download.")

# تحميل النموذج عند بدء التطبيق
download_model_if_needed()

# تحميل النموذج إلى الذاكرة
net = U2NET(3, 1)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval()

@app.route("/", methods=["GET"])
def home():
    return "U2Net is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
