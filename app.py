
import os
import gdown
import torch
from model.u2net import U2NET
from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import io

# تحميل النموذج تلقائيًا من Google Drive إذا لم يكن موجودًا
def download_model_if_needed():
    model_path = './saved_models/u2net/u2net.pth'
    if not os.path.exists(model_path):
        print("[INFO] Model not found. Downloading from Google Drive...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(id='1pLzdyE2-XXMpVfUbuQp7Vn97wzeJSqyS', output=model_path, quiet=False)
        print("[INFO] Model downloaded and saved to:", model_path)

# بدء تطبيق Flask
app = Flask(__name__)
CORS(app)

# تحميل النموذج
model_path = './saved_models/u2net/u2net.pth'
download_model_if_needed()
net = U2NET(3, 1)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval()

# معالجة الصورة وإزالة الخلفية
def remove_background(input_image):
    image = input_image.convert('RGB')
    im = np.array(image).astype(np.float32)
    im = im / 255.0
    im = im.transpose((2, 0, 1))
    im = torch.from_numpy(im).unsqueeze(0)
    d1, _, _, _, _, _, _ = net(im)
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = pred.squeeze().cpu().data.numpy()
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(image.size)
    empty = Image.new("RGBA", image.size)
    image.putalpha(mask)
    return image

# نقطة النهاية لاستقبال الصور وإرجاع النتيجة
@app.route('/remove_background', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return 'No image file provided', 400
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    result = remove_background(image)
    byte_io = io.BytesIO()
    result.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
