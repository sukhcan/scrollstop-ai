# 1_api.py
from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

# Load Model
class ScrollStop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(5, 64, batch_first=True, dropout=0.0)
        self.fc = torch.nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = ScrollStop()
model.load_state_dict(torch.load("scrollstop_model.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    rates = data['rates']
    seq = np.repeat(np.array(rates)[np.newaxis, np.newaxis, :], 10, axis=1)
    with torch.no_grad():
        pred = model(torch.tensor(seq, dtype=torch.float32)).item()
    return jsonify({"stop_second": round(pred, 1)})

@app.route('/')
def home():
    return "<h1>ScrollStop™ AI API LIVE!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)