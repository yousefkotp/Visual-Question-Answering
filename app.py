from flask import Flask, request, jsonify, render_template
import torch
import pickle
from vqa_model import VQAModel
import urllib.request

app = Flask(__name__)

# Loading the fitted One Hot Encoders from the disk
with open('Saved_Models/answer_onehotencoder.pkl', 'rb') as f:
    ANSWER_ONEHOTENCODER = pickle.load(f)
with open('Saved_Models/answer_type_onehotencoder.pkl', 'rb') as f:
    ANSWER_TYPE_ONEHOTENCODER = pickle.load(f)

# Loading the model from the disk
DEVICE = torch.device("cpu")
MODEL_NAME = "ViT-L/14@336px"
NUM_CLASSES = 5410
MODEL_PATH = "Saved_Models/model.pth"
model = VQAModel(num_classes=NUM_CLASSES, device= DEVICE, hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
model.load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image and question from the request
    image_url = request.form.get('image_url')
    question = request.form.get('question')

    if 'image' in request.files:
        # The image is a file uploaded from a device
        image = request.files['image']
        image_path = 'templates/static/user_image.jpg'
        image.save(image_path)
    elif image_url:
        # The image is a URL
        image_path = 'templates/static/user_image.jpg'
        urllib.request.urlretrieve(image_url, image_path)
    else:
        # No image was provided
        return 'No image provided'

    # Predict the answer and answer type
    predicted_answer, predicted_answer_type, answerability = model.test_model(image_path = image_path, question = question)
    answer = ANSWER_ONEHOTENCODER.inverse_transform(predicted_answer.cpu().detach().numpy())
    answer_type = ANSWER_TYPE_ONEHOTENCODER.inverse_transform(predicted_answer_type.cpu().detach().numpy())
    
    # Return the predicted answer and answer type as a JSON response
    response = {'answer': answer[0][0], 'answer_type': answer_type[0][0], 'answerability': answerability.item()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)