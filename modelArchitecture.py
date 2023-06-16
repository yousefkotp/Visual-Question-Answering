import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class VQAModel(nn.Module):

    def __init__(self, num_classes, hidden_size, model_name = "RN50x64", device = torch.device("cpu")):
        super(VQAModel, self).__init__()

        self.training_losses = []
        self.validation_losses = []
        self.device = device
        self.model_name = model_name
        self.clip_model, self.preprocess = clip.load(model_name, device = device)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # now let's add our own layers
        self.linear_layer1 = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim + self.clip_model.text_projection.shape[1], hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.5)
        )

        self.linear_layer2 = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.5)
        )

        self.answer_type_layer = nn.Linear(hidden_size, 4)
        self.answer_mask_layer = nn.Linear(4, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, question):
        
        features = torch.cat((image, question), dim=1) # Concatenate image and text features
        features = self.linear_layer1(features)

        answer_type = self.answer_type_layer(features)
        final_answer_type = self.softmax(answer_type)

        answer_mask = self.answer_mask_layer(answer_type)
        answer_mask = self.sigmoid(answer_mask)

        output = self.linear_layer2(features)

        output = output * answer_mask

        output = self.softmax(output)

        return output, final_answer_type
    
    def train_model(self, training_dataloader, validation_dataloader, criterion, optimizer, epochs = 10, save_path = None, save_every = 1):
        for epoch in range(1,epochs+1):
            training_loss = self.training_step(training_dataloader, criterion, optimizer, self.device)
            validation_loss = self.validation_step(validation_dataloader, criterion, self.device)

            self.training_losses.append(training_loss)
            self.validation_losses.append(validation_loss)

            print("Epoch: {} | Training Loss: {} | Validation Loss: {}".format(epoch, training_loss, validation_loss))

            if save_path != None and epoch % save_every == 0:
                self.save_model(save_path + "epoch_{}.pth".format(epoch))
            
        return self.training_losses, self.validation_losses


    def training_step(self, dataloader, criterion, optimizer, device):
        training_loss = 0.0
        self.train()
        for _, batch in enumerate(dataloader):
            image, question, answer, answer_type = batch
            image, question, answer, answer_type = image.to(device), question.to(device), answer.to(device), answer_type.to(device)

            optimizer.zero_grad()
            output, answer_type_predicted = self.forward(image, question)

            loss = criterion(output, answer) + criterion(answer_type, answer_type_predicted)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(dataloader)
        return training_loss
            
    
    def validation_step(self, dataloader, criterion, device):
        validation_loss = 0.0
        self.eval()
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                image, question, answer, answer_type = batch
                image, question, answer, answer_type = image.to(device), question.to(device), answer.to(device), answer_type.to(device)
                output, answer_type_predicted = self.forward(image, question)
                loss = criterion(output, output) + criterion(answer_type, answer_type_predicted)
                validation_loss += loss.item()
        validation_loss /= len(dataloader)
        return validation_loss                

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
    
    def predict(self, image, question):
        output, answer_type = self.forward(image, question)
        predicted_answer = torch.argmax(output, dim = 1)
        predicted_answer_type = torch.argmax(answer_type, dim = 1)
        return predicted_answer, predicted_answer_type
    
    def plot_history(self):
        plt.plot(self.training_losses, label = "Training Loss")
        plt.plot(self.validation_losses, label = "Validation Loss")
        plt.legend()
        plt.show()

    def test_model(self, image_path, question):
        image = Image.open(image_path)
        predicted_answer, predicted_answer_type = self.predict(image, question)
        print("Predicted Answer:", predicted_answer.item())
        print("Predicted Answer Type:", predicted_answer_type.item())

    def print_CLIP_model(self):

        input_resolution = self.clip_model.visual.input_resolution
        context_length = self.clip_model.context_length
        vocab_size = self.clip_model.vocab_size

        print("Selected model:", self.model_name)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.clip_model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

'''
dataset = MyVQADataset(...)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = VQAModel(num_classes, model_name, device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train_model(dataloader, criterion, optimizer, epochs = 10, save_path = "models/", save_every = 1)
model.plot_history()


model = VQAModel(10, "RN50x64", "cpu")
model.print_CLIP_model()
print(clip.available_models())

'''