import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import matplotlib.pyplot as plt
import numpy as np


class VQAModel(nn.Module):

    training_losses = []
    validation_losses = []


    def __init__(self, num_classes, model_name = "ViT-B/32", device = "cpu"):
        super(VQAModel, self).__init__()
        self.device = device
        self.model_name = model_name
        self.clip_model, self.preprocess = clip.load(model_name, device = device)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.linear_layer = nn.Linear(512, num_classes)

    def forward(self, image, question):

        with torch.no_grad():

            image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image).float()

            text_tokens = clip.tokenize(question).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens).float()

        features = torch.cat((image_features, text_features), dim=1) # Concatenate image and text features
        output = self.linear_layer(features)
        return output
    
    def train_model(self, dataloader, criterion, optimizer, epochs = 10, device = "cpu", save_path = None, save_every = 1):
        for epoch in range(1,epochs+1):

            training_loss = self.training_step(dataloader, criterion, optimizer, device)
            validation_loss = self.validation_step(dataloader, criterion, device)

            self.training_losses.append(training_loss)
            self.validation_losses.append(validation_loss)

            print("Epoch: {} | Training Loss: {} | Validation Loss: {}".format(epoch, training_loss, validation_loss))

            if save_path != None and epoch % save_every == 0:
                self.save_model(save_path + "epoch_{}.pth".format(epoch))
            
        return self.training_losses, self.validation_losses


    def training_step(self, dataloader, criterion, optimizer, device = "cpu"):
        training_loss = 0.0
        for batch in dataloader:
            images, questions, labels = batch
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = self.forward(images, questions)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(dataloader)
        return training_loss
    
    def validation_step(self, dataloader, criterion, device = "cpu"):
        validation_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                images, questions, labels = batch
                images, questions, labels = images.to(device), questions.to(device), labels.to(device)
                predictions = self.forward(images, questions)
                loss = criterion(predictions, labels)
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
        output = self.forward(image, question)
        _, predicted = torch.max(output.data, 1)
        return predicted
    
    def predict_prob(self, image, question):
        output = self.forward(image, question)
        return F.softmax(output, dim=1)
    
    def plot_history(self):
        plt.plot(self.training_losses, label = "Training Loss")
        plt.plot(self.validation_losses, label = "Validation Loss")
        plt.legend()
        plt.show()

    def print_CLIP_model(self):
        if self.device == "cpu":
            self.clip_model.eval()
        else:
            self.clip_model.cuda().eval()

        input_resolution = self.clip_model.visual.input_resolution
        context_length = self.clip_model.context_length
        vocab_size = self.clip_model.vocab_size

        print("Selected model:", self.model_name)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.clip_model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

        print(self.clip_model)

model = VQAModel(10, "ViT-B/32", "cpu")
model.print_CLIP_model()
print(clip.available_models())


'''
dataset = MyVQADataset(...)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = VQAModel(num_classes, model_name, device)

for batch in dataloader:
    images, questions, labels = batch
    images, questions, labels = images.to(device), questions.to(device), labels.to(device)
    predictions = model.predict(images, questions)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

'''