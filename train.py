import argparse
import data_process as dp
import model_class as mc

import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str)

args = parser.parse_args()

# Load the datasets with ImageFolder
image_datasets = { x: datasets.ImageFolder(args.data_dir+'/'+x, transform=t) for x, t in dp.data_transforms.items()}
# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) 
                for x, t in dp.data_transforms.items()}

# model transfered from densenet
model = models.densenet121(pretrained = True)

# frozen parameters
for param in model.parameters():
    param.requires_grad = False

# build the classifier
drop_p = 0.2
classifier = mc.Network(1024, 102, [256], drop_p=drop_p)
model.classifier = classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# train the model
epochs = 3
steps = 0
print_every = 5
running_loss = 0

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

for epoch in range(epochs):
    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        lg_probs = model.forward(inputs)
        loss = criterion(lg_probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    lg_probs = model.forward(inputs)
                    loss = criterion(lg_probs, labels)

                    valid_loss += loss.item()

                    probs = torch.exp(lg_probs)
                    top_p, top_index = probs.topk(1, dim=1)
                    results = top_index == labels.view(*top_index.shape)
                    accuracy += torch.mean(results.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.."
                  f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                  f"Valid Accuracy: {accuracy/len(dataloaders['valid']):.3f}")
            running_loss = 0
            model.train()
    
# Do validation on the test
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        lg_probs = model.forward(inputs)
        test_loss += criterion(lg_probs, labels).item()
    
        probs = torch.exp(lg_probs)
        top_value, top_indice = probs.topk(1, dim=1)
        results = top_indice == labels.view(*top_indice.shape)
        accuracy += torch.mean(results.type(torch.FloatTensor)).item()
model.train()

print(f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
      f"Test Accuracy: {accuracy/len(dataloaders['test']):.3f}")

# save checkpoint
# model.to("cpu")

checkpoint = {'input_size': model.classifier.hidden_layers[0].in_features,
              'output_size': model.classifier.output.out_features,
              'epochs': epochs,
              'drop_p': model.classifier.dropout.p,
              'hidden_layers': [e.out_features for e in model.classifier.hidden_layers],
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': image_datasets['test'].class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')