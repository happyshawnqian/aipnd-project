import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import data_process as dp

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()

        self.hidden_layers =nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(nn.Linear(h1, h2) for h1, h2 in layer_sizes)

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    classifier = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'],
                         drop_p=checkpoint['drop_p'])
    #classifier.load_state_dict(checkpoint['model_state_dict'])
    model = models.densenet121(pretrained = False)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier=classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = dp.process_image(image_path)
    image = image.view(1, *image.shape) # create one more dimension to match the input of model

    #image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        lg_probs = model.forward(image)
    model.train()
    probs = torch.exp(lg_probs)
    top_p, top_i = probs.topk(topk)
    top_p = top_p[0].tolist() # top_p[0] returns the first row of the two-dimension tensor 
    top_i = top_i[0].tolist()
    # reverse class_to_idx
    #idx_to_class = {index:label for label,index in train_data.class_to_idx.items()}
    idx_to_class = {index:label for label,index in model.class_to_idx.items()}
    
    labels = [idx_to_class[index] for index in top_i]
    
    return top_p, labels