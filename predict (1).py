import argparse
import torch
import json
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
class predict:
    def __init__(self, checkpoint_path, category_names_path, topk, gpu):
        self.checkpoint_path = checkpoint_path
        self.topk = topk
        self.gpu = gpu
        self.class_to_name = self.load_category_names(category_names_path)
        self.model = self.model_loader()
        
    def model_loader(self):
        checkpoint = torch.load(self.checkpoint_path)
        if checkpoint["arch"] == "vgg19" : 
            model = models.vgg19(pretrained=True)
            classifier_layer = 25088
        elif checkpoint["arch"] == "vgg16" :
            model = models.vgg16(pretrained=True)
            classifier_layer = 25088
        elif checkpoint["arch"] == "densenet121" :
            model = models.densenet121(pretrained=True)
            classifier_layer = 1024
        model.classifier = nn.Sequential(nn.Linear(classifier_layer, checkpoint["hidden_units"]), 
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(checkpoint["hidden_units"], 102),
                                 nn.LogSoftmax(dim=1))
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model
    def load_category_names(self,category_names_path):
        with open(category_names_path,'r') as f:
            category_names = json.load(f)
        return category_names
    def image_processor(self, image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        return image
    def prob_predictor(self, image_path):
        image = self.image_processor(image_path)
        device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)
        image = image.unsqueeze(0).to(device)
    
        with torch.no_grad():
            output = self.model(image)
            topk_probabilities, topk_indices = output.topk(self.topk)
        idx_to_class = {v: k for k,v in self.model.class_to_idx.items()}
        topk_indices = [idx_to_class[idx.item()] for idx in topk_indices[0]]
        top_class_labels = [self.class_to_name[labels] for labels in topk_indices]
        topk_probabilities = torch.exp(topk_probabilities)
    
        return topk_probabilities, topk_indices, top_class_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict flower type from an image")
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--topk', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='/home/workspace/ImageClassifier/cat_to_name.json', help='Path to the category names JSON file')    
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')     
    args = parser.parse_args()
    predictor = predict(args.checkpoint_path, args.category_names, args.topk, args.gpu)
    probabilities, indices, class_names = predictor.prob_predictor(args.image_path)
    probabilities = probabilities.cpu().numpy()
    
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probabilities[0][i]:.4f}")
