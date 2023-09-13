import argparse
import torch
from torch import nn,optim
from torchvision import datasets, transforms, models
class train:
    def __init__(self, data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.gpu = gpu
    def data_loader(self):
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        test_dir = self.data_dir + '/test'
        train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        validation_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        testing_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        # Load the datasets with ImageFolder
        train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
        valid_datasets = datasets.ImageFolder(valid_dir,transform=validation_transforms)
        test_datasets = datasets.ImageFolder(test_dir,transform=testing_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        train_loaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
        valid_loaders = torch.utils.data.DataLoader(valid_datasets,batch_size=64)
        test_loaders = torch.utils.data.DataLoader(test_datasets,batch_size=64)
        
        return train_loaders, valid_loaders, test_loaders
    def model_builder(self):
        if self.arch == "vgg19" : 
            model = models.vgg19(pretrained=True)
            classifier_layer = 25088
        elif self.arch == "vgg16" :
            model = models.vgg16(pretrained=True)
            classifier_layer = 25088
        elif self.arch == "densenet121" :
            model = models.densenet121(pretrained=True)
            classifier_layer = 1024
        else :
            raise ValueError("Architecture not supported. Please choose vgg19, vgg16, or densenet121.")
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(nn.Linear(classifier_layer,self.hidden_units),                                 
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(self.hidden_units,102),
                                 nn.LogSoftmax(dim=1))
        return model 
    def model_trainer(self, model, train_loaders, valid_loaders):
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr= self.learning_rate)
        device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training the model on {device} ")
        for epoch in range(self.epochs):
            model.train()  # Set the model in training mode
            running_loss = 0                
            for inputs, labels in train_loaders:  # Use train_loader here
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()                    
            epoch_loss = running_loss / len(train_loaders)
            # Validation
            model.eval()  # Set the model in evaluation mode
            validation_loss = 0
            validation_correct = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loaders:  # Use valid_loader here
                    inputs, labels = inputs.to(device), labels.to(device)
                
                    logps = model.forward(inputs)          
                    validation_loss += criterion(logps, labels).item()
                
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_correct += torch.mean(equals.type(torch.FloatTensor)).item()
                validation_loss /= len(valid_loaders)
                validation_accuracy = validation_correct / len(valid_loaders)
            print(f"Epoch {epoch+1}/{self.epochs}.. "
                  f"Training loss: {epoch_loss:.3f}.. "
                  f"Validation loss: {validation_loss:.4f}.. "
                  f"Validation accuracy: {validation_accuracy:.4f}")             
            
        #Save Checkpoint
        model.class_to_idx = train_loaders.dataset.class_to_idx
        checkpoint = {
            "arch" : self.arch,
            "classifier" : model.classifier,
            "state_dict" : model.state_dict(),
            "class_to_idx" : model.class_to_idx,
            "hidden_units" : self.hidden_units
        }
        torch.save(checkpoint,self.save_dir + '/checkpoint2.pth')
        print("Model Checkpoint Saved Successfully ")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train a neural network on dataset")
    
    parser.add_argument("data_dir", type = str, help = "path to the data directory")
    parser.add_argument("--save_dir", type = str, default="/home/workspace/ImageClassifier", help = "path to save model checkpoint")
    parser.add_argument("--arch", type = str, default = "vgg19", help = "choose Architecture vgg19, vgg16, or densenet121")
    parser.add_argument("--learning_rate", type = float , default = 0.001 , help = "Learning Rate")
    parser.add_argument("--hidden_units", type = int , default = 512, help = "number of hidden units in the classifier")
    parser.add_argument("--epochs", type = int, default = 5, help = "number of epochs")
    parser.add_argument("--gpu", action = "store_true", help = "use gpu for training")
    
    args = parser.parse_args()
    trainer = train(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
    
    train_loaders, valid_loaders, test_loaders = trainer.data_loader()
    model = trainer.model_builder()
    trainer.model_trainer(model, train_loaders, valid_loaders)
        
             
    