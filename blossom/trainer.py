import time
import torch


class Trainer:
    """_summary_
    """

    def __init__(self, model, optimizer, criterion, device='cuda', checkpoint=None, path=None):
        self.classifier = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_losses = []
        self.valid_losses = []
        self.device = device
        self.completed_epochs = 0
        self.save = checkpoint
        self.path = path

    def fit(self, trainds, epochs=20, validation_data=None):
        """_summary_

        Args:
            trainds (_type_): _description_
            epochs (int, optional): _description_. Defaults to 20.
            validation_data (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.classifier.to(self.device)
        for idx in range(epochs):
            train_start = time.time()
            running_loss, running_accuracy = 0, 0
            for images, labels in trainds:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                log_prob = self.classifier(images)
                loss = self.criterion(log_prob, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                #TO DO: convert to a method
                prob = log_prob.exp()
                top_p, top_class = prob.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor))

            self.train_losses.append(running_loss/len(trainds))
            train_stop = time.time() - train_start

            if validation_data:
                valid_start = time.time()
                valid_loss, valid_accuracy = 0, 0
                self.classifier.eval()
                with torch.no_grad():
                    for images, labels in validation_data:
                        images, labels = images.to(
                            self.device), labels.to(self.device)
                        log_prob = self.classifier(images)
                        valid_loss += self.criterion(log_prob, labels)

                        prob = log_prob.exp()
                        top_p, top_class = prob.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(
                            equals.type(torch.FloatTensor))
                self.valid_losses.append(valid_loss/len(validation_data))
                valid_stop = time.time() - valid_start
                
            #TO DO: add model name to checkpoint path
            self.completed_epochs += 1
            if self.save and (idx+1) %15 ==0:
                self.classifier.to('cpu') 
                new_path = self.path + str(idx+1) +'.pth'
                self.check_point(new_path)
                self.classifier.to(self.device) 
            
            # TO DO: fix train logging, specifically when validation data in not given
            print("Epoch: {}/{}..".format(idx+1, epochs),
                  "Train Time:{:.3f}.".format(train_stop/60),
                  "Training Loss:{:.3f}..".format(running_loss/len(trainds)),
                  "Training Accuracy:{:.3f}..".format(
                      running_accuracy/len(trainds)),
                  "Valid Loss:{:.3f}..".format(
                      valid_loss/len(validation_data)),
                  "Valid Time:{:.3f}.".format(valid_stop/60),
                  "Valid Accuracy: {:.3f}".format(valid_accuracy/len(validation_data)))
            self.classifier.train()
        

    def check_point(self, path='checkpoint.pth'):
        """_summary_

        Args:
            path (str, optional): _description_. Defaults to 'checkpoint.pth'.
        """
        info = {'num_classes': self.classifier.num_classes,
                'epoch': self.completed_epochs,
                'model_state_dict': self.classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'valid_losses': self.valid_losses
                }
        torch.save(info, path)


        
    def load_checkpoint(self, path):
        """_summary_
        
        Args:
            path (_type_): _description_
        """
        checkpoint = torch.load(path)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.valid_losses = checkpoint['valid_losses']
        self.completed_epochs = checkpoint['epoch']
        self.classifier.num_classes = checkpoint['num_classes']

    def predict(self, image, top=5):
        """_summary_

        Args:
            image (_type_): _description_
            top (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        log_prob = self.classifier(image)
        prob = log_prob.exp()
        top_p, top_class = prob.topk(top, dim=1)
        return top_p, top_class
