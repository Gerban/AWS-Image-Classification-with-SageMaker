#TODO: Import your dependencies. - ToDo: there may be more dependencies to add
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook

# Python Imaging Library: default format is png
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

#TODO: Import dependencies for Debugging andd Profiling
hook = get_hook(create_if_not_exists=True)

# Add logging details - current logging level set to DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test(model, test_loader, loss_criterion, device, hook):
    """
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuracy/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    """
    logger.info('Testing is starting...')

    if hook:
        hook.set_mode(modes.EVAL)
    
    model.to(device)
    model.eval()

    loss_test = 0
    correct_test = 0

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            outputs = model(images)
            loss = loss_criterion(outputs, target)
           
            _, pred_test = torch.max(outputs, 1)
            loss_test += loss.item() * images.size(0)
            correct_test += torch.sum(pred_test == target.data)

        total_loss_test = loss_test / len(test_loader.dataset)
        accuracy_test = correct_test.double() / len(test_loader.dataset)

        print(f'prediction: {pred_test}')
        print(f'actual value: {target.data}')
        print(f'correct: {correct_test.double()}')
        print(f'size of dataset: {len(test_loader.dataset)}')
        print(f'total loss test: {total_loss_test}')
        print(f'total test accuracy: {accuracy_test}')

        logger.info(f'Testing Loss: {total_loss_test}')
        logger.info(f'Testing Accuracy: {accuracy_test}')

    logger.info('Testing is now complete.')

    return accuracy_test, total_loss_test


def train(model, train_loader, valid_loader, epochs, loss_criterion, optimizer, device, hook):
    """
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    """
   
    model.to(device)

    for epoch in range(epochs):

        model.train()
        hook.set_mode(smd.modes.TRAIN)

        loss_train = 0.0
        correct_train = 0

        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)
            outputs = model(images)
            loss = loss_criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_train = torch.max(outputs, 1)
            loss_train += loss.item() * images.size(0)
            correct_train += torch.sum(pred_train == target.data)

        total_loss_train = loss_train / len(train_loader.dataset)
        accuracy_train = correct_train.double() / len(train_loader.dataset)

        print(f'correct: {correct_train.double()}')
        print(f'size of train dataset: {len(train_loader.dataset)}')
        print(f'total loss train: {total_loss_train}')
        print(f'total train accuracy: {accuracy_train}')
        
        logger.info(f'For Epoch {epoch+1}, Train loss is: {total_loss_train: .3f}, Train accuracy is: {accuracy_train}')

        model.eval()
        hook.set_mode(smd.modes.EVAL)
        loss_eval = 0.0
        correct_eval = 0

        with torch.no_grad():
            for images, target in valid_loader:
                images = images.to(device)
                target = target.to(device)
                
                outputs = model(images)
                loss = loss_criterion(outputs, target)
                
                _, pred_eval = torch.max(outputs, 1)
                loss_eval += loss.item() * images.size(0)
                correct_eval += torch.sum(pred_eval == target.data)

        total_loss_eval = loss_eval / len(valid_loader.dataset)
        accuracy_eval = correct_eval.double() / len(valid_loader.dataset)

        logger.info(f'For Epoch {epoch + 1}, Validation loss is: {total_loss_eval: .3f}, Validation accuracy is: {accuracy_eval}')

        print(f'correct: {correct_eval.double()}')
        print(f'size of eval dataset: {len(valid_loader.dataset)}')
        print(f'total loss eval: {total_loss_eval}')
        print(f'total eval accuracy: {accuracy_eval}')

    logger.info('Training has completed.')

    return model


def net(class_count):
    """
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    """
    logger.info('Initialize model fusing pre-trained model ResNet34')
    model = models.resnet34(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    
    logger.info(f'number of linear layer input features are: {num_features}')

    ## Add a fully connected layer to adapt the pre-trained model to the dog breed image classes
    model.fc = nn.Linear(num_features, class_count)

    return model


def create_data_loaders(data, batch_size):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """
    logger.info('Setting up data loaders...')

    train_data_path = os.path.join(data, 'train')
    valid_data_path = os.path.join(data, 'valid')
    test_data_path = os.path.join(data, 'test')
    
    # Transform dataset first - resize, crop and normalize, then load train, valid, test data
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_data_path, transform=transform_train)
    valid_dataset = torchvision.datasets.ImageFolder(valid_data_path, transform=transform_test)
    test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    logger.info('Main code starting')
    logger.info(f'epochs:{args.epochs}')
    logger.info(f'batch_size:{args.batch_size}')

    # Use GPU unit if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running model training on device {device}')

    class_count = 133
    model = net(class_count)
    model = model.to(device)
  
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    # Import data using data loader function
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=class_count)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, eps=args.eps)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info('Start model training...')
    model = train(model, train_loader, valid_loader, args.epochs, loss_criterion, optimizer, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info('Testing model to check accuracy...')

    test(model, test_loader, loss_criterion, device, hook)

    '''
    TODO: Save the trained model
    '''
    logger.info('Saving trained model...')
   
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--batch_size', type=int, default=256, help='Add batch size for training (default is: 256)')
    parser.add_argument('--epochs', type=int, default=10, help='Add number of epochs to train (default is: 10)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='Add learning rate (default is: 0.003)')
    parser.add_argument('--eps', type=float, default=0.000001, metavar='EPS')
   
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args = parser.parse_args()
    
    main(args)
