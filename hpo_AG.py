#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import logging
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# Python Imaging Library: default format is png
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set log level etc
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, loss_criterion, device):
    """
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    """
    logger.info('Testing is starting...')

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
    
        logger.info(f'Testing Loss: {total_loss_test}')
        logger.info(f'Testing Accuracy: {accuracy_test}')

        print(f'total loss test: {total_loss_test}')
        print(f'total test accuracy: {accuracy_test}')

    logger.info('Testing is now complete.')

    return accuracy_test, total_loss_test


def train(model, train_loader, valid_loader, epochs, loss_criterion, optimizer, device):
    """
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    """
    model.to(device)

    for epoch in range(epochs):

        model.train()
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

        print(f'total loss train: {total_loss_train}')
        print(f'total train accuracy: {accuracy_train}')
    
        logger.info(f'For Epoch {epoch+1}, Train loss is: {total_loss_train: .3f}, Train accuracy is: {accuracy_train}')

        model.eval()
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

        print(f'total loss eval: {total_loss_eval}')
        print(f'total eval accuracy: {accuracy_eval}')

        logger.info(f'For Epoch {epoch + 1}, Validation loss is: {total_loss_eval: .3f}, Validation accuracy is: {accuracy_eval}')

    logger.info('Training has completed.')
    
    return model


def net(class_count):
    """
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    param: class_count: integer - the number of output classes of the image classification
    """
    model = models.resnet34(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    logger.info(f'number of linear layer input features are: {num_features}')

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
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform_train)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_data_path, transform=transform_test)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


def main(args):
    """
    TODO: Initialize a model by calling the net function
    """
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data_dir}')

    # Use GPU unit if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running model training on device {device}')

    class_count = 133
    model = net(class_count)
    model = model.to(device)
   
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
    model = train(model, train_loader, valid_loader, args.epochs, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info('Testing model...')
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info('Saving model...')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info('Model has been saved, hpo code has finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument('--batch_size', type=int, default=256, help='Add batch size for training (default is: 256)')
    parser.add_argument('--epochs', type=int, default=10, help='Add number of epochs to train (default is: 10)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='Add learning rate (default is: 0.003)')
    parser.add_argument('--eps', type=float, default=0.000001, metavar='EPS')

    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args = parser.parse_args()
    print(args)
    
    main(args)
