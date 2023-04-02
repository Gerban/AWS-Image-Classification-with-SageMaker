## Project 3: Image Classification using AWS SageMaker

AWS Sagemaker was used to use transfer learning to train a pretrained model (Resnet50) 
to predict which dog breed a dog image is. The input data consisted of 133 classes of dog images.
Aside from hyper-parameter tuning Sagemaker profiler and debugger were used during the training process.
Training and evaluation losses were plotted and some recommendations on model improvements provided.
The model was deployed at an endpoint with a couple of inference examples carried out for testing. Process:

- A pre-trained Resnet50 model was adapted for dog image classification by adding additional dense layers.
- Hyper parameters were tuned to find the best hyper parameter combination.
- The model was trained, where Sagemaker debugger and profiler were used with their outputs recorded.
- The model was deployed to an endpoint and a number of inferences were carried out on test images.

## Kernel, Python versions and main code files
Kernel:  PyTorch 1.13.0 with Python 3.9 on ml.t3.medium

- ![train_and_deploy_AG.ipynb](train_and_deploy_AG.ipynb) : the main notebook from which tuning, training and endpoint deployment is triggered
- ![hpo_AG.py](hpo_AG.py) : script that contains functions for hyperparameter tuning
- ![train_model_AG.py](train_model_AG.py) : script that contains functions for model training
- ![inferency.py](inference.py) : script that contains code used as entry point for deployed model endpoint 

## Dataset
The input dataset used for the dog image classification is the dataset suggested by the project (number of classes - 133: 
https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data:
Data was uploaded to S3 bucket: s3://sagemaker-us-east-1-308298057408/data/dogImages

## Hyperparameter Tuning and Training
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges 
used for the hyperparameter search:
A multi-class image classification model was used to classify dog images with 133 classes.
A CNN based model (Resnet50) was chosen as a model base, and transfer learning was applied in the form
of the addition of dense layers added to the Resnet50 network with 133 final outputs corresponding to the 133 classes.

For loss optimisation the Adam Optimiser (or MSProp - tbc) was used and CrossEntropyLoss as a loss function.

Tuned hyperparameters were (provisional):
- epochs: 5, 10
- learning rate (lr) - range: 0.001 - 0.1
- batch_size - range: 64, 128, 256

Best hyperparameters were: epochs = 5, lr = 0.01, batch_size = 128

Links to screenshots for hyper parameter tuning and training jobs:
![HPO_tuning_jos](screenshots/hpo_training_jobs.png)
![Training_jobs](screenhots/training_jobs.png)
![Metrics_log](screenshots/metrics_log.png)

Losses for training and validation data are plotted in the graph, showing a gradual decline for both datasets:
**ToDo**: Add:Image of loss function 

## Debugging and Profiling with Results
Sagemaker provides model debugging functionality where one can set rules. The rules for overfitting, overtraining, 
vanishing gradient, poor weight initialisation, and loss not decreasing were added and the debugger logs results.
Logging statements were also added to functions and the output could then be inspected in CloudWatch logs of the corresponding
tuning or training jobs. This is an additional tool that can assist in debugging the code to track down issues. 

Results from Sagemaker Debugger:
**ToDo**: Add results

For computing performance Sagemaker Profiler can be used, which shows system usage statistics, such as low or high CPU, GPU 
utilisation, carries out training loop analysis and also provides a framework metrics summary.  

Results from Sagemaker Profiler:
**ToDo**: add issues

Link to SageMaker profiler report:
![profiler-report.html](profiler-report.html)

## Model Deployment
The model was deployed on a ml.m5.large instance. As an entry point the script inference.py was used.

A sceenshot shows the name of the deployed endpoint:
![deployed_endpoint](screenshorts/deployed_endpoint.png)

The endpoint can be queries by submitting test images as input with it then returning the inferred results:

image_path = './dogImages/test/130.Welsh_springer_spaniel/Welsh_springer_spaniel_08215.jpg'
image_class = "130"

with open(image_path, "rb") as f:
    payload = f.read()
    response=predictor.predict(payload, initial_args={"ContentType": "image/jpeg"})
    pred_class = np.argmax(response, 1) + 1
    print(pred_class)

Note: when code has completed it is important to delete deployed endpoints and to shut down instances no longer used.

