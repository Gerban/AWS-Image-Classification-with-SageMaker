# Image Classification using AWS SageMaker

AWS Sagemaker was used to train a pretrained model that can perform image classification 
by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML
engineering practices. The dog breed dataset was used as a classification task.

- A pre-trained Resnet50 model was adapted for the task by adding additional dense layers.
- Hyperparameters were tuned to find the best hyperparameter combination.
- The model was trained, where debugging with hooks and profiling was also used.
- The model was deployed to an endpoints with some inferences carried out.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Kernel, Python versions and main code files
PyTorch Kernel 1.13.0, Python 3.9, ml.t3.medium

- train_and_deploy_AG.ipynb: the main notebook from which tuning, training and endpoint deployment is triggered
- hpo_AG.py: script that contains functions for hyperparameter tuning
- train_model_AG.py: scrit that contains functions for model training

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
Dataset used: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
Number of image classes: 133 (currently set to 20 for testing)

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
Data uploaded to S3 bucket: s3://sagemaker-us-east-1-.../data/dogImages

## Hyperparameter Tuning and Training
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
A multi-class image classification model was used to classify dog images. 
A CNN based model (Resnet50) was used as a base, with additional dense layers added for transfer learning.

Tuned hyperparameters were (provisional):
- epochs: 5
- learning rate (lr) - range: 0.01 - 0.1
- batch_size - range: 32, 64

Best hyperparameters were: epochs = 5, lr = 0.7, batch_size = 32

Screenshots (to be added):
![HPO_tuning_jos] (screenshots/hpo_training_jobs.png)
![Training_jobs] (screenhots/training_jobs.png)
![Metrics_log] (screenshots/metrics_log.png)


## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
Rules for overfitting, overtraining, vanishing gradient, poor weight initialisation, and loss not decreasing
were added. Profiler rules for low GPU Utilisation and for a Profiler Report were added using methods 
from sagemaker.debugger.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
A number of initial bugs had to be sorted out, such as a mis-placed comma in the data loader function.
Log files on CouldWatch were helpful in assisting to track down further bugs. 

Results from debugger: 

Results from profiler:


**TODO** Remember to provide the profiler html/pdf file in your submission.
'add file to repo'

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
Model endpoint name: to be added for final model

The model endpoint takes a random image from test-folder (check syntax) as input.
Calling the predictor.predict method then generates an inference for the image.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![Model_endpoint] (screenshots/model_endpoint.png)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
