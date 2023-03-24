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
- inference.py: script that is used as entry-point for deployed endpoint
- profiler-report.html: profiler report with details of training job
- screenshots: folder that stores screenshots submitted as part of project

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
Dataset used: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
Number of image classes: 133 (currently set to 20 for testing)

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
Data uploaded to S3 bucket: s3://sagemaker-us-east-1-.../data/dogImages

## Hyperparameter Tuning and Training
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters
and their ranges used for the hyperparameter search
A multi-class image classification model was used to classify dog images. 
A CNN based model (Resnet50) was used as a base, with additional dense layers added for transfer learning.

Tuned hyperparameters were (provisional):
- epochs: 5
- learning rate (lr) - range: 0.01 - 0.1
- batch_size - range: 32, 64

Best hyperparameters were: epochs = 5, lr = 0.7, batch_size = 32

Screenshots (to be added):

[HPO_tuning_jobs](screenshots/hpo_training_jobs.png)

[Training_jobs](screenhots/training_jobs.png)

[Metrics_log](screenshots/metrics_log.png)

[Endpoint_for_inference](screenshots/Pr3_Endpoint-for-inference.png)


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
The model was deployed using the .deploy() method - currently to run cells in notebook.
Querying the endpoint can be done by e.g. supplying a url with a dog image, such as:
request_dict={ "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/
uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg" }

Then, using the.predict() method this will return an array of probabilities for the image to be part of
each class. Then, np.argmax() returns the class label of the class with highest probability.
In the above example, the dog image label was ...

The model endpoint name was (to be updated): pytorch-inference-2023-03-18-17-39-38-520

