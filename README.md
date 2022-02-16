# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The dataset "benchmarking_train.csv" contains information about bank customers. There are 10,000 records in the dataset and 21 columns if we should provide a loan to a customer based on the customer's characteristics. Therefore, it is a classification problem.
In this project, I was provided with a custom mode "train.py", which uses the Sklearn library to train a logistic regression. I applied two Azure Machine Learning powerful technics to train the best model possible: HyperDrive and AutoML. The best Hyperdrive model has Accuracy = 0.91189, while the best model picked by the AutoML method was VotingEnsemble and has Accuracy = 0.91675. The best performing model was developed using the AutoML method. I used compute cluster, which was created using Azure ML SDK (vm_size = "Standard_D2_V2").

## Scikit-learn Pipeline
The custom mode - "train.py" uses the Sklearn library to train a logistic regression. There are two hyperparameters that I was tuning using the HyperDrive method: C - Inverse of regularization strength and max_iter - Maximum number of iterations to converge. The training data was uploaded using TabularDatasetFactory and split into the train (70%) and test (30%) using the train_test_split function.
In some situations, a business might need to use a specific ML model type, or a data scientist has a  preference based on her expertise. The HyperDrive approach (creation of multiple models with different hyperparameters values) would be preferred in those situations. 
I used an early stoping policy to limit Azure ML model development cost, preventing the algorithms from running when there are no more significant accuracy improvements.
The best Accuracy (0.91189) was achieved with the following hyperparameters: C = 0.5, max_iter = 200. I register the model for further use.

## AutoML
The second method I leveraged to train the model was AutoML. After loading data, I prepare automl_config to select the highest accuracy model for the classification task. The best model was VotingEnsemble with Accuracy = 0.91675. Another great feature of the AutoML is Explanations, which gives the top features by their importance (in this model, they were: duration, nr.employed, emp.var.rate and cons.conf.id). 

## Pipeline comparison
Both models provide high Accuracy of more than 0.91. However, the VotingEnsemble by AutoML model shows the highest result. The HyperDrive method might be beneficial when we are limited or know in advance which model type should be used for the specific task. It allows for fine-tuning the model by testing multiple hyperparameters combinations. However, it requires more data preparations steps. 
On the other hand, the AutoML method allows testing a number of model types, testing different features and hyperparameters. It will work better when a data scientist is unsure which model to start with or doesn't have much time to build a model manually. In the current fast-paced business environment, the AutoML will have a number of benefits. It allows addressing multiple business challenges in parallel and leveraging all power of the Azure cloud Machine Learning.


## Future work

Due to limited resources (available Azure credits and limited lab time), I had to restrict the train time for both models. The HyperDrive approach would benefit from testing more arguments, while the AutoML method might be able to fine-tune its selection if provided more training time.

## Proof of cluster clean up
I was using my Azure subscription and confirming cluster deletion after the end of the project.
![Screenshot](https://github.com/Mnarbekov/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/a6eea08ffc132c9c16b10c0012a5d30b5a6c3047/Screenshot%202022-02-16%20215246.png)
 
