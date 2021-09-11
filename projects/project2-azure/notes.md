# Azure DataScience Notes
## Module 1 - Overview
## Module 2 - NoCode ML
## Module 3 - Running Experiment and Training Models
### 3.3 
Experiment object called from azure ml python sdk, keeps track of experiment data and metadata in log files - can be run local or cloud

code:

    from azureml.core import Workspace, Experiment
    import pandas as pd
    
    ws = Workspace.from_config()
    experiment = Experiment(workspace, "My-Experiment")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")
    data = pd.read_csv('data.csv')
    row_count = (len(data))
    run.log('observations', row_count)
    run.log_metric("Accuracy", accuracy)
    data.sample(100).to_csv('sample.csv', index=False, header=True)
    run.upload_file(name='outputs/sample.csv',path_or_stream='./sample.csv')
    run.complete()
   

### 3.4
Run script as experiment
run = Run.get_context()
    or
script_config = ScriptRunConfig(source_directory='my_dir',script='script.py')
run = experiment.submit(config=script_config)

### 3.5 
MLFlow - Model Management
### 3.6
Training models
### 3.7
Parameters and hyperparameters and argparse for argument parsing
### 3.8 
register models


## Module 4 - Working with DataStores
### 4.3
Registering and adding Data Stores

Code Samples:

    from azureml.core import Workspace, Datastore
    
    ws = Workspace.from_config()
    
    blob_ds = Datastore.register_azure_blob_container(workspace=ws,
                                                        datastore_name='blob_data'
                                                        container_name='data_container'
                                                        account_name='az_store_acct',
                                                        account_key='12345abscde8...')
                                                        
    ds = Datastore.get(ws, datastore_name='blob_data)
    
    ds.upload(src_dir='/files',target_path='/data/files')
    ds.download(target_path='downloads',prefix='/data')
    
### 4.5
Datasets defined (tabular and file)
### 4.6
Creating and Registering Datasets (tabular and file)
### 4.7
ScriptRunConfig vs Script

    code example
    
### 4.8
Pass dataset as named input (scriptrunconfig and script)

    code example

### 4.9
passing dataset as script argument (scriptrunconfig and script)

    code example
    
### 4.10
pass a dataset as named input (scriptrunconfig and script)

    code example

### 4.11 
Create new version of existing dataset

    code example
    

## Module 5 - Working with Compute
### 5.2 
Environments 
Run Contexts for experiments in a container 
### 5.3
Conda Env
    Create env from specification file
        file in YAML format for conda
    Create env from existing conda env

    code example

Specificying packages

    code example
    
### 5.4
Docker Images env
configure env containers 

    code examples
    
### 5.5
save env to workspace

### 5.6
Compute options
    local
    cluster
    attached - used outside of workspace (e.g. vm, azure databricks, azure HDInsight)

### 5.7 
Cluster

    code example

### 5.8 
Databricks

    code example

### 5.9 
Using databrick with config

    code example

## Module 6 - Orchestrating Machine Learning Workflows
### 6.2
Intro to pipelines
Run as experiment or through REST

Can automate data prep, training, deployement
trigger based on events or schedules (e.g. data released and model updated)

### 6.3 
Common Pipeline Steps
Run py script, copy data between data stores, run databricks notebook, script or JAR
Run azure data lake analytics U-SQL script
Run py script as distrivbuted task on mult compute nodes

    step1 = PythonScriptStep(name='prepare_data')
    step2 = PythonScriptStep(name='train_model',...)
    training_pipeline=Pipeline(workspace=ws, steps=[step1,step2])
    pipeline_experiment = Experiment(workspace=ws, name='training-pipeline')
    pipeline_run = experiment.submit(pipeline_experiment)
    
### 6.4
Passing data between steps

Use PipelineData object

    data_store = ws.get_default_datastore()
    prepped = PipelineData('prepped_data',datastore=data_store)
    
    step1 = PythonScriptStep(name='prepare data',
                arguments=['--out_folder', prepped]
                outputs=[prepped],                  #this is PipelineData output 
                ...
                )

    step2 = PythonScriptStep(name='train model',
                arguments=['--in_folder', prepped]
                inputs=[prepped],                   #this is PipelineData input 
                ...
                )

### 6.5
Pipeline Step Reuse

    step1 = PythonScriptStep(name='prepare data',
                arguments=['--out_folder', prepped]
                outputs=[prepped],                   
                allow_reuse=True,                   #Reuse cached step output if unchanged
                ...
                )

use arg to force rerun of steps
    
    regenerate_outputs=True 


### 6.6
Publish pipeline to create REST endpoint

    published_pipeline = pipeline_run.publish(name='training_pipeline',
                            description='Model training pipeline',
                            version='1.0')

post JSON req to initiate a pipeline

    import requests
    response = requests.post(rest_endpoint,
                    headers=auth_header,
                    json={"ExperimentName":"run training pipeline"})
    run_id = response.json()["Id"]

### 6.7 
Pipeline parameters

Parameterize a pipeline before publishing (increases flexibility by allowing variable input)

    code example
    
Pass parameters in JSON request

    code example
    
### 6.8 
Schedule Pipelines 
e.g. Time based
    
    code example
    
### 6.9
Event driven workflows
    e.g. for run completion, failure, model registration, model deplotement, data drift, triggers (az functions, az logic apps, az event hubs, azure data fact pipelines, generic webhooks)

## Module 7 - Deploying and Consuming Models
### 7.2
Real-time Inferencing
- immediate prediction from new data
- usually deployed as web service endpoint

### 7.3 
Deployement
    Register a trained model
    define inference config
        create scoring script
        create env
    define deployement config
        create compute target
        deploy model as service
    
    service = Model.deploy(ws, 'my_service', [model], inference_config, deploy_config)

### 7.4
Consuming real-time inference service
use SDK

    import json
    
    x_new = [[0.1,2.3,4.1,2.0],[0.2,1.8,3.9,2.1]] #Array of feature vectors
    json_data = json.dumps({"data":x_new})
    response = service.run(input_data=json_data)
    predictions = json.loads(response)
    
use REST endpoint

    import json
    import requests

    x_new = [[0.1,2.3,4.1,2.0],[0.2,1.8,3.9,2.1]] #Array of feature vectors
    json_data = json.dumps({"data":x_new})
    request_headers = {'Content-Type':'application/json'}
    response = requests.post(url=endpoint, data=json_data, headers=request_headers)
    predictions = json.loads(response.json())

### 7.5
Troubleshooting

    print(service.state)
    print(service.get_logs())
    
### 7.6
Batch Inferencing
Asycnh prediction from batached data as pipeline

### 7.7
Creating batch inferenceing pipeling
    register model
    create scoring script
    create pipeline with parallelrunstep to run script
    retrieve batch predctions from output

### 7.8 
public batch inferencing

    code example
    
### 7.9
continuous integration delivery
    number of example uses
    
### 7.10
Azure pipelines
    use python or cli to build release pipelines
    automate builds, publishes of models, etc...
    
### 7.11
use github actions
aml-run, aml-registermodel, aml-deploy


## Module 8 - Training Optimal Models

### 8.2
Hyperparameter tuning
    Training multiple models with same algo but different hyperparameters 

### 8.3 
Discrete Hyperparameters 
    choice (list or range)
    discrete dist (qnormal, quniform, qlognormal, qloguniform)
Continous Hyperparameters
    continous dist (normal, uniform, lognormal, loguniform)
    
    param_space = {
        '--batch_size':choice(16,32,64),
        '--learning_rate':normal(10,3)
        }

### 8.4 
Hyperparameter sampling
    grid
    bayesian
    random

        from azureml.train.hyperdrive import RandomParameterSampling
        param_sampling = RandomParameterSampling(param_space)
    
    
### 8.5
Early Termination Policy
    Bandit
    Median
    Truncation Selection
        
        from azureml.train.hyperdrive import TruncationSelectionPolict
        stop_policy = TruncationSelectionPolicy(evaluation_interval=1, truncation_percentage=20, delay_evaluation=5)
        
### 8.6
Tuning hyperparameters with Hyperdrive
experiment script

    parser.add_argument('--reg', type=float, dest='reg_rate') 
    ...
    run.log('Accuracy', model_accuracy)
    
Hyperdrive configuration

    hyperdrive = HyperDriveConfig(run_config=script_config,
                    hyperparameter_sampling=param_sampling,
                    policy=stop_policy,
                    primary_metric_name='Accuracy',
                    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                    max_total_runs=6,
                    max_concurrent_runs=4)
                    
    hyperdrive_run = experiment.submit(config=hyperdrive)
    
### 8.7
automated ml

### 8.8
Prepping data (if only training data is supplied cross validation will be applied automatically)

    tabs_ds = ws.datasets.get("tabular dataset")
    
    train_ds,test_ds = tabs_ds.random_split(percentage=0.7, seed=123)
    
### 8.9
Configure auto ml experiment run

    from azureml.train.automl import AutoMLConfig
    
    automl_config = AutoMLConfig(name='AutomatedMLExperiment',
                        task='classification',
                        compute_target=aml_cluster,
                        training_data=train_ds,
                        validation_data=test_ds,
                        label_column_name='Label',
                        iterations=20,
                        primary_metric='AUC_weighted',
                        max_concurrent_iterations=4,
                        featureization='auto')
    automl_run = automl_experiment.submit(automl_config)
                        
### 8.10
monitoring and reviewing runs

    code example

## Module 9 - Responsible ML
### 9.2
Differential Privacy
Data privacy is important, protect information given or information revealed

### 9.3
Differential privacy is ...?? no idea

### 9.4
epsilon - privacy loss parameter

### 9.5
model interpretability
based on open source interpret-community package
    SHAP (shapely additive explantions)
    LIME (local interpretable model-agnostic explanations)

### 9.6
Global feature importance
Local feature importance
global is importance over all test data, local is importance for an individual data sample

### 9.7
Explainers
use azureml-interpret package
    mimicexplainer
    tabularexplainer
    PFIExplainer
    
### 9.8
global or local feature explanations

    from interpret.ext.blackbox import TabularExplainer
    
    tab_explainer = TabularExplainer(model, X_train, features=features, classes=labels)
    global_explanation = tab_explainer.explain_global(X_train)
    
### 9.9
adding explanation to training experiments
import ExplanationClient class
generate explanations

    code example

use explanationclient to download explanations

### 9.10
visualising model explanations

### 9.11
Interpretability during inferencing

    code example
    
deploy service with model and explainer

    code example
    
### 9.12
Fairness

### 9.13
Evaluation model fairness

### 9.14
Mitigating Unfairness
    demographic parity
    true positive rate parity
    false positive rate parity
    equalized odds
    error rate parity
    bounded group loss

## Module 10 - Monitoring Models

### 10.2
Monitoring models with applications insights
enables capture, storage, analysis of telemetry data

### 10.3
enabling applications insights

    code examples
    
### 10.4
capturing and viewing application insights data

    code example
    
query logs

    code examples
    
### 10.5
monitoring data drift

### 10.6
creating data drift monitor

    code example
    
### 10.7
data drift schedulers and alerts
    frequency
    drift threshold for alerting
    alert config
    schedule start (for model data drift monitors)
    data latency
        
        code example
        
### 10.8
reviewing data drift

### 10.9

# Questions
Difference between 
    pip install azureml
        this doesnt seem to work contain the azureml.core components
        check pip list for installed packages
    pip install azureml-core
    pip install azureml-sdk
    pip install azureml-sdk[notebooks]

diff between
    azureml CLI
    azureml python SDK

azureml docker image
    what is going on with azureml docker image, why doesnt it come with azureml?
    ruamel
        https://azure.github.io/azureml-sdk-for-r/articles/troubleshooting.html#modulenotfounderror-no-module-named-ruamel
        https://github.com/Azure/MachineLearningNotebooks/issues/1110
        why doesnt the official image get updated with the right pip version if the solution is so easy?

config
    what does the config file need to look like to set up workspace?

# Pages
## Docs
general - https://docs.microsoft.com/en-us/azure/machine-learning/

## Setup Config
?

## Deploy Local
https://github.com/Azure/azureml-examples/blob/main/python-sdk/tutorials/deploy-local/1.deploy-local.ipynb

## MNIST
https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-train-models-with-aml
https://github.com/Azure/azureml-examples/tree/main/python-sdk/workflows/deploy

## Examples
python sdk - https://github.com/Azure/azureml-examples/tree/main/python-sdk
    hello world - https://github.com/Azure/azureml-examples/blob/main/python-sdk/tutorials/an-introduction/1.hello-world.ipynb
notebooks - https://github.com/Azure/MachineLearningNotebooks



## setup python dev for az ml
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#local

## Docker and Azure
https://github.com/Azure/AzureML-Containers
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-custom-image

## Azure ML Templates
https://github.com/Azure/azureml-template

