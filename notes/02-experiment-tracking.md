# 2 Experiment Tracking
Experiment tracking is a way to track the parameters, code, and data that you use to train your model.

## MLFlow
Use MLFlow to keep track of your experiments. Using the command `mlflow.autolog()` or libraries specific to your library (such as `mlflow.sklearn.autolog()`), you can log parameters (such as preprocessing, paths etc.), metrics, models, environments and artifacts automatically. Runs are trainings of a model. Experiments are a collection of runs. Use `mlflow.set_experiment()` to set the experiment. MLFlow logs automatically the source code and its version, start and end time and the author

By default, MLFlow will assume you will store everything locally. It will create a folder called `mlruns` in the current directory. In this case, it is not possible to see the model registry. Using `mlflow ui` in the right directory, you can start a server that will allow you to see the runs. Delete cookies if you have problems. 


You can also use a database to store the runs. To start it manually, use the following code:

<pre>
<code>
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts 
</code>
</pre>

This creates a database and an artifacts folder. To do this on the cloud, look at mlflow_on_aws.md.

Parameters are hyperparameters. Metrics are the results of the model. Artifacts are files that are saved during the run. Source is the code that originally ran the experiment. For Spark models, MLFlow can only log PipeLineModels.

To log a model, use the following code:

<pre>
<code>
import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment") #will add a new experiment if it doesn't exist

with mlflow.start_run(run_name="LR-Single-Feature") as run: #all associated with this run
    mlflow.set_tag("stage", "experiment")

    # Define pipeline
    vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="price")
    pipeline = Pipeline(stages=[vec_assembler, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log parameters
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "bedrooms")
    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

    # Log model
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas()) 

    # Evaluate predictions
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regression_evaluator.evaluate(pred_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
</code>
</pre>

In the UI, you can select and compare runs. MLFlow will automatically gives comparative plots. You can filter records using, e.g., tags.model = "xgboost". Some models have their own autologging functions, such as `mlflow.sklearn.autolog()`.

### Query past runs
To query past runs, you need to access the client: 

<pre>
<code>
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.list_experiments()

experiment_id = run.info.experiment_id #run is the output of mlflow.start_run()
runs_df = mlflow.search_runs(experiment_id)

runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics
runs[0].info.run_id

</code>
</pre>

Databricks has a built-in UI for MLFlow. Just press the beaker icon on the top of the screen.

### Hyperparameter tuning
You can either use a validation set to tune the hyperparameters or use cross validation. Cross validation is better because it is more robust to overfitting.

There are two ways to scale hyperopt with Apache Spark:
* Use single-machine hyperopt with a distributed training algorithm (e.g. MLlib)
* Use distributed hyperopt with single-machine training algorithms (e.g. scikit-learn) with the SparkTrials class.

Unfortunately you can’t use hyperopt to distribute the hyperparameter optimization for distributed training algorithms at this time.

First, we define our objective function. The objective function has two primary requirements:
* An input params including hyperparameter values to use when training the model
* An output containing a loss metric on which to optimize
In this case, we are specifying values of max_depth and num_trees and returning the RMSE as our loss metric.


<pre>
<code>
def objective_function(params):    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    with mlflow.start_run(): #log all of the rounds in the optimization process
        estimator = pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
        model = estimator.fit(train_df)

        preds = model.transform(val_df)
        rmse = regression_evaluator.evaluate(preds)
        mlflow.log_metric("rmse", rmse)

    return rmse 

</code>
</pre>

Next, we define our search space. This is similar to the parameter grid in a grid search process. However, we are only specifying the range of values rather than the individual, specific values to be tested. It's up to hyperopt's optimization algorithm to choose the actual values.

<code>
<pre>
from hyperopt import hp

search_space = {
    "max_depth": hp.quniform("max_depth", 2, 5, 1),
    "num_trees": hp.quniform("num_trees", 10, 100, 1)
}
</code>
</pre>

`fmin()` generates new hyperparameter configurations to use for your `objective_function`.Hyperopt allows for parallel hyperparameter tuning using either random search or Tree of Parzen Estimators (TPE).

<pre>
<code>
from hyperopt import fmin, tpe, Trials
import numpy as np

num_evals = 4
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))
</code>
</pre>


### Model management 
You can log a model as an artifact using the following code:

<pre>
<code>
with open("models/preprocessor.b", "wb") as f_out:
    pickle.dump(dv, f_out)
mlflow.log_artifact(local_path="models/preprocessor.b", artifact_path="preprocessor")
#OR 
mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
</code>
</pre>

To make predictions using a registered model, MLFlow gives you a code snippet (to load as a pyfunc). You can also use the following code to load as a flavoured model:

<pre>
<code>
xgboost_model = mlflow.xgboost.load_model(model_uri="runs:/<run_id>/model")
</code>
</pre>

#### Model registry
The model registry is a central repository for models. It allows you to version and stage models. You can also add descriptions and tags to models. The MLFlow registry does not do anything, it just labels the models. You need to supply your own code to deploy the models.

You can access it in DataBricks by switching to the Machine Learning persona and clicking on the Models tab on the left.

You can register and update models from previous runs using the following code: 

<pre>
<code>
from mlflow.tracking import MlflowClient

MLFLOW_URI = "sqlite:///mlflow.db"
client = MlflowClient(tracking_uri=MLFLOW_URI)

#search for the best models
runs = client.search_runs(experiment_ids='1'
    filter_string="metrics.rmse < 100", 
    run_view_type=ViewType.ACTIVE_ONLY, 
    max_results=5, 
    order_by=["metrics.rmse ASC"])

mlflow.set_tracking_uri(MLFLOW_URI)

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name="model_name")

client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using OLS linear regression with sklearn."
)
#to update a certain version
</code>
</pre>

To transition a model to a different stage, use the following code:

<pre>
<code>
client.list_registered_models()

latest_versions = client.get_latest_versions(name=model_details.name, stages=["None"])

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Staging"
)
</code>
</pre>

Fetch the latest model using a pyfunc. Loading the model in this way allows us to use the model regardless of the package that was used to train it.

<pre>
<code>
import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

</code>
</pre>

If you register a new model with the same name in a new run, it will be assigned a new version number. You can also use the `mlflow.register_model` function to register a model that was not trained in the current run.

If you want to push a new model to production and deleted archived versions, use the following code:

<pre>
<code>
client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production", 
    archive_existing_versions=True # Archive existing model in production 
)

client.delete_model_version(
    name=model_name,
    version=1
)
</code>
</pre>

You should also do data versioning and model versioning. You can use DVC for this. 
