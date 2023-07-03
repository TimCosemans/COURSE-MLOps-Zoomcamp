# Scalable Machine Learning with Apache Spark
## Apache Spark
Apache Spark is a fast and general engine for large-scale data processing. It provides high-level APIs in Java, Scala, Python, and R, and an optimized engine that supports general execution graphs. 

Spark has drivers. They pass tasks over to workers, who pass them to executors. They are all part of the JVMs. Dataframes are the main data structure in Spark. Custom functions (UDF) have speeds that differ by language, otherwise they are all approximately the same speed. Spark executes a command by taking the API (e.g. PySpark), creating a logical plan and using a catalyst optimizer to create the physical execution. 

Spark is great when the data is too large to process on a single machine and you get out-of-memory errors or you need to speed up your processing. 

### Dataframes
Spark does not actually executes jobs until the data is called upon to e.g. view. If jobs are executed, you can click on them to see event timelines, executed queries etc.

It is much faster to access data if we first run the `cache()` command. This will store the data in memory.

Storing data using e.g. `.toPandas()` will store the data in the driver. This is not recommended for large datasets and can still cause out-of-memory errors. Try explicetly limiting the number of records pulled back to the driver. 

## Delta Lake
Delta Lake is an open-source storage layer that sits on top of a data lake. It includes data versioning, time travel, schema enforcement, and ACID transactions. It is formatted in parquet and compatible with Apache Spark.

You can write a dataframe to a Delta Lake using `df.write.format("delta").saveAsTable("path/to/delta")`. 

Delta supports partitioning. Partitioning puts data with the same value for the partitioned column into its own directory. Operations with a filter on the partitioned column will only read directories that match the filter. This optimization is called partition pruning. Partitioning can be done by calling on the `partitionBy()` function.	Delta will hen store the partitions in different files. 

When a user creates a Delta Lake table, that table’s transaction log is automatically created in the _delta_log subdirectory. As he or she makes changes to that table, those changes are recorded as ordered, atomic commits in the transaction log. Each commit is written out as a JSON file, starting with 000000.json. Additional changes to the table generate more JSON files. Each of these commits is a snapshot of the table’s state at a particular point in time.

Reading data from a Delta table can be done using `spark.read.format("delta").load("path/to/delta")`. After modifying it, you can save it using `df.write.format("delta").mode("overwrite").save("path/to/delta")`.

Accessing previous versions is possible up to thirty days 

<pre>
<code> 
%sql
DROP TABLE IF EXISTS train_delta;
CREATE TABLE train_delta USING DELTA LOCATION '${DA.paths.working_dir}'
DESCRIBE HISTORY train_delta

time_stamp_string = str(spark.sql("DESCRIBE HISTORY train_delta").collect()[-1]["timestamp"])
df = spark.read.format("delta").option("timestampAsOf", time_stamp_string).load(DA.paths.working_dir)
</code>
</pre>

You can, after your done, clean up your directory and permanently delete everything previous to your desired retention period (in hours). 

<pre>
<code>
from delta.tables import DeltaTable

spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false") #bypass default retention check
delta_table = DeltaTable.forPath(spark, DA.paths.working_dir)
delta_table.vacuum(0) #retain 0 hours in the past
</code>
</pre>
 
## Machine Learning

Using Spark's `.withColumn()` function, you can add a new column to a dataframe. Changing datatypes can be done using `.cast()`. 

Summary statistics can be found using `.describe()` and `.summary()`. A more detailed summary can be found using `dbutils.data.summarize(dataframe)`.

Selecting columns can be done using `.select()` and filtering can be done using `.filter()`.

If-else statements are not supported in Spark. Instead, use `when()` and `otherwise()` like so: `df.withColumn("new_column", when(df["column"] == "value", "new_value").otherwise("old_value"))`.

Transformers do not learn any parameters from your data and simply apply rule-based transformations. Estimators are fit on a DataFrame to produce a Transformer. An imputer is an example of an estimator.

<pre>
<code>
from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)

imputer_model = imputer.fit(df)
imputed_df = imputer_model.transform(df)
</code>
</pre>

### Linear Regression
Scikit learn is a single node machine learning framework. If data is too large, we can use SparkML. Its implementation is slighly different from Scikit learn and it does not have all of the same algorithms. Spark has two different libraries. MLlib, which is based on RDDs and sligly older, and Spark ML, which is based on Dataframes and is newer.

When creating a split using `.randomSplit()`, use seeds to make the split deterministic. If we change the number of partitions, this will create a shuffle in the background. Cache the dataframe in advance before reparitioning to avoid this.

You can perform a linear regression using the following code:

<pre>
<code>
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df)
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)

lr_model.coefficients
lr_model.intercept
</code>
</pre>

To make predictions, use the following code:

<pre>
<code>
test_df = assembler.transform(test_df)
predictions = lr_model.transform(test_df)
</code>
</pre>


To evaluate the model, use the following code:

<pre>
<code>
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
rmse = evaluator.evaluate(predictions)
</code>
</pre>

#### One-hot encoding

SparkML automatically drops last columnn after one-hot encoding. To avoid this, use `dropLast=False` when creating the `OneHotEncoderEstimator`.

If there are a lot of categories, Spark reduces them to sparse vectors that indicate (number of elements, [indices of the non-zero elements], [their values]). 

To do one-hot encoding, use the following code:

<pre>
<code>
from pyspark.ml.feature import OneHotEncoder, StringIndexer


categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip") #map a string column of labels to an ML column of label indices
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)

</code>
</pre>

#### Pipeline 
Create a pipeline to chain together multiple estimators and transformers using the following code:

<pre>
<code>
from pyspark.ml import Pipeline

stages = stages = [string_indexer, ohe_encoder, assembler, lr]
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

</code>
</pre>

To save and load these models, use the following code:

<pre>
<code>
from pyspark.ml import PipelineModel

pipeline_model.write().overwrite().save("path/to/model")
loaded_pipeline_model = PipelineModel.load("path/to/model")
</code>
</pre>

### Decision tree
One-hot encoding categorical variables with high cardinality can cause inefficiency in tree-based methods. Continuous variables will be given more importance than the dummy variables by the algorithm, which will obscure the order of feature importance and can result in poorer performance.

It is therefore best to just use the stringindexer and pass these as ordinal variables instead of one hot encoded variables. 

To build a decision tree, use the following code:

<pre>
<code>
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline

dt = DecisionTreeRegressor(labelCol="price")
dt.setMaxBins(40)
# data is partitioned by row
# each worker has to compute summary statistics for every feature for each split point. 
# summary statistics have to be aggregated
# What if worker 1 had the value 32 but none of the others had it? 
# maxBins parameter for discretizing continuous variables into buckets, but the number of buckets has to be as large as the categorical variable with the highest cardinality

# Combine stages into pipeline
stages = [string_indexer, vec_assembler, dt]
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train_df)

</pre>
</code>

To get the feature importance, use the following code:

<pre>
<code>
import pandas as pd

features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), dt_model.featureImportances)), columns=["feature", "importance"])
features_df
</code>
</pre>

With decision trees, the scale of the features does not matter. For example, it will split 1/3 of the data if that split point is 100 or if it is normalized to be .33. The only thing that matters is how many data points fall left and right of that split point - not the absolute value of the split point. This is not true for linear regression, and the default in Spark is to standardize first.

### Random forest

<pre>
<code>
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

rf = RandomForestRegressor(labelCol="price", maxBins=40)

param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
              .build())

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

cv_model = cv.setParallelism(4).fit(train_df)
</code>
</pre>


### AutoML
AutoML returns an MLflow experiment, a data exploration notebook to explore summary statistics and a reproducible notebook to reproduce the best model. You can use it to quickly verify the predictive power of your data and to quickly build a baseline models. Currently, AutoML uses a combination of XGBoost and sklearn (only single node models) but optimizes the hyperparameters within each. If using a Spark DataFrame, it will convert it to a Pandas DataFrame under the hood by calling `.toPandas()` - just be careful you don't OOM!

<pre>
<code>
from databricks import automl
import mlflow

summary = automl.regress(train_df, target_col="price", primary_metric="rmse", timeout_minutes=5, max_trials=10)

print(summary.best_trial)

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("price").columns))
display(pred_df)
</code>
</pre>

### Feature Store
Feature engineering is time-consuming and error-prone. The feature store is a centralised repository of features and ensures the same code is used for model training and inference. It also allows you to share features across teams and projects. 

It contains a feature registry and feature provider. The registry is a centralised repository of features and metadata. The provider is a library that allows you to access the features in the registry.

You first have to create a feature store client: 

<pre>
<code>
from databricks import feature_store

fs = feature_store.FeatureStoreClient()
</code>
</pre>

To create a feature table, use the following code: 

<pre>
<code>
fs.create_table(
    name=table_name,
    primary_keys=["index"],
    df=numeric_features_df,
    schema=numeric_features_df.schema, #list of column names and types
    description="Numeric features of airbnb data"
)

#OR without df and using .write_table()

fs.create_table(
    name=table_name,
    primary_keys=["index"],
    schema=numeric_features_df.schema,
    description="Original Airbnb data"
)

fs.write_table(
    name=table_name,
    df=numeric_features_df,
    mode="overwrite"
)
#you can also use this piece of code to overwrite the table
</code>
</pre>


To get the attributes you can use `fs.get_table(table_name).path_data_sources` or `fs.get_table(table_name).description`. 

To query a feature table, use the following code:

<pre>
<code>
model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

# fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="price", exclude_columns="index")
training_pd = training_set.load_df().toPandas()
</code>
</pre>

During an MLflow run, you can also log feature store models. 

<pre>
<code>
fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"feature_store_airbnb_{DA.cleaned_username}",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )
</code>
</pre>

These models can then be used to do batch scoring. 

<pre>
<code>
predictions_df = fs.score_batch(f"models:/feature_store_airbnb_{DA.cleaned_username}/1", 
                                  batch_input_df, result_type="double")
</code>
</pre>

### Gradient Boosted Trees

<pre>
<code>
from sparkdl.xgboost import XgboostRegressor
from pyspark.ml import Pipeline

params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42, "missing": 0}

xgboost = XgboostRegressor(**params)
model = xgboost.fit(train_df)
</pre>
</code>

## Pandas 
As of Spark 2.3, there are Pandas UDFs available in Python to improve the efficiency of UDFs. Pandas UDFs utilize Apache Arrow to speed up computation. A Pandas UDF is defined using the pandas_udf() as a decorator or to wrap the function, and no additional configuration is required.

### Type hints 
In Spark 3.0, you should define your Pandas UDF using Python type hints.

<pre>
<code>
def greet(name: str) -> str:
    return "Hello, " + name

</pre>
</code>

The `name: str` syntax indicates the name argument should be of type str. The `->` syntax indicates the `greet()` function will return a string.


### Pandas UDFs

<code>
<pre>
from pyspark.sql.functions import pandas_udf

@pandas_udf("double")
def predict(*args: pd.Series) -> pd.Series:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    pdf = pd.concat(args, axis=1)
    return pd.Series(model.predict(pdf))

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))
display(prediction_df)
</pre>
</code>

### Pandas Scalar Iterator UDF

If your model is very large, then there is high overhead for the Pandas UDF to repeatedly load the same model for every batch in the same Python worker process. In Spark 3.0, Pandas UDFs can accept an iterator of pandas.Series or pandas.DataFrame so that you can load the model only once instead of loading it for every series in the iterator.

<code>
<pre>
from typing import Iterator, Tuple

@pandas_udf("double")
def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    for features in iterator:
        pdf = pd.concat(features, axis=1)
        yield pd.Series(model.predict(pdf))

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))
display(prediction_df)

</pre>
</code>

When you call a function that has a yield statement, as soon as a yield is encountered, the execution of the function halts and returns a generator iterator object instead of simply returning a value. The state of the function, which includes variable bindings, the instruction pointer, the internal stack, and a few other things, is saved. When the iterator’s next() method is called, the function resumes where it left off, and the execution continues until the next yield is encountered. This process continues until the function terminates.

### Pandas function API
Instead of using a Pandas UDF, we can use a Pandas Function API. This new category in Apache Spark 3.0 enables you to directly apply a Python native function, which takes and outputs Pandas instances against a PySpark DataFrame. `mapInPandas()` takes an iterator of pandas.DataFrame as input, and outputs another iterator of `pandas.DataFrame`.

<code>
<pre>
def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    for features in iterator:
        yield pd.concat([features, pd.Series(model.predict(features), name="prediction")], axis=1)
    
display(spark_df.mapInPandas(predict, """`host_total_listings_count` DOUBLE,`neighbourhood_cleansed` BIGINT,`latitude` DOUBLE,`longitude` DOUBLE,`property_type` BIGINT,`room_type` BIGINT,`accommodates` DOUBLE,`bathrooms` DOUBLE,`bedrooms` DOUBLE,`beds` DOUBLE,`bed_type` BIGINT,`minimum_nights` DOUBLE,`number_of_reviews` DOUBLE,`review_scores_rating` DOUBLE,`review_scores_accuracy` DOUBLE,`review_scores_cleanliness` DOUBLE,`review_scores_checkin` DOUBLE,`review_scores_communication` DOUBLE,`review_scores_location` DOUBLE,`review_scores_value` DOUBLE, `prediction` DOUBLE""")) 
</pre>
</code>

#### Example 
<pre>
<code>
train_return_schema = "device_id integer, n_used integer, model_path string, mse float"

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an sklearn model on grouped instances
    """
    # Pull metadata
    device_id = df_pandas["device_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0] # Pulls run ID to do a nested run

    # Train the model
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Evaluate the model
    predictions = rf.predict(X)
    mse = mean_squared_error(y, predictions) # Note we could add a train/test split

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id) as outer_run:
        # Small hack for running as a job
        experiment_id = outer_run.info.experiment_id
        print(f"Current experiment_id = {experiment_id}")

        # Create a nested run for the specific device
        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)
            mlflow.set_tag("device", str(device_id))

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            # Create a return pandas DataFrame that matches the schema above
            return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                                    columns=["device_id", "n_used", "model_path", "mse"])

    return return_df 

with mlflow.start_run(run_name="Training session for all devices") as run:
    run_id = run.info.run_id

    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id)) # Add run_id
        .groupby("device_id")
        .applyInPandas(train_model, schema=train_return_schema)
        .cache()
    )

combined_df = df.join(model_directories_df, on="device_id", how="left")
display(combined_df)

apply_return_schema = "record_id integer, prediction float"

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Applies model to data for a particular device, represented as a pandas DataFrame
    """
    model_path = df_pandas["model_path"].iloc[0]

    input_columns = ["feature_1", "feature_2", "feature_3"]
    X = df_pandas[input_columns]

    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "record_id": df_pandas["record_id"],
        "prediction": prediction
    })
    return return_df

prediction_df = combined_df.groupby("device_id").applyInPandas(apply_model, schema=apply_return_schema)
display(prediction_df)

</pre>
</code>

MLflow allows models to deploy as real-time REST APIs. At the moment, a single MLflow model serves from one instance (typically one VM). However, sometimes multiple models need to be served from a single endpoint. Imagine 1000 similar models that need to be served with different inputs. Running 1000 endpoints could waste resources, especially if certain models are underutilized.

One way around this is to package many models into a single custom model, which internally routes to one of the models based on the input and deploys that 'bundle' of models as a single 'model.'

Below we demonstrate creating such a custom model that bundles all of the models we trained for each device. For every row of data fed to this model, the model will determine the device id and then use the appropriate model trained on that device id to make predictions for a given row.

<pre>
<code>
device_to_model = {row["device"]: mlflow.sklearn.load_model(f"runs:/{row['run_id']}/{row['device']}") for row in model_df.collect()}
                                
from mlflow.pyfunc import PythonModel

class OriginDelegatingModel(PythonModel):
    
    def __init__(self, device_to_model_map):
        self.device_to_model_map = device_to_model_map
        
    def predict_for_device(self, row):
        '''
        This method applies to a single row of data by
        fetching the appropriate model and generating predictions
        '''
        model = self.device_to_model_map.get(str(row["device_id"]))
        data = row[["feature_1", "feature_2", "feature_3"]].to_frame().T
        return model.predict(data)[0]
    
    def predict(self, model_input):
        return model_input.apply(self.predict_for_device, axis=1)

example_model = OriginDelegatingModel(device_to_model)
example_model.predict(combined_df.toPandas().head(20))

</pre>
</code>

### Pandas API 

<code>
<pre>
import pyspark.pandas as ps

ps.set_option("compute.default_index_type", "distributed-sequence")
df = ps.read_parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")

#OR
spark_df = spark.read.parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")

df = ps.DataFrame(spark_df)
df = spark_df.to_pandas_on_spark()

</pre>
</code>

Based on the type of visualization, the pandas API on Spark has optimized ways to execute the plotting.
<br><br>

![](https://files.training.databricks.com/images/301/ps_plotting.png)
