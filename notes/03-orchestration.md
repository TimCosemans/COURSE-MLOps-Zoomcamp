# 3 Orchestration 
Most ML workflows look similar. And there are many points where this flow can fail. Orchestration is the process of automating the flow of the ML workflow, sets up logging, make it retry if it fails and sends messages when it fails etc. Prefect provides tools for working with complex systems. It is an open-source Python framework to turn standard pipelines into fault-tolerant dataflows. Install it using `pip install -U prefect`.

A prefect server contains an API and a UI. The API is used to interact with the server. The UI is used to monitor the server. The server can be run locally or in the cloud. The server is used to store flows and runs. A flow is a collection of tasks. It is a container that has logic and a parent function. A task is a Python function that can be run. A run is an execution of a flow. You will use task- and flow-decorators to denote these. A subflow is a flow called by another flow. 

To start the server, use

<pre>
<code>
prefect server start
</code>
</pre>

Configurate the prefect API URL using 

<pre>
<code>
prefect config set PREFECT_API_URL=url
</code>
</pre>

## Tasks and Flows
You can add decorators just above functions to define tasks and flows.

For example, to define a task, use

<pre>
<code>
@task(retries=4, retry_delay_seconds=10, log_prints=True)
def task1():
    pass
</code>
</pre>

To define a flow, use

<pre>
<code>
@flow(log_prints=True)
def flow1():
    pass
</code>
</pre>

## Deployment 
To deploy a flow, use

<pre>
<code>
prefect project init
</code>
</pre>

This creates a .prefectignore file (similar to .gitignore), a deployment.yaml (for templating), a prefect.yaml and a prefect folder. The prefect.yaml file has a pull step that specifies which code to pull.

We then need a worker to poll the work pool. To do this, use

<pre>
<code>
prefect worker start -p my-pool -t process #process = locally
prefect deploy 3.4/orchestrate.py:main_flow -n 'my deployment' -p my-pool #to deploy
prefect worker  start -p my-pool #to start the worker 
prefect deployment run Hello/my-deployment #to run from the CLI
</code>
</pre>

Worker polls for work and sets up infrastructure according to project steps. It then runs the flow.

## Deployment
To deploy a flow on AWS, create the necessary blocks. 

<pre>
<code>
!prefect block register -m prefect_aws

from time import sleep 
from prefect_aws import S3Bucket, AwsCredentials

def create_aws_creds_block():
    my_aws_creds = AwsCredentials(
        aws_access_key_id="",
        aws_secret_access_key=""
    )

    my_aws_cred.save(name="my-aws-creds", overwrite=True)

def create_s3_bucket_block():
    aws_creds = AwsCredentials.load(name="my-aws-creds")    
    my_s3_bucket = S3Bucket(bucket_name="my-bucket", credentials=aws_creds)

    my_s3_bucket.save(name="my-s3-bucket", overwrite=True)

if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()

</code>
</pre>

Then create an orchestration file (like above, with the following modifications).

<pre>
<code>
from prefect import task, Flow
from prefect_aws import S3Bucket, AwsCredentials

s3_bucket_block = S3Bucket.load(name="my-s3-bucket")
s3_bucket_block.download_folder_to_path(from_folder="data", to_folder="data") #download data from S3 to local folder

</code>
</pre>

Then create a deployment.yaml file (like above, with the following modifications).

<pre>
<code>
deployments: 
-   name: taxi_local_data
    entrypoint: orchestrate.py:main_flow
    work_pool: my-pool
        name: my-pool
-   name: taxi_s3_data
    entrypoint: orchestrate_s3.py:main_flow_s3
    work_pool: my-pool
        name: my-pool

</code>
</pre>

Then do 

<pre>
<code>
prefect deploy --all
</code>
</pre>

## Artifacts
Artifacts are files that are saved during the run. They can be used to store models, data, images etc. To log an artifact, use

<pre>
<code>
from prefect.artifacts import create_markdown_artifact  
from datetime import date 

markdown__rmse_report = f"""# RMSE Report

## Sumary 

Diration Prediction 

## RMSE XGBoost Model

| Date | RMSE |
| --- | --- |
| {date.today()} | {rmse:.2f} |
"""

create_markdown_artifact(
    key="rmse-report",
    markdown=markdown__rmse_report
    )

</code>
</pre>

To set a schedule, type

<pre>
<code>
prefect deployment set-schedule main_flow/taxi --interval 120
</code>
</pre>

