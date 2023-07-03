# Deployment
When do we want predictions: 
- Batch/offline: model is not running all the time 
- Online: model is running all the time, either through a web service or a streaming service

## Web service 
When users asks for a prediction, the model is applied and the prediction is returned. Model is up and running all the time.

## Streaming 
There is no 1-to-1 relationship between input and output. There is a 1-to-many relationship. A request is used by different models to make predictions. The connection is more implicit. Back-end is connected to a data stream, the data stream feeds multiple applications. It is less important how many services are connected to the stream. They can be added or removed at any time.

## Batch deployment 
At a regular interval. A scoring job takes the data, applies the model and gives back predictions. 
To do batch deployment, you first need a script that makes predictions, given a certain data frame. The output of this script needs to be another `pandas` dataframe that you can save to the output directory (e.g., as a parquet file).  Using your notebook, use the followiong commands:

`jupyter nbconvert --to script notebook.ipynb`

In the notebook, define a run function that takes a pandas dataframe as input and returns a pandas dataframe as output. In the same script, also define the following in your script: 

<pre>
<code> 
import sys

def run()
    argument1 = sys.argv[1]
    apply_model(argument1)

if __name__ == '_main__':
    run() 
</code>
</pre>

Then you can run the script in your command line: 
<pre>
<code> 
python script.py argument1
</code>
</pre>


It is also good to use `pipenv` to create a requirements.txt file. 

You can then use Prefect to chreate a flow that runs the script. Decorate the run function with `@task` and create a flow using `@flow`. Your main function defines the flow of individual functions. Each of the subfunctions are tasks. To log the flow to Prefect import the `logger` and use `logger.info("message")`. To run the flow, use `flow.run()`.

You can then start the interface by typing 


<pre>
<code> 
prefect orion start
</code>
</pre>

To deploy the flow, create a `deployment.py` file with the following contents: 

<pre>
<code> 
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow_location="path/to/flow.py",
    name="flow-name",
    parameters={
        "param1": "value1", #that need to be passed to the flow
    },
    flow_storage="string", 
    schedule=CronSchedule(cron="0 3 2 * *"), #crontab.guru can generate these expressions
    flow_runner=SubprocessFlowRunner(),
    tags=["tag1", "tag2"]
)

</code>
</pre>

Then run 
<pre>
<code> 
prefect deployment create deployment.py
</code>
</pre>

In the UI, there will then be a new deployment. Create a work queue in the interface and add the deployment to the queue. Then start an agent on that queue using 

<pre>
<code>
prefect agent start work-queue-id
</code>
</pre>

To execute the process for older data, create a new flow that calls on your main flow for every month you want to backfill and execute that script in the command line. 