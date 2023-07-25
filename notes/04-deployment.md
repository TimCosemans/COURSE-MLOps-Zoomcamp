# Deployment
When do we want predictions: 
- Batch/offline: model is not running all the time 
- Online: model is running all the time, either through a web service or a streaming service

## Web service 
When users asks for a prediction, the model is applied and the prediction is returned. Model is up and running all the time.

### Saving the model

Save a model using pickle to a .bin file. Alse save dependent variables transformers if they are e.g. vectorizers./ 
Put new observations in dictionaries. 
Put training in separate script: 

- Parameters at top of script
- Save model at end of script
- Use logging (print statements) to log progress
- Predctions in separate script

<pre>
<code>
import pickle
with open('model.bin', 'wb') as f_out:
    pickle.dump((dcit_vectorizer, model), f_out)
f_out.close() ## After opening any file it's nessecery to close it

with open('mode.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    dict_vectorizer, model = pickle.load(f_in)
f_in.close()

</code>
</pre>
  
To check your version of the package used to train the model, use the following code:

<pre>
<code>
pip freeze | grep scikit-learn
</pre>
</code>

This is necessary to make sure your model will unpickle right. Then you can install the same version of the package in your production environment using the following command:

<pre>
<code>
pipenv install scikit-learn==0.23.2 --python=3.8
</pre>
</code>

### Flask  

Users can send requests and get things back. 
  - A web service is a method used to communicate between electronic devices.
  - There are some methods in web services we can use it to satisfy our problems. Here below we would list some.
    - **GET:**  GET is a method used to retrieve files, For example when we are searching for a cat image in google we are actually requesting cat images with GET method.
    - **POST:** POST is the second common method used in web services. For example in a sign up process, when we are submiting our name, username, passwords, etc we are posting our data to a server that is using the web service. (Note that there is no specification where the data goes)
    -  **PUT:** PUT is same as POST but we are specifying where the data is going to.
    -  **DELETE:** DELETE is a method that is used to request to delete some data from the server.

<pre>
<code>
from flask import Flask, request, jsonify 
import pickle

with open('churn-model.bin', 'rb') as f_in:
  dv, model = pickle.load(f_in)

def predict_single(customer, dv, model):
  X = dv.transform([customer])  ## apply the one-hot encoding feature to the customer data 
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]

app = Flask('churn') ## create a flask app with the name churn

@app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
def predict():
    customer = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction), ## we need to conver numpy data into python data in flask framework
        'churn': bool(churn),  ## same as the line above, converting the data using bool method
    }

    return jsonify(result)  ## send back the data in json format to the user

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696) ## run the app on port 9696
</code>
</pre>

To run the service, run the file with the code above in the command line. Flask automatically detects changes in the source file and reloads it. To test the service, we can use the following code:

<pre>
<code>
## a new customer informations
customer = {
  'customerid': '8879-zkjof',
  'gender': 'female',
  'seniorcitizen': 0,
  'partner': 'no',
  'dependents': 'no',
  'tenure': 41,
  'phoneservice': 'yes',
  'multiplelines': 'no',
  'internetservice': 'dsl',
  'onlinesecurity': 'yes',
  'onlinebackup': 'no',
  'deviceprotection': 'yes',
  'techsupport': 'yes',
  'streamingtv': 'yes',
  'streamingmovies': 'yes',
  'contract': 'one_year',
  'paperlessbilling': 'yes',
  'paymentmethod': 'bank_transfer_(automatic)',
  'monthlycharges': 79.85,
  'totalcharges': 3320.75
}

import requests ## to use the POST method we use a library named requests
url = 'http://localhost:9696/predict' ## this is the route we made for prediction
response = requests.post(url, json=customer) ## post the customer information in json format
result = response.json() ## get the server response
print(result)

</code>
</pre>

 - Until here we saw how we made a simple web server that predicts the churn value for every user. When you run your app you will see a warning that it is not a WGSI server and not suitable for production environmnets. To fix this issue and run this as a production server there are plenty of ways available. 
   - One way to create a WSGI server is to use gunicorn. To install it use the command ```pip install gunicorn```, And to run the WGSI server you can simply run it with the   command ```gunicorn --bind 0.0.0.0:9696 churn:app```. Note that in __churn:app__ the name churn is the name we set for our the file containing the code ```app = Flask('churn')```(for example: churn.py), You may need to change it to whatever you named your Flask app file.  
   -  Windows users may not be able to use gunicorn library because windows system do not support some dependencies of the library. So to be able to run this on a windows machine, there is an alternative library waitress and to install it just use the command ```pip install waitress```. 
   -  to run the waitress wgsi server use the command ```waitress-serve --listen=0.0.0.0:9696 churn:app```.
   -  To test it just you can run the code above and the results is the same.
 - So until here you were able to make a production server that predict the churn value for new customers. In the next session we can see how to solve library version conflictions in each machine and manage the dependencies for production environments.


 ### Docker
- Once our project was packed in a Docker container, we're able to run our project on any machine.
- First we have to make a Docker image. In Docker image file there are settings and dependecies we have in our project. To find Docker images that you need you can simply search the [Docker](https://hub.docker.com/search?type=image) website.

Here a Dockerfile (There should be no comments in Dockerfile, so remove the comments when you copy)

<code>
<pre>
# First install the python 3.8, the slim version uses less space
FROM python:3.8.12-slim

# Install pipenv library in Docker 
RUN pip install -U pip
RUN pip install pipenv

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                
# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["*.py", "churn-model.bin", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "churn_serving:app"]

</pre>
</code>

The flags `--deploy` and `--system` makes sure that we install the dependencies directly inside the Docker container without creating an additional virtual environment (which `pipenv` does by default). 

If we don't put the last line `ENTRYPOINT`, we will be in a python shell.
Note that for the entrypoint, we put our commands in double quotes.

After creating the Dockerfile, we need to build it:

```bash
docker build -t churn-prediction:v1 .
```

To run it,  execute the command below:

```bash
docker run -it -p 9696:9696 churn-prediction:latest
```

Flag explanations: 

- `-t`: is used for specifying the tag name "churn-prediction".
- `-it`: in order for Docker to allow us access to the terminal (interactive mode).
- `--rm`: allows us to remove the image from the system after we're done.  
- `-p`: to map the 9696 port of the Docker to 9696 port of our machine. (first 9696 is the port number of our machine and the last one is Docker container port.)
- `--entrypoint=bash`: After running Docker, we will now be able to communicate with the container using bash (as you would normally do with the Terminal). Default is `python`.

### Loading from mlflow

You can make scikit-learn pipelines to make loading preprocessing steps easier.

<pre>
<code>
from sklearn.pipeline import make_pipeline

with mlflow.start_run():
    params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
    mlflow.log_params(params)

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(**params, n_jobs=-1)
    )

    pipeline.fit(dict_train, y_train)
    y_pred = pipeline.predict(dict_val)

    rmse = mean_squared_error(y_pred, y_val, squared=False)
    print(params, rmse)
    mlflow.log_metric('rmse', rmse)

    mlflow.sklearn.log_model(pipeline, artifact_path="model")
</code>
</pre>

To load a model directly from MLFlow, use the following code:

<pre>
<code>
RUN_ID = os.getenv('RUN_ID')

logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)
</code>
</pre>

Instead of referring to the tracking server, we directly point to the storage of the model. This prevents an overreliance on the tracking server. You can also point to the model version directly. For this, read the documentation on mlflow.pyfunc.load_model.

## Streaming 
There is no 1-to-1 relationship between input and output. There is a 1-to-many relationship. A request is used by different models to make predictions. The connection is more implicit. Back-end is connected to a data stream, the data stream feeds multiple applications. It is less important how many services are connected to the stream. They can be added or removed at any time.

 ### Lambda 
 Run code without thinking about servers. It's executed somewhere. You first need to create a role to define what authorization the lambda function has. Then you can create a lambda function in the UI. In the function code, you can write what needs to be done. the lambda_handler gives the output. It takes an event and a context. The event is the input. The context is the runtime information. 

 Next, you create a kinesis stream to deliver the data. To send data to kinesis, use the followin code in the command line: 

<pre>
<code>
aws kinesis put-record --stream-name my-stream 
--data '{
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66
        }, 
        "ride_id": 156
    }'
--partition-key 123
</code>
</pre>

Then, you can filter out the data in the lambda function like so 

<pre>
<code>
def lambda_handler(event, context):
    # print(json.dumps(event)) --> use this to print the event as it is sent by kinesis and make your own test event out of it
    
    predictions_events = []
    
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)

        # print(ride_event)
        ride = ride_event['ride']
        ride_id = ride_event['ride_id']
    
        features = prepare_features(ride)
        prediction = predict(features)
    
        prediction_event = {
            'model': 'ride_duration_prediction_model',
            'version': '123',
            'prediction': {
                'ride_duration': prediction,
                'ride_id': ride_id   
            }
        }
        
        predictions_events.append(prediction_event)


    return {
        'predictions': predictions_events
    }
</code>
</pre>

Test events look like this

<pre>
<code>

{
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1654161514.132
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::XXXXXXXXX:role/lambda-kinesis-role",
            "awsRegion": "eu-west-1",
            "eventSourceARN": "arn:aws:kinesis:eu-west-1:XXXXXXXXX:stream/ride_events"
        }
    ]
}

</code>
</pre>

Yet usually an end user cannot see the output of a function. So we must output it to another kinesis stream: 

<pre>
<code>

kinesis_client = boto3.client('kinesis')
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')

 def lambda_handler(event, context):
    kinesis_client.put_record(
        StreamName=PREDICTIONS_STREAM_NAME,
        Data=json.dumps(prediction_event),
        PartitionKey=str(ride_id)
    )

</code>
</pre>

We can then package everything using Docker and deploy it to AWS. To do so, we install a specific version of python in the Docker file (and do not need to specify a working directory)

<pre>
<code>
FROM public.ecr.aws/lambda/python:3.9

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "lambda_function.py", "./" ]

CMD [ "lambda_function.lambda_handler" ] #where to find the lambda function

</code>
</pre>

Then, we can build the image and push it to ECR.

<pre>
<code>
docker build -t stream-model-duration:v1 .

docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="eu-west-1" \
    stream-model-duration:v1
</code>
</pre>

You can then register the image in ECR and create a lambda function using the image.

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

You can then use Prefect to create a flow that runs the script. Decorate the run function with `@task` and create a flow using `@flow`. Your main function defines the flow of individual functions. Each of the subfunctions are tasks. To log the flow to Prefect import the `logger` and use `logger.info("message")`. To run the flow, use `flow.run()`.

You can then start the project and deploy the flow by typing 


<pre>
<code> 
prefect project init
prefect deploy file.py:flow -n flow-name -p local-work-pool #register flow
prefect worker start -t process -p local-work-pool #start worker
</code>
</pre>

You can do ad hoc runs from the UI. But you can also schedule one from the UI or programmatically. The latter is done using the deployment.yaml file. 