# Monitoring
You can check for 
- Service health (e.g., uptime, latency, throughput)
- Data quality and integrity (e.g., missing values, value range)
- Model quality (e.g., accuracy, precision, recall, F1 score)
- Data and concept drift (e.g., distribution of features, target, and predictions)

And others, such as 
- Performance by segment 
- Model bias/fairness
- Outliers
- Explainability (e.g., feature importance, SHAP values)

## Batch vs. online 
You can add ML metrics to already existing service health monitoring (like Grafana) or use a dedicated dashboard. 

### Batch model
Some statistics can only be calculated in batch mode using past batches or the training data: 
- expected data quality (e.g., missing values, value range)
- data distribution 
- descriptive statistics (e.g., mean, median, standard deviation)

### Online model
In online mode, things are a little bit more complicated.
- descriptive statistics (e.g., mean, median, standard deviation): calculate continuously or incrementally
- statistical tests (e.g., t-test, chi-square test): pick a window (with or without moving reference) and compare windows

## Monitoring scheme
The monitoring scheme can look like this 
- Request and reponse from service 
- Prediction logs used to calculate metrics using prefect and evidently 
- Build monitoring infrastructure using Grafana 

After you've activated your environment, make a docker file. Docker makes YAML files that list all services and their dependencies.

<pre>
<code>
version: '3.7'

volumes:
    grafana_data: {} #artifacts will be stores here

networks: 
  front-tier:
  back-tier:

services:
    db: 
        image: postgres
        restart: always
        environment: #environment variable
            POSTGRES_PASSWORD: example
        ports:
            - "5432:5432"
        networks: 
            - back-tier #not accessed from browser
    
    adminer: #manage content of db 
        image: adminer
        restart: always
        ports:
            - "8080:8080"
        networks:
            - back-tier
            - front-tier
    
    grafana: #visualize data
        image: grafana/grafana
        user: "472"
        ports:
            - "3000:3000"
        volumes:
            - ./config/grafana_datasources.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro #data source
            - ./config/grafana_dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro #for daashboard 
            - ./dashboards:/opt/grafana/dashboards
        networks:
            - back-tier #communication between db and dashboard
            - front-tier #access dashboard
        restart: always 
</code>
</pre>

Also make a datasources file for grafana. This should be a the location specified in the docker file.

<pre>
<code>
# config file version
apiVersion: 1

# list of datasources that should be deleted from the database
deleteDatasources:
  - name: Prometheus
    orgId: 1

# list of datasources to insert/update depending
# what's available in the database
datasources:
  - name: PostgreSQL
    type: postgres
    access: proxy
    url: db.:5432 #equal to docker file 
    database: test
    user: postgres
    secureJsonData:
      password: 'example' #same as in docker file 
    jsonData:
      sslmode: 'disable'

</code>
</pre>

Now run docker compose up. This will create the containers.

<pre>
<code>
docker-compose up --build
</code>
</pre>

Now you can access the dashboard at localhost:3000. The default username and password is admin. You can change this in the docker file.

Now write a script to download the data. 

<pre>
<code>
files = [{'green_tripdata_2022-02.parquet', './data'}, {'green_tripdata_2022-01.parquet', './data'}]

print('Downloading data...')
for file, path in files:
    url = f'https://nyc-tlc.s3.amazonaws.com/trip+data/{file}'
    resp = requests.get(url, stream=True)
    save_path = f'{path}/{file}'
    with open(save_path, 'wb') as handle:
        for data in tqdm(resp.iter_content(), 
                        desc=f'{file}', 
                        postfix=f'save to {save_path}',
                        totam=int(resp.headers['content-length'])):
            handle.write(data)

</code>
</pre>

Then write a script to clean the data and train the model. In the last part of the script, dump the model and create a reference dataset. 

<pre>
<code>
with open('models/model.pkl', 'wb') as handle:
    pickle.dump(model, handle)

val_data.to_parquet('data/reference_data.parquet')
</code>
</pre>

Then create a monitoring script. 

<pre>
<code>
import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

reference_data = pd.read_parquet('data/reference.parquet')
with open('models/lin_reg.bin', 'rb') as f_in:
	model = joblib.load(f_in)

raw_data = pd.read_parquet('data/green_tripdata_2022-02.parquet') #this can be replaced with the pipeline data if you use this 

begin = datetime.datetime(2022, 2, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task #to create a pipeline, data is not read from the internet here
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn: #same as in docker file above
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'") #check if database 'test' exists
		if len(res.fetchall()) == 0: #if database doesn't exist, create it
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, i): #curr = current position of the cursor 
	current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
		(raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

	current_data.fillna(0, inplace=True)
	current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

	report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

	curr.execute(
		"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
	)



@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10) #approximate time of last send
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 100):
			with conn.cursor() as curr:
				calculate_dummy_metrics_postgresql(curr)

            #allow for some time between sends 
			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()

</code>
</pre>

If your run this script (when the docker image has been builft), you can login to adminer (localhost:8080, server db, username postgres, password example, database test) and see the data in the dummy_metrics table. Log into grafana (localhost:3000, username admin, password admin) and create a new panel to start displaying the data.

To save the grafana dashboard, create a new config file: 

<pre>
<code>
apiVersion: 1

providers:
  # <string> an unique provider name. Required
  - name: 'Evidently Dashboards'
    # <int> Org id. Default to 1
    orgId: 1
    # <string> name of the dashboard folder.
    folder: ''
    # <string> folder UID. will be automatically generated if not specified
    folderUid: ''
    # <string> provider type. Default to 'file'
    type: file
    # <bool> disable dashboard deletion
    disableDeletion: false
    # <int> how often Grafana will scan for changed dashboards
    updateIntervalSeconds: 10
    # <bool> allow updating provisioned dashboards from the UI
    allowUiUpdates: false
    options:
      # <string, required> path to dashboard files on disk. Required when using the 'file' type
      path: /opt/grafana/dashboards
      # <bool> use folder names from filesystem to create folders in Grafana
      foldersFromFilesStructure: true

</code>
</pre>

Then also save a json file of the dashboard. This can be empty at first. You can copy the json from the dashboard in grafana and put it in the file. Then add this to the docker file (already done above).

## Debugging 
If your metrics go above the thresholds you set, you can debug the model.
Test suites contain different tests to compare metrics to thresholds. Metric presets are well combined metrics that help you make good reports. 

<pre>
<code>
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

problematic_data = current_data.loc[(current_data.lpep_pickup_datetime >= datetime.datetime(2022,2,2,0,0)) & 
                               (current_data.lpep_pickup_datetime < datetime.datetime(2022,2,3,0,0))]

column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

problematic_data['prediction'] = model.predict(problematic_data[num_features + cat_features].fillna(0))

test_suite = TestSuite(tests = [DataDriftTestPreset()]) #you can change these to other tests
test_suite.run(reference_data=ref_data, current_data=problematic_data, column_mapping=column_mapping)

test_suite.show(mode='inline')

report = Report(metrics = [DataDriftPreset()])
report.run(reference_data=ref_data, current_data=problematic_data, column_mapping=column_mapping)

report.show(mode='inline')

#or use report.save('report.html') to save the report to a file

</code>
</pre>

