# CI/CD
## Pre-commit hooks
A pre-commit hook is a script that runs before you commit. It can be used to check for formatting, linting, etc.
To install the tool, code 

<pre>
<code>
pipenv install --dev pre-commit 
</code>
</pre>

Git hooks are run for Git repos. To make a file, use 

<pre>
<code>
pre-commit sample-config > .pre-commit-config.yaml 
pre-commit install #is in the .git-folder. so all devs need to run this locally once after setting up the repo

ls .git/hooks/ #now contains sample files and your pre-commit hook
less /git/hooks/pre-commit #to see what it does
</code>
</pre>

This gives you a file that you can adapt with hooks from pre-commit.com/hooks.html. If you push to the repo, this hook will be run. 

To add isort etc (you can find these things online), 

<pre>
<code>
- repo: https://github.com/pycqa/isort
  rev: 5.10.1
  hooks:
    - id: isort
      name: isort (python)
- repo: https://github.com/psf/black
  rev: 22.6.0
  hooks:
    - id: black
      language_version: python3.9
- repo: local
  hooks:
    - id: pylint
      name: pylint
      entry: pylint
      language: system
      types: [python]
      args: [
        "-rn", # Only display messages
        "-sn", # Don't display the score
        "--recursive=y"
      ]
- repo: local
  hooks:
    - id: pytest-check
      name: pytest-check
      entry: pytest
      language: system
      pass_filenames: false
      always_run: true
      args: [
        "tests/"
      ]
</code>
</pre>

## Makefiles
Install make using 

<pre>
<code>
sudo apt install make
</code>
</pre>

An example of a Makefile 

<pre>
<code>
LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=stream-model-duration:${LOCAL_TAG}

test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test #depends on quality checks and test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integraton-test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install
</code>
</pre>

You can run individual commands using

<pre>
<code>
make quality_checks
</code>
</pre>

## Continuous integration 
Before a pull request is merged, unit and integration tests are run. In addition, the terraform plan is initiated and reviewed. 
With GitHub Actions, you can automate this process without setting up VMs to execute the tests.

First create a folder .github/workflows. In this folder, create two files: ci.yml and cd.yml.
In the CI file, you can define the steps that need to be taken before a pull request is merged. This file looks as follows: 

<pre>
<code>
name: CI
on:
    pull_request: #request that triggers the workflow
        - branches: [ main ] 
    paths: #only run the workflow when these paths are changed
        - 'terraform/**'
        - 'src/**'
        - 'tests/**'
        - 'Makefile'
        - '.pre-commit-config.yaml'

env: #variables from github secrets
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    AWS_DEFAULT_REGION: eu-west-1
    AWS_ACCOUNT_ID: ${{ secrets.AWS_ACCOUNT_ID }}

jobs:
    test:
        runs-on: ubuntu-latest #type of machine to run the workflow on, from github or self-run 
        steps:
            - uses: actions/checkout@v2 #checkout repo to run the workflow on, clone to virtual machine and install docker 
            - name: Set up Python 3.9
              uses: actions/setup-python@v2
              with:
                  python-version: 3.9
                  
            - name: Install dependencies
              working-directory: 06-best-practices/code"
              run: pip install pipenv && pipenv install --dev

            - name: Run tests
              working-directory: 06-best-practices/code"
              run: pipenc run pytest tests/

            - name: Linting
              working-directory: 06-best-practices/code"
              run: pipenv run pylint --recursive=y 
            
            - name: Configure AWS credentials
              uses: aws-actions/configure-aws-credentials@v1
              with: 
                    aws-access-key-id: ${{ env.AWS_ACCESS_KEY_ID }}
                    aws-secret-access-key: ${{ env.AWS_SECRET_ACCESS_KEY }}
                    aws-region: eu-west-1

            - name: Integration Test
                working-directory: 06-best-practices/code"
                run: |
                    . run.sh
            
    tf-plan: #jobs run in parallel, so dependencies need to be defined again 
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v2
            - name: Configure AWS credentials
              uses: aws-actions/configure-aws-credentials@v1
              with: 
                    aws-access-key-id: ${{ env.AWS_ACCESS_KEY_ID }}
                    aws-secret-access-key: ${{ env.AWS_SECRET_ACCESS_KEY }}
                    aws-region: eu-west-1
            
            - uses: hashicorp/setup-terraform@v2

            - name: TF plan 
              id: plan 
              working-directory: "06-best-practices/code/terraform"
              run: |
                terraform init -backend-config="key=mlops-zoomcamp-prod.tfstate" --reconfigure #points to staging file in main.tf, also creata a new state without erasing the old one
                terraform plan --var-file vars/prod.tfvars 

</code>
</pre>

If you commit this file, it will automatically run the workflow.
Also adapt the run.sh-file like so:

<pre>
<code>
if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

</code>
</pre>

In settings on Github, you can add secrets. These are then available in the workflow.

## Continuous development
This stage works in three substages; build infrastructure, build docker image and push it to repo and deploy lambda config etc.
The file looks as follows:

<pre>
<code>
name: CD
on:
    push:
        branches: [ main ]
    paths:
        - 'terraform/**'
        - 'src/**'
        - 'tests/**'
        - 'Makefile'
        - '.pre-commit-config.yaml'

jobs:
  build-push-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repo
        uses: actions/checkout@v3
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: "eu-west-1"
      - uses: hashicorp/setup-terraform@v2
        with:
          terraform_wrapper: false

      # Define the infrastructure
      - name: TF plan
        id: tf-plan
        working-directory: '06-best-practices/code/infrastructure'
        run: |
          terraform init -backend-config="key=mlops-zoomcamp-prod.tfstate" -reconfigure && terraform plan -var-file=vars/prod.tfvars

      - name: TF Apply
        id: tf-apply
        working-directory: '06-best-practices/code/infrastructure'
        if: ${{ steps.tf-plan.outcome }} == 'success' #only do if plan id successful in previous step
        run: |
          terraform apply -auto-approve -var-file=vars/prod.tfvars
          echo "::set-output name=ecr_repo::$(terraform output ecr_repo | xargs)"
          echo "::set-output name=predictions_stream_name::$(terraform output predictions_stream_name | xargs)"
          echo "::set-output name=model_bucket::$(terraform output model_bucket | xargs)"
          echo "::set-output name=lambda_function::$(terraform output lambda_function | xargs)"

      # Build-Push
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, and push image to Amazon ECR
        id: build-image-step
        working-directory: "06-best-practices/code"
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: ${{ steps.tf-apply.outputs.ecr_repo }}
          IMAGE_TAG: "latest"   # ${{ github.sha }}
        run: |
          docker build -t ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG} .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "::set-output name=image_uri::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"

      # Deploy
      - name: Get model artifacts
      # The steps here are not suited for production.
      # In practice, retrieving the latest model version or RUN_ID from a service like MLflow or DVC can also be integrated into a CI/CD pipeline.
      # But due to the limited scope of this workshop, we would be keeping things simple.
      # In practice, you would also have a separate training pipeline to write new model artifacts to your Model Bucket in Prod.

        id: get-model-artifacts
        working-directory: "06-best-practices/code"
        env:
          MODEL_BUCKET_DEV: "mlflow-models-alexey"
          MODEL_BUCKET_PROD: ${{ steps.tf-apply.outputs.model_bucket }}
        run: |
          export RUN_ID=$(aws s3api list-objects-v2 --bucket ${MODEL_BUCKET_DEV} \
          --query 'sort_by(Contents, &LastModified)[-1].Key' --output=text | cut -f2 -d/)
          aws s3 sync s3://${MODEL_BUCKET_DEV} s3://${MODEL_BUCKET_PROD}
          echo "::set-output name=run_id::${RUN_ID}"

      - name: Update Lambda
        env:
          LAMBDA_FUNCTION: ${{ steps.tf-apply.outputs.lambda_function }}
          PREDICTIONS_STREAM_NAME: ${{ steps.tf-apply.outputs.predictions_stream_name }}
          MODEL_BUCKET: ${{ steps.tf-apply.outputs.model_bucket }}
          RUN_ID: ${{ steps.get-model-artifacts.outputs.run_id }}
        run: |
          variables="{ \
                    PREDICTIONS_STREAM_NAME=$PREDICTIONS_STREAM_NAME, MODEL_BUCKET=$MODEL_BUCKET, RUN_ID=$RUN_ID \
                    }"

          STATE=$(aws lambda get-function --function-name $LAMBDA_FUNCTION --region "eu-west-1" --query 'Configuration.LastUpdateStatus' --output text)
              while [[ "$STATE" == "InProgress" ]]
              do
                  echo "sleep 5sec ...."
                  sleep 5s
                  STATE=$(aws lambda get-function --function-name $LAMBDA_FUNCTION --region "eu-west-1" --query 'Configuration.LastUpdateStatus' --output text)
                  echo $STATE
              done

          aws lambda update-function-configuration --function-name $LAMBDA_FUNCTION \
                    --environment "Variables=${variables}"

</code>
</pre>

