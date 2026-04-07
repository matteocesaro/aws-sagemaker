# AWS SageMaker Project Setup

This repository contains the configuration and scripts to interact with **Amazon SageMaker** using the AWS SDK for Python (`boto3`) and (`sagemaker`).

---

## 1. IAM User Configuration (AWS Console)

Before running the script, ensure your IAM user has the necessary permissions and valid access keys.

### Permissions

Your IAM user must have the following policies attached:

- **AdministratorAccess**
- **AmazonSageMakerFullAccess**
- **AmazonSageMaker-ExecutionPolicy-2026...**
- **IAMUserChangePassword**

---

### Generate Access Keys

1. Log in to the **AWS Management Console** and navigate to **IAM**  
2. In the left sidebar, click **Users** and select your user  
3. Go to the **Security credentials** tab  
4. Scroll to **Access keys** → click **Create access key**  
5. Select **Command Line Interface (CLI)**  
6. ⚠️ **Important:** Download the `.csv` or copy:
   - Access Key ID  
   - Secret Access Key  
   *(You won't be able to see the Secret Key again)*

---

## 2. Local Environment Setup (Terminal)

Make sure you have the AWS CLI installed:  
https://aws.amazon.com/cli/

Run:

```bash
aws configure --profile aws-project
```

Provide:

```
AWS Access Key ID: [Paste your Access Key]
AWS Secret Access Key: [Paste your Secret Key]
Default region name: us-east-1
Default output format: json
```

---

## 3. Configuration Storage

Credentials are stored locally:

### Windows
```
%USERPROFILE%\.aws\credentials
```

### macOS/Linux
```
~/.aws/credentials
```

### Example content
```ini
[aws-project]
aws_access_key_id = AKIAXXXXXXXXXXXXXXXX
aws_secret_access_key = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 4. Python Implementation

Example script using `boto3` and `sagemaker`:

```python
import boto3
import sagemaker

# Initialize session with local profile
boto3.setup_default_session(
    profile_name="aws-project",
    region_name="us-east-1"
)

# SageMaker setup
sm_boto3 = boto3.client("sagemaker")
bucket = "bucketsagemaker-firsttest"

sess = sagemaker.Session(default_bucket=bucket)

print(f"Using bucket: {bucket}")
print(f"Region: {sess.boto_session.region_name}")
```

---

## 5. Troubleshooting & Verification

To verify your AWS profile:

```bash
aws sts get-caller-identity --profile aws-project
```

If everything is correct, you'll see:
- Account ID  
- User ARN  

---

## ⚠️ Security Warning

**Never commit sensitive credentials to Git.**

Add to your `.gitignore`:

```gitignore
.aws/
*.csv
```

## Model Training on AWS SageMaker

This project uses **AWS SageMaker** to move model training from your local computer to the cloud. Instead of running the training on your PC:

1. SageMaker launches an EC2 instance (e.g., `ml.m5.large`) on AWS.
2. It pulls a **Docker container** with Python and all required libraries already installed (in this case, `scikit-learn`).
3. Copies your training and test data from S3 to the remote instance.
4. Executes the **`script.py`** inside the container.
5. Saves the trained model (`model.joblib`) back to S3.
6. Automatically shuts down the instance after training is complete.

---

### The `script.py`

This script handles:

- Reading training and test data from CSV files.
- Splitting into **features** and **label**.
- Training a **RandomForestRegressor** from `scikit-learn`.
- Evaluating the model on the test set (MAE, R²).
- Saving the trained model as `model.joblib`.

It also defines the `model_fn(model_dir)` function, which SageMaker uses to load the model during deployment.

---

### Initializing and configuring Training Job with SageMaker

The `SKLearn` estimator class is used to run training in the cloud:

```python
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point="script.py",  # the script SageMaker runs inside the container
    role="...",
    instance_count=1,
    instance_type="ml.m5.large",  # EC2 instance type
    framework_version="0.23-1",  # sklearn container version
    base_job_name="RF-custom-sklearn",
    hyperparameters={
        "n_estimators": 10,
        "random_state": 0
    },
    use_spot_instances=False,  # if True, uses cheaper spot instances
    max_run=3600  # maximum runtime of 1 hour
)
```
What happens under the hood:

SageMaker creates an isolated EC2 instance.
Launches a Docker container with Python, sklearn, and all dependencies.
Copies data from S3 to the container.
Runs the training as if it were local, using script.py.
Saves the trained model to S3.
Shuts down the instance automatically after the job completes.

Essentially, you are running your script inside a remote container, without installing libraries or consuming resources on your local machine.
Docker is a container with
- python
- libraries
- script
The training job is configured with SKLearn as follows:

---

### Launching Training Job async

```python
sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=False, logs=False)

config_path = os.path.expanduser("~\\.sagemaker-code-config")
with open(config_path, 'w') as f:
    import json
    json.dump({}, f)
os.environ["SAGEMAKER_CODE_CONFIG_PATH"] = config_path
```

---

### Retrieve the model artifact from S3 and wrap the model into a deployable SageMaker SKLearnModel

```python
sklearn_estimator.latest_training_job.wait(logs="None")
artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]
```


```python
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime

model_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model = SKLearnModel(
    name=model_name,
    model_data=artifact,
    role=ROLE,
    entry_point="script.py",
    framework_version=FRAMEWORK_VERSION
)
```

---

### Endpoint + predictions + delete endpoint

```python
endpoint_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name
)

X_input = X.drop(columns=["Fuel_Price_Index"])
print(predictor.predict(X_input[0:2].values.tolist()))


sm_boto3.delete_endpoint(EndpointName=endpoint_name)
```

---
