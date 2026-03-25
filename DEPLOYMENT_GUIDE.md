# Diabetes Prediction API - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Diabetes Prediction API built with FastAPI and XGBoost to AWS and other platforms.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [AWS EC2 Deployment](#aws-ec2-deployment)
4. [AWS ECS Deployment](#aws-ecs-deployment)
5. [AWS Lambda Deployment](#aws-lambda-deployment)
6. [API Usage](#api-usage)

---

## Local Deployment

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Trained model files in `models/` directory

### Steps

1. **Train and Save the Model**

```bash
python train_and_save_model.py
```

This creates the `models/` directory with:
- `xgboost_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoders.pkl` - Categorical encoders
- `feature_names.pkl` - Feature names list
- `model_metadata.pkl` - Model metadata

2. **Install Dependencies**

```bash
pip install -r requirements_api.txt
```

3. **Run the API**

```bash
python main.py
```

The API will be available at:
- **Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Docker Deployment

### Prerequisites

- Docker installed
- Docker Compose (optional, for container orchestration)

### Build and Run with Docker

1. **Build the Docker Image**

```bash
docker build -t diabetes-prediction-api:latest .
```

2. **Run the Container**

```bash
docker run -d \
  --name diabetes-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  diabetes-prediction-api:latest
```

3. **Verify Container is Running**

```bash
docker ps
curl http://localhost:8000/health
```

### Docker Compose (Optional)

1. **Start All Services**

```bash
docker-compose up -d
```

2. **View Logs**

```bash
docker-compose logs -f diabetes-api
```

3. **Stop Services**

```bash
docker-compose down
```

---

## AWS EC2 Deployment

### Prerequisites

- AWS account with EC2 access
- EC2 instance (Ubuntu 22.04 LTS recommended)
- Security group with inbound rules for port 8000
- SSH key pair for EC2 access

### Step-by-Step Deployment

1. **Launch EC2 Instance**

   - AMI: Ubuntu Server 22.04 LTS (Free Tier eligible)
   - Instance Type: t2.micro or t3.micro
   - Security Group: Allow inbound on port 8000 (0.0.0.0/0)
   - Storage: 20 GB SSD

2. **Connect to EC2 Instance**

```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

3. **Update System**

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

4. **Install Dependencies**

```bash
sudo apt-get install -y python3.11 python3-pip git curl
python3 -m pip install --upgrade pip
```

5. **Clone/Upload Your Project**

```bash
# Option 1: Clone from GitHub
git clone <your-repo-url>
cd <repo-name>

# Option 2: Upload via SCP
scp -i your-key.pem -r ./diabetes-api ubuntu@<EC2-PUBLIC-IP>:/home/ubuntu/
```

6. **Install Python Dependencies**

```bash
cd ~/diabetes-api
pip install -r requirements_api.txt
```

7. **Run the API with Gunicorn (Production)**

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

8. **Set Up Systemd Service (Optional, for Auto-Start)**

Create `/etc/systemd/system/diabetes-api.service`:

```ini
[Unit]
Description=Diabetes Prediction API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/diabetes-api
ExecStart=/usr/local/bin/gunicorn -w 4 -b 0.0.0.0:8000 main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable diabetes-api
sudo systemctl start diabetes-api
sudo systemctl status diabetes-api
```

9. **Access the API**

```bash
# From your local machine
curl http://<EC2-PUBLIC-IP>:8000/health
```

---

## AWS ECS Deployment (Recommended)

### Prerequisites

- AWS ECR (Elastic Container Registry) access
- AWS ECS (Elastic Container Service) cluster
- AWS IAM permissions

### Steps

1. **Push Docker Image to ECR**

```bash
# Set AWS region (replace us-east-1 with your region)
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<your-account-id>
export ECR_REPO=diabetes-prediction-api

# Create ECR repository
aws ecr create-repository \
  --repository-name $ECR_REPO \
  --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push image
docker tag diabetes-prediction-api:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

2. **Create ECS Task Definition**

Create `task-definition.json`:

```json
{
  "family": "diabetes-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "diabetes-api",
      "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/diabetes-prediction-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/diabetes-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "PYTHONUNBUFFERED",
          "value": "1"
        }
      ]
    }
  ],
  "executionRoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/ecsTaskExecutionRole"
}
```

Register the task definition:

```bash
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json
```

3. **Create ECS Service**

```bash
aws ecs create-service \
  --cluster diabetes-cluster \
  --service-name diabetes-api-service \
  --task-definition diabetes-api:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxxx],securityGroups=[sg-xxxxxx],assignPublicIp=ENABLED}"
```

---

## AWS Lambda Deployment

For serverless deployment, use AWS Lambda with API Gateway. This requires a modified Mangum wrapper:

1. **Install Mangum**

```bash
pip install mangum
```

2. **Modify main.py for Lambda**

```python
from mangum import Mangum

handler = Mangum(app)
```

3. **Create Lambda function** via AWS Console or AWS CLI
4. **Set up API Gateway** to trigger the Lambda function

---

## API Usage

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

### Get Model Info

```bash
curl -X GET http://localhost:8000/model-info
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 25.5,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 120
  }'
```

### Batch Predictions

```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "gender": "Male",
        "age": 45.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 25.5,
        "HbA1c_level": 5.8,
        "blood_glucose_level": 120
      },
      {
        "gender": "Female",
        "age": 55.0,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "former",
        "bmi": 28.2,
        "HbA1c_level": 6.5,
        "blood_glucose_level": 150
      }
    ]
  }'
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` to test endpoints interactively using Swagger UI.

---

## Monitoring and Logging

### Local Logging

API logs are printed to console. Enable file logging by modifying `main.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    filename='api.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### CloudWatch Logs (AWS)

Ensure your ECS task has proper IAM permissions and view logs in AWS CloudWatch:

```bash
aws logs tail /ecs/diabetes-api --follow
```

---

## Performance Optimization

### Load Balancing

For high traffic, use AWS ELB (Elastic Load Balancer):

```bash
aws elbv2 create-load-balancer \
  --name diabetes-api-lb \
  --subnets subnet-xxxxx subnet-xxxxx \
  --scheme internet-facing
```

### Auto Scaling

Configure ECS Auto Scaling:

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/diabetes-cluster/diabetes-api-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 \
  --max-capacity 10
```

---

## Security Best Practices

1. **Use HTTPS** - Set up SSL certificates with AWS Certificate Manager
2. **API Keys** - Add authentication to main.py:

```python
from fastapi import Depends, HTTPException, Header

async def verify_api_key(x_token: str = Header(...)):
    if x_token != "your-secret-api-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_token
```

3. **Rate Limiting** - Install and use `slowapi`:

```bash
pip install slowapi
```

4. **CORS** - Already configured, adjust as needed
5. **Environment Variables** - Use AWS Secrets Manager for sensitive data

---

## Troubleshooting

### API Not Responding

```bash
# Check if service is running
curl -v http://localhost:8000/health

# Check logs
docker logs diabetes-api
```

### Model Loading Error

```bash
# Verify model files exist
ls -la models/

# Check file permissions
chmod 644 models/*
```

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python main.py --port 8001
```

---

## Contact & Support

For issues or questions, refer to:
- FastAPI Docs: https://fastapi.tiangolo.com/
- AWS Documentation: https://docs.aws.amazon.com/
- XGBoost Documentation: https://xgboost.readthedocs.io/

---

## License

This project is provided as-is for educational and commercial use.
