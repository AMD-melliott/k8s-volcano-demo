#!/bin/bash
set -e

# Update the ConfigMap
echo "Updating ConfigMap..."
kubectl create configmap pytorch-train-code --from-file=train.py -o yaml --dry-run=client | kubectl apply -f -

# Apply the service first
echo "Applying service..."
kubectl apply -f pytorch-service.yaml

# Delete the existing job if it exists
echo "Deleting existing job..."
kubectl delete -f pytorch-gpu-job.yaml --ignore-not-found

# Apply the job
echo "Creating new job..."
kubectl apply -f pytorch-gpu-job.yaml

# Watch the pods
echo "Watching pods..."
kubectl get pods -w