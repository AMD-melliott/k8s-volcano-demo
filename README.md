# Volcano AMD GPU Demo: Distributed PyTorch Training with ROCm

This demo showcases Volcano's batch scheduling capabilities using AMD GPUs for distributed PyTorch training with ROCm.

## Prerequisites

- Kubernetes cluster (version 1.16+) with AMD GPUs
- AMD device plugin installed
- (optional) Helm 3 installed

## Installing Volcano

### Using Helm

- Add the Volcano Helm repository:

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
```

- Install Volcano:

```bash
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

### Using YAML files

1. Download the Volcano release:

```bash
wget https://github.com/volcano-sh/volcano/releases/download/v1.7.0/volcano-v1.7.0-linux-amd64.tar.gz
tar -xzvf volcano-v1.7.0-linux-amd64.tar.gz
```

1. Install Volcano:

```bash
kubectl apply -f volcano-v1.7.0-linux-amd64/volcano-crds.yaml
kubectl apply -f volcano-v1.7.0-linux-amd64/volcano-development.yaml
```

## Configuring Volcano

### Configuring Scheduler

- Edit the ConfigMap for the Volcano scheduler:

```bash
kubectl edit configmap volcano-scheduler-configmap -n volcano-system
```

- Modify settings like:
  - Actions
  - Plugins
  - Tiers

### Configuring Admission

- Edit the ConfigMap for Volcano admission:
  
```bash
kubectl edit configmap volcano-admission-configmap -n volcano-system
```

- Adjust settings for job validation and mutation.

### Monitoring

- Check Volcano components:

```bash
kubectl get pods -n volcano-system
```

- View scheduler logs:

```bash
kubectl logs -f deployment/volcano-scheduler -n volcano-system
```

- View controller logs:

```bash
kubectl logs -f deployment/volcano-controllers -n volcano-system
```

## Uninstallation

1. If installed with Helm:

```bash
helm uninstall volcano -n volcano-system
```

1. If installed with YAML:

```bash
kubectl delete -f volcano-v1.7.0-linux-amd64/volcano-development.yaml
kubectl delete -f volcano-v1.7.0-linux-amd64/volcano-crds.yaml
```

Remember to adjust resource requests, limits, and other configurations based on your specific cluster and workload requirements.

## Volcano ROCm GPU Testing

### Verify GPU Availability

- Verify server GPU status:

```bash
sudo amd-smi monitor -putmq
```

- Check that AMD GPUs are recognized:

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable'
```

Look for `amd.com/gpu` in the output.

## Deploy and Monitor

- Run `deploy.sh` to start the demo training job. The job should complete in about 5 minutes.

Monitor GPU utilization:

```bash
sudo watch amd-smi monitor -putmq
```

Output from `amd-smi monitor` will show increased GPU utilization and list running processes

- Check the logs:

```bash
kubectl logs pytorch-distributed-pytorch-worker-0
kubectl logs pytorch-distributed-pytorch-worker-1
```

## Clean up

Delete the job once it has completed and log output has been examined.

```bash
kubectl apply -f pytorch-gpu-job.yaml
```

## Adjusting training parameters

The demo can be scaled by adjusting:

1. The number of replicas in the Volcano job
2. The model architecture in the SimpleModel class
3. The dataset size and batch size
4. The number of training epochs
