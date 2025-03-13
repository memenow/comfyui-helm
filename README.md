# ComfyUI Helm Chart

This Helm chart deploys [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a powerful and modular Stable Diffusion GUI, on Kubernetes. The project is open source and welcomes community contributions.

## Overview

The Helm chart is optimized for deploying ComfyUI with GPU support on Kubernetes clusters. It integrates seamlessly with a provided Dockerfile, which builds a CUDA-enabled container image based on Ubuntu 22.04, ensuring optimal performance for ComfyUI.

## Key Features

- **Dockerfile Integration:**  
  - Builds ComfyUI from the official repository.
  - Installs necessary system and Python dependencies.
  - Configures essential environment variables (`NVIDIA_DRIVER_CAPABILITIES`, `LD_PRELOAD`, `PYTHONPATH`, `TORCH_CUDA_ARCH_LIST`, `NVIDIA_VISIBLE_DEVICES`).
  - Runs ComfyUI server with optimized settings:
    ```
    python3 main.py --cuda-malloc --use-pytorch-cross-attention --listen 0.0.0.0 --port 8188
    ```

- **Kubernetes Optimized:**
  - Deploys Kubernetes resources: Deployment, Service, Ingress, HPA, ServiceAccount.
  - Enables NVIDIA GPU support.
  - Exposes application on port 8188.
  - Uses a dynamically generated service account name (`{{ .Release.Name }}-sa` by default).

- **Customizable Image Settings:**  
  Adjust Docker image details in `values.yaml`:
  ```
  image:
    repository: your-registry/comfyui-helm
    tag: your-tag
    pullPolicy: IfNotPresent
  ```

- **Flexible Service Exposure:**  
  Supports ClusterIP, NodePort, LoadBalancer, and Ingress types.

## Prerequisites

- Kubernetes cluster with NVIDIA GPU nodes.
- Helm v3 installed.
- Docker installed for image building.

## Installing and Upgrading the Helm Chart

1. **Configure Chart Version:**  
   Set ComfyUI version in `Chart.yaml` (`appVersion`) and optionally add an icon URL.

2. **Update Image Details:**  
   Modify `values.yaml`:
   ```
   image:
     repository: your-registry/comfyui-helm
     tag: your-tag
     pullPolicy: IfNotPresent
   ```

3. **Lint Chart (Optional):**
   ```
   helm lint .
   ```

4. **Deploy or Upgrade Chart:**

   Install:
   ```
   helm install comfyui-helm . --set image.repository=your-registry/comfyui-helm,image.tag=your-tag
   ```

   Upgrade existing deployment:
   ```
   helm upgrade comfyui-helm . --set image.repository=your-registry/comfyui-helm,image.tag=your-tag
   ```
   
5. **Horizontal Pod Autoscaler (HPA):**
   - HPA is enabled by default, scaling the number of pods based on CPU utilization.
   - You can configure HPA settings in `values.yaml`:
     ```
     autoscaling:
       enabled: true
       minReplicas: 1
       maxReplicas: 100
       targetCPUUtilizationPercentage: 80
     ```
   - To disable HPA, set `autoscaling.enabled` to `false`.

6. **Service Exposure Options:**
   - **ClusterIP:** Default internal access; use port-forwarding externally.
   - **NodePort:** Exposes service externally on node ports.
   - **LoadBalancer:** Automatically provisions external IP if supported.
   - **Ingress:** Enable and configure ingress in `values.yaml`.

## Accessing ComfyUI

### ClusterIP (Port Forwarding)

```
export POD_NAME=$(kubectl get pods -l "app.kubernetes.io/name=comfyui-helm" -o jsonpath="{.items.metadata.name}")
kubectl port-forward $POD_NAME 8188:8188
```
Visit [http://127.0.0.1:8188](http://127.0.0.1:8188).

### NodePort

Retrieve NodePort number:

```
kubectl get svc comfyui-helm -o=jsonpath='{.spec.ports[?(@.name=="http")].nodePort}'
```
Access via `http://:`.

### LoadBalancer

Get external IP:

```
kubectl get svc comfyui-helm -o=jsonpath='{.status.loadBalancer.ingress.ip}'
```
Access via external IP at port 8188.

### Ingress

Configure ingress in `values.yaml`, ensure DNS setup, then access using configured hostname.

## Testing the Chart

Run provided tests:

```
helm test comfyui-helm
```

## Contributing

This project is open source—contributions are encouraged! Fork the repository, submit issues or feature requests, and create pull requests to improve this Helm chart.

See the [LICENSE](LICENSE) file for licensing details.

## About ComfyUI

For more information on ComfyUI, visit the official [ComfyUI GitHub repository](https://github.com/comfyanonymous/ComfyUI).
