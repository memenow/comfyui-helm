# ComfyUI Helm Chart

This Helm chart deploys [ComfyUI](https://github.com/Comfy-Org/ComfyUI), a powerful and modular Stable Diffusion GUI, on Kubernetes. The project is open source and welcomes community contributions.

## Overview

The Helm chart is optimized for deploying ComfyUI with GPU support on Kubernetes clusters. It integrates seamlessly with a provided Dockerfile, which builds a CUDA-enabled container image based on Ubuntu 22.04, ensuring optimal performance for ComfyUI.

## Key Features

- **Dockerfile Integration:**  
  - Builds ComfyUI from the official repository.
  - Installs the curated custom nodes needed by this environment during the image build instead of at container startup.
  - Installs necessary system and Python dependencies.
  - Sets the required environment variables (`NVIDIA_DRIVER_CAPABILITIES`, `LD_PRELOAD`, `PYTHONPATH`, `TORCH_CUDA_ARCH_LIST`) at the image layer. `NVIDIA_VISIBLE_DEVICES` is injected per pod by the NVIDIA device plugin and must not be set in `values.yaml`.
  - Runs ComfyUI server with optimized settings:
    ```
    python3 main.py --listen 0.0.0.0 --port 8188 --cuda-malloc
    ```

- **Kubernetes Optimized:**
  - Deploys Kubernetes resources: Deployment, Service, Ingress, HPA, ServiceAccount.
  - Enables NVIDIA GPU support.
  - Exposes application on port 8188.
  - Uses a generated service account name from the chart fullname helper (override via `serviceAccount.name` in `values.yaml`).

- **Customizable Image Settings:**  
  Adjust Docker image details in `values.yaml`:
  ```
  image:
    repository: ghcr.io/memenow/comfyui-helm
    tag: ""  # Defaults to .Chart.AppVersion (currently v2.0.0)
    pullPolicy: IfNotPresent
  ```

- **Flexible Service Exposure:**  
  Supports ClusterIP, NodePort, LoadBalancer, and Ingress types.

- **ERNIE-Image Ready:**  
  - Follows the latest ComfyUI core, which includes native ERNIE-Image support.
  - Keeps the required text and tokenizer dependencies available in the container image.

- **Custom Nodes Baked In:**  
  - Preinstalls `ComfyUI-Manager`, `ComfyUI-GGUF`, `ComfyUI-KJNodes`, `ComfyUI-LTXVideo`, `ComfyUI-SeedVR2_VideoUpscaler`, `ComfyUI-VideoHelperSuite`, `ComfyUI-WanVideoWrapper`, and `Nvidia_RTX_Nodes_ComfyUI`.
  - Installs each node's Python dependencies during the Docker build so the container can start cleanly without first-run bootstrap steps.

## Prerequisites

- Kubernetes cluster with NVIDIA GPU nodes.
- Helm v3 installed.
- Docker installed for image building.

## Installing and Upgrading the Helm Chart

1. **(Maintainers only) Bump Chart Version:**
   When you ship a new image, bump `Chart.yaml` `appVersion` to match the image tag and increment `version` per [SemVer](https://semver.org/). Application users do not need to touch this file.

2. **Update Image Details:**  
   Build the image locally with the GHCR-style name and tag that match the chart app version:
   ```
   docker build -t ghcr.io/memenow/comfyui-helm:v2.0.0 .
   ```

   You do not need to push the image if your cluster can access the local image cache.

   If you want a reproducible build, you can pin the ComfyUI core or any curated custom node to a specific revision:
   ```
   docker build \
     --build-arg COMFYUI_REF=<comfyui-commit> \
     --build-arg COMFYUI_LTXVIDEO_REF=<ltxvideo-ref> \
     -t ghcr.io/memenow/comfyui-helm:v2.0.0 .
   ```

   Modify `values.yaml` only if you need a different repository or tag:
   ```
   image:
     repository: ghcr.io/memenow/comfyui-helm
     tag: ""  # Defaults to .Chart.AppVersion (currently v2.0.0)
     pullPolicy: IfNotPresent
   ```

3. **Lint Chart (Optional):**
   ```
   helm lint .
   ```

4. **Deploy or Upgrade Chart:**

   Install:
   ```
   helm install comfyui-helm .
   ```

   Upgrade existing deployment:
   ```
   helm upgrade comfyui-helm .
   ```
   
5. **Horizontal Pod Autoscaler (HPA):**
   - HPA is **disabled by default**. ComfyUI is GPU-bound, so CPU/memory utilization is a poor scaling signal: replicas added under CPU pressure typically end up `Pending` because no GPU is free, and the per-replica model load makes scale-up slow.
   - For real autoscaling on GPU workloads, prefer [KEDA](https://keda.sh/) with the [NVIDIA DCGM exporter](https://github.com/NVIDIA/dcgm-exporter) (`DCGM_FI_DEV_GPU_UTIL`) or a queue-depth metric. KEDA also supports scale-to-zero, which is useful when the cluster is shared with other workloads.
   - The simple Resource-based HPA is still wired up for users who want it:
     ```yaml
     autoscaling:
       enabled: true
       minReplicas: 1
       maxReplicas: 5
       targetCPUUtilizationPercentage: 80
     ```

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

Configure ingress in `values.yaml`, ensure DNS setup, then access using the configured hostname.

ComfyUI uses a WebSocket connection at `/ws` to stream progress updates and previews. Most ingress controllers buffer responses or apply short read timeouts that break this channel; set the controller-specific annotations below.

NGINX Ingress (`ingress-nginx`):

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    nginx.ingress.kubernetes.io/proxy-body-size: "200m"
```

Traefik (IngressRoute or `ingress.annotations`):

```yaml
ingress:
  annotations:
    traefik.ingress.kubernetes.io/router.middlewares: "default-comfyui-buffering@kubernetescrd"
```

…and apply a Traefik middleware that disables response buffering and lifts the read timeout. For other controllers, look up their equivalents for "disable proxy buffering" and "WebSocket idle timeout".

## Persistent Storage

Models, generated outputs, and user state are lost on pod restart unless backed by a `PersistentVolumeClaim`. The chart exposes four opt-in mounts under `persistence.*`; each one creates its own claim (or reuses an `existingClaim`) and mounts it at the canonical ComfyUI path.

| Key | Mount path | Default size |
|---|---|---|
| `persistence.models` | `/app/ComfyUI/models` | `200Gi` |
| `persistence.output` | `/app/ComfyUI/output` | `50Gi` |
| `persistence.input`  | `/app/ComfyUI/input`  | `10Gi` |
| `persistence.user`   | `/app/ComfyUI/user`   | `10Gi` |

Default access mode is `ReadWriteMany` because the chart is intended to scale beyond a single replica. Use a RWX-capable backend such as AWS EFS / Mountpoint-S3 CSI, GCP Filestore, Azure Files, CephFS, or Longhorn RWX. If you only run a single replica on block storage, override `accessModes` with `[ReadWriteOnce]`.

Example (AWS EKS with the EFS CSI driver):

```yaml
persistence:
  models:
    enabled: true
    storageClass: efs-sc
    size: 500Gi
  output:
    enabled: true
    storageClass: efs-sc
    size: 100Gi
```

To attach a pre-provisioned claim instead of creating one, set `existingClaim`:

```yaml
persistence:
  models:
    enabled: true
    existingClaim: my-shared-models-pvc
```

For bringing in additional model storage paths (for example, a separate volume per model family), use ComfyUI's `extra_model_paths.yaml` mechanism by mounting the file with the standard `volumes` / `volumeMounts` keys.

## Security & Pod Security Standards

The chart is configured to satisfy the Pod Security Standards **baseline** profile out of the box:

- The container image runs as the unprivileged `comfyui` user (UID/GID 10001).
- Pod-level `securityContext` enforces `runAsNonRoot: true`, `seccompProfile: RuntimeDefault`, and a `fsGroup` so PVC mounts are writable by the user.
- Container-level `securityContext` drops all capabilities, disables privilege escalation, and is ready to be flipped to `readOnlyRootFilesystem: true` once writable paths are externalized via `persistence.*`.
- The ServiceAccount token is **not** projected into the pod (`automountServiceAccountToken: false`) because ComfyUI does not call the Kubernetes API.

Do not set `NVIDIA_VISIBLE_DEVICES` in `values.yaml`. Kubernetes does not expand shell variables in env values, and overriding it defeats the per-pod GPU isolation provided by the NVIDIA device plugin.

To run under the **restricted** profile, enable persistence so writable directories live on PVCs, then set `securityContext.readOnlyRootFilesystem: true`.

## GPU Node Scheduling

`nodeSelector` defaults to empty so the pod lands on whichever node has a free `nvidia.com/gpu`. If your cluster does not advertise that resource cluster-wide, pin the pod to GPU nodes using either a manual label or the labels emitted by the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/) plus [Node Feature Discovery](https://kubernetes-sigs.github.io/node-feature-discovery/):

```yaml
# Manual label
nodeSelector:
  nvidia.com/gpu: "true"
```

```yaml
# NFD / GPU Operator
nodeSelector:
  feature.node.kubernetes.io/pci-10de.present: "true"
```

For GPU-flavored mixed clusters, also consider:

- `priorityClassName` to preempt batch / training jobs when interactive ComfyUI requests come in.
- `topologySpreadConstraints` to spread replicas across zones and hosts when scaling out.
- `podDisruptionBudget` (opt-in) so node drains preserve at least one ComfyUI pod.
- `networkPolicy` (opt-in) to restrict ingress to your ingress controller's namespace and limit egress to model registries.

Beyond the device-plugin model, Kubernetes 1.35+ supports [Dynamic Resource Allocation (DRA)](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) with the NVIDIA DRA driver. Migrating to DRA is out of scope for this chart but is the recommended direction for clusters that need MIG, time-slicing, or richer GPU selectors.

## Testing the Chart

Run provided tests:

```
helm test comfyui-helm
```

## ERNIE-Image Support

This container image tracks the latest ComfyUI core. As of the April 2026 ERNIE-Image launch, current upstream ComfyUI includes native support for Baidu ERNIE-Image and the immediate follow-up fixes.

After rebuilding the image, ERNIE-Image models can be mounted into the normal ComfyUI model directories under `/app/ComfyUI/models`.

References:

- [ComfyUI Blog: ERNIE-Image Day-0 Support](https://blog.comfy.org/p/ernie-image-day-0-support)
- [ComfyUI releases](https://github.com/Comfy-Org/ComfyUI/releases)

## Contributing

This project is open source—contributions are encouraged! Fork the repository, submit issues or feature requests, and create pull requests to improve this Helm chart.

See the [LICENSE](LICENSE) file for licensing details.

## About ComfyUI

For more information on ComfyUI, visit the official [ComfyUI GitHub repository](https://github.com/Comfy-Org/ComfyUI).
