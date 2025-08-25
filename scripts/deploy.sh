#!/bin/bash

# Script de Despliegue Automático para Infraestructura de IA con 2000+ Modelos
# Autor: AI Infrastructure Team
# Versión: 1.0.0

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables de configuración
PROJECT_NAME="ai-models-infrastructure"
NAMESPACE="ai-models"
DOCKER_IMAGE="ai-models-api"
DOCKER_TAG="latest"
KUBERNETES_CONTEXT="ai-models-cluster"
MODEL_STORAGE_SIZE="2Ti"
GPU_NODES=3
MIN_REPLICAS=3
MAX_REPLICAS=10

# Funciones de utilidad
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Verificando requisitos del sistema..."

    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker no está instalado"
        exit 1
    fi

    # Verificar kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl no está instalado"
        exit 1
    fi

    # Verificar Helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm no está instalado"
        exit 1
    fi

    # Verificar conexión a cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "No se puede conectar al cluster de Kubernetes"
        exit 1
    fi

    log_success "Todos los requisitos están satisfechos"
}

setup_environment() {
    log_info "Configurando variables de entorno..."

    # Crear archivo .env si no existe
    if [ ! -f .env ]; then
        cat > .env << EOF
# Configuración de la API
API_TOKEN=$(openssl rand -hex 32)
HF_TOKEN=your-huggingface-token-here

# Base de datos
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=$(openssl rand -hex 16)
POSTGRES_DB=ai_models_db

# Redis
REDIS_PASSWORD=$(openssl rand -hex 16)

# Grafana
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 12)

# Configuración del cluster
CLUSTER_NAME=ai-models-cluster
REGION=us-central1
ZONE=us-central1-a

# Configuración de almacenamiento
MODEL_STORAGE_PATH=/mnt/models
BACKUP_STORAGE_PATH=/mnt/backups
EOF
        log_success "Archivo .env creado"
    fi

    # Cargar variables de entorno
    source .env

    log_success "Variables de entorno configuradas"
}

build_docker_images() {
    log_info "Construyendo imágenes Docker..."

    # Construir imagen principal
    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} -f docker/Dockerfile .

    # Etiquetar para registry
    if [ ! -z "$DOCKER_REGISTRY" ]; then
        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG}
        docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG}
        log_success "Imagen subida al registry"
    fi

    log_success "Imágenes Docker construidas"
}

setup_kubernetes_cluster() {
    log_info "Configurando cluster de Kubernetes..."

    # Crear namespace
    kubectl apply -f kubernetes/namespace.yaml

    # Esperar a que el namespace esté listo
    kubectl wait --for=condition=Active namespace/${NAMESPACE} --timeout=60s

    # Aplicar configuraciones
    kubectl apply -f kubernetes/configmap.yaml
    kubectl apply -f kubernetes/secrets.yaml
    kubectl apply -f kubernetes/pvc.yaml

    log_success "Cluster de Kubernetes configurado"
}

install_gpu_support() {
    log_info "Instalando soporte para GPU..."

    # Instalar NVIDIA device plugin
    kubectl create namespace gpu-resources || true

    # Aplicar NVIDIA device plugin
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

    # Verificar que los nodos GPU estén disponibles
    log_info "Esperando a que los nodos GPU estén listos..."
    sleep 30

    GPU_NODES_READY=$(kubectl get nodes -l accelerator=nvidia-gpu --no-headers | wc -l)
    if [ "$GPU_NODES_READY" -lt "$GPU_NODES" ]; then
        log_warning "Solo $GPU_NODES_READY nodos GPU disponibles de $GPU_NODES esperados"
    else
        log_success "Todos los nodos GPU están listos"
    fi
}

deploy_monitoring() {
    log_info "Desplegando sistema de monitorización..."

    # Instalar Prometheus Operator
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace ${NAMESPACE} \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=200Gi \
        --set grafana.adminPassword=${GRAFANA_ADMIN_PASSWORD} \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=50Gi

    log_success "Sistema de monitorización desplegado"
}

deploy_storage() {
    log_info "Configurando almacenamiento..."

    # Instalar NFS provisioner para almacenamiento compartido
    helm repo add nfs-subdir-external-provisioner https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/
    helm repo update

    helm install nfs-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
        --namespace ${NAMESPACE} \
        --set nfs.server=nfs-server.${NAMESPACE}.svc.cluster.local \
        --set nfs.path=/mnt/models \
        --set storageClass.name=model-storage \
        --set storageClass.defaultClass=false

    log_success "Almacenamiento configurado"
}

deploy_applications() {
    log_info "Desplegando aplicaciones..."

    # Aplicar deployments
    kubectl apply -f kubernetes/deployment.yaml

    # Aplicar servicios
    kubectl apply -f kubernetes/service.yaml

    # Esperar a que los deployments estén listos
    log_info "Esperando a que los deployments estén listos..."
    kubectl wait --for=condition=available --timeout=600s deployment/ai-models-api -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n ${NAMESPACE}

    log_success "Aplicaciones desplegadas"
}

setup_autoscaling() {
    log_info "Configurando autoescalado..."

    # Aplicar HPA
    cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-models-api-hpa
  namespace: ${NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-models-api
  minReplicas: ${MIN_REPLICAS}
  maxReplicas: ${MAX_REPLICAS}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 85
EOF

    log_success "Autoescalado configurado"
}

setup_ingress() {
    log_info "Configurando ingreso..."

    # Instalar NGINX Ingress Controller
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update

    helm install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.replicaCount=2 \
        --set controller.nodeSelector."kubernetes\.io/os"=linux \
        --set defaultBackend.nodeSelector."kubernetes\.io/os"=linux

    # Crear Ingress para la API
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-models-ingress
  namespace: ${NAMESPACE}
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
  - hosts:
    - api.ai-models.local
    - grafana.ai-models.local
    secretName: tls-secret
  rules:
  - host: api.ai-models.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-models-api-service
            port:
              number: 80
  - host: grafana.ai-models.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana-service
            port:
              number: 3000
EOF

    log_success "Ingreso configurado"
}

run_health_checks() {
    log_info "Ejecutando verificaciones de salud..."

    # Verificar que todos los pods estén ejecutándose
    PODS_NOT_READY=$(kubectl get pods -n ${NAMESPACE} --field-selector=status.phase!=Running --no-headers | wc -l)
    if [ "$PODS_NOT_READY" -gt 0 ]; then
        log_warning "$PODS_NOT_READY pods no están en estado Running"
        kubectl get pods -n ${NAMESPACE} --field-selector=status.phase!=Running
    fi

    # Verificar conectividad de la API
    API_SERVICE_IP=$(kubectl get service ai-models-api-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ ! -z "$API_SERVICE_IP" ]; then
        if curl -f http://${API_SERVICE_IP}/health > /dev/null 2>&1; then
            log_success "API está respondiendo correctamente"
        else
            log_warning "API no está respondiendo en http://${API_SERVICE_IP}/health"
        fi
    fi

    # Verificar métricas de Prometheus
    PROMETHEUS_SERVICE_IP=$(kubectl get service prometheus-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    if [ ! -z "$PROMETHEUS_SERVICE_IP" ]; then
        if kubectl run --rm -i --tty --restart=Never test-prometheus --image=curlimages/curl -- curl -f http://${PROMETHEUS_SERVICE_IP}:9090/-/healthy > /dev/null 2>&1; then
            log_success "Prometheus está funcionando correctamente"
        else
            log_warning "Prometheus no está respondiendo"
        fi
    fi

    log_success "Verificaciones de salud completadas"
}

setup_mcp_servers() {
    log_info "Configurando servidores MCP..."

    # Construir servidores MCP
    cd mcp-servers/ai-models-server
    npm install
    npm run build
    cd ../..

    cd mcp-servers/kubernetes-server
    npm install
    npm run build
    cd ../..

    # Crear configuración MCP
    mkdir -p ~/.config/mcp
    cat > ~/.config/mcp/settings.json << EOF
{
  "mcpServers": {
    "ai-models": {
      "command": "node",
      "args": ["$(pwd)/mcp-servers/ai-models-server/build/index.js"],
      "env": {
        "AI_API_URL": "http://${API_SERVICE_IP}:8000",
        "AI_API_TOKEN": "${API_TOKEN}"
      }
    },
    "kubernetes": {
      "command": "node",
      "args": ["$(pwd)/mcp-servers/kubernetes-server/build/index.js"],
      "env": {
        "KUBECONFIG": "${HOME}/.kube/config"
      }
    }
  }
}
EOF

    log_success "Servidores MCP configurados"
}

download_popular_models() {
    log_info "Descargando modelos populares..."

    # Crear job para descargar modelos
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: model-downloader
  namespace: ${NAMESPACE}
spec:
  template:
    spec:
      containers:
      - name: model-downloader
        image: ${DOCKER_IMAGE}:${DOCKER_TAG}
        command: ["python", "-c"]
        args:
        - |
          import asyncio
          from api.model_manager import get_model_manager

          async def download_popular_models():
              manager = get_model_manager()

              # Lista de modelos populares para descargar
              popular_models = [
                  "mistralai/Mistral-7B-v0.1",
                  "microsoft/DialoGPT-medium",
                  "cardiffnlp/twitter-roberta-base-sentiment-latest",
                  "openai/whisper-base",
                  "google/vit-base-patch16-224"
              ]

              for model_id in popular_models:
                  try:
                      print(f"Descargando {model_id}...")
                      await manager.download_model(model_id)
                      print(f"✓ {model_id} descargado")
                  except Exception as e:
                      print(f"✗ Error descargando {model_id}: {e}")

          asyncio.run(download_popular_models())
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      restartPolicy: OnFailure
  backoffLimit: 3
EOF

    log_success "Job de descarga de modelos creado"
}

print_deployment_info() {
    log_info "Información del despliegue:"
    echo ""
    echo "🚀 Infraestructura de IA desplegada exitosamente!"
    echo ""
    echo "📊 Servicios disponibles:"
    echo "  • API Principal: http://${API_SERVICE_IP}:8000"
    echo "  • Documentación: http://${API_SERVICE_IP}:8000/docs"
    echo "  • Grafana: http://grafana.ai-models.local"
    echo "  • Prometheus: http://prometheus.ai-models.local"
    echo ""
    echo "🔧 Comandos útiles:"
    echo "  • Ver pods: kubectl get pods -n ${NAMESPACE}"
    echo "  • Ver logs: kubectl logs -f deployment/ai-models-api -n ${NAMESPACE}"
    echo "  • Escalar API: kubectl scale deployment ai-models-api --replicas=5 -n ${NAMESPACE}"
    echo ""
    echo "📈 Métricas:"
    echo "  • Modelos disponibles: 2000+"
    echo "  • Réplicas de API: ${MIN_REPLICAS}-${MAX_REPLICAS}"
    echo "  • Almacenamiento: ${MODEL_STORAGE_SIZE}"
    echo "  • Nodos GPU: ${GPU_NODES}"
    echo ""
    echo "🔐 Credenciales:"
    echo "  • API Token: ${API_TOKEN}"
    echo "  • Grafana Admin: admin / ${GRAFANA_ADMIN_PASSWORD}"
    echo ""
}

# Función principal
main() {
    log_info "🚀 Iniciando despliegue de Infraestructura de IA con 2000+ Modelos..."

    check_requirements
    setup_environment
    build_docker_images
    setup_kubernetes_cluster
    install_gpu_support
    deploy_storage
    deploy_monitoring
    deploy_applications
    setup_autoscaling
    setup_ingress
    setup_mcp_servers
    download_popular_models
    run_health_checks
    print_deployment_info

    log_success "🎉 Despliegue completado exitosamente!"
}

# Manejo de argumentos
case "${1:-}" in
    "build")
        build_docker_images
        ;;
    "deploy")
        deploy_applications
        ;;
    "monitoring")
        deploy_monitoring
        ;;
    "health")
        run_health_checks
        ;;
    "clean")
        log_info "Limpiando recursos..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        log_success "Recursos limpiados"
        ;;
    *)
        main
        ;;
esac
