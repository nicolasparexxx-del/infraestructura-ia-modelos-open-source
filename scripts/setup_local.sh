#!/bin/bash

# Script de Configuración Local para Infraestructura de IA
# Versión simplificada usando Docker Compose

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

    # Verificar Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose no está instalado"
        exit 1
    fi

    # Verificar que Docker esté ejecutándose
    if ! docker info &> /dev/null; then
        log_error "Docker no está ejecutándose"
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
GRAFANA_ADMIN_PASSWORD=admin123

# Configuración local
MODEL_CACHE_PATH=./models
LOG_LEVEL=INFO
EOF
        log_success "Archivo .env creado"
    fi

    # Crear directorios necesarios
    mkdir -p models logs notebooks monitoring/grafana/dashboards

    log_success "Directorios creados"
}

create_requirements() {
    log_info "Creando archivo requirements.txt..."

    cat > requirements.txt << 'EOF'
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# AI/ML libraries
torch==2.1.1
transformers==4.36.0
diffusers==0.24.0
accelerate==0.25.0

# Hugging Face
huggingface-hub==0.19.4
datasets==2.15.0

# Computer Vision
opencv-python==4.8.1.78
Pillow==10.1.0

# Audio processing
librosa==0.10.1
soundfile==1.0.4

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Cache and messaging
redis==5.0.1

# HTTP clients
httpx==0.25.2
requests==2.31.0

# Monitoring
prometheus-client==0.19.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
psutil==5.9.6
numpy==1.24.4
EOF

    log_success "requirements.txt creado"
}

build_images() {
    log_info "Construyendo imágenes Docker..."

    # Construir imagen principal
    docker-compose build ai-models-api

    log_success "Imágenes construidas"
}

start_services() {
    log_info "Iniciando servicios..."

    # Iniciar servicios básicos primero
    docker-compose up -d postgres redis

    # Esperar a que estén listos
    log_info "Esperando a que la base de datos esté lista..."
    sleep 10

    # Iniciar el resto de servicios
    docker-compose up -d

    log_success "Servicios iniciados"
}

wait_for_services() {
    log_info "Esperando a que los servicios estén listos..."

    # Esperar a que la API esté lista
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_success "API está lista"
            break
        fi
        sleep 2
    done

    # Esperar a que Grafana esté listo
    for i in {1..30}; do
        if curl -f http://localhost:3000 > /dev/null 2>&1; then
            log_success "Grafana está listo"
            break
        fi
        sleep 2
    done
}

setup_mcp_servers() {
    log_info "Configurando servidores MCP..."

    # Instalar dependencias de Node.js para servidores MCP
    if command -v npm &> /dev/null; then
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
        "AI_API_URL": "http://localhost:8000",
        "AI_API_TOKEN": "$(grep API_TOKEN .env | cut -d'=' -f2)"
      }
    }
  }
}
EOF

        log_success "Servidores MCP configurados"
    else
        log_warning "Node.js no está instalado, saltando configuración MCP"
    fi
}

print_info() {
    log_info "Información del despliegue:"
    echo ""
    echo "🚀 Infraestructura de IA desplegada exitosamente!"
    echo ""
    echo "📊 Servicios disponibles:"
    echo "  • API Principal: http://localhost:8000"
    echo "  • Documentación: http://localhost:8000/docs"
    echo "  • Grafana: http://localhost:3000"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Jupyter Lab: http://localhost:8888"
    echo ""
    echo "🔧 Comandos útiles:"
    echo "  • Ver logs: docker-compose logs -f ai-models-api"
    echo "  • Reiniciar API: docker-compose restart ai-models-api"
    echo "  • Parar servicios: docker-compose down"
    echo ""
    echo "🔐 Credenciales:"
    echo "  • Grafana: admin / admin123"
    echo "  • Jupyter: Token: ai-models-jupyter"
    echo ""
    echo "📈 Próximos pasos:"
    echo "  1. Visita http://localhost:8000/docs para explorar la API"
    echo "  2. Usa la API para cargar y probar modelos"
    echo "  3. Monitorea el sistema en Grafana"
    echo ""
}

# Función principal
main() {
    log_info "🚀 Iniciando configuración local de Infraestructura de IA..."

    check_requirements
    setup_environment
    create_requirements
    build_images
    start_services
    wait_for_services
    setup_mcp_servers
    print_info

    log_success "🎉 Configuración completada exitosamente!"
}

# Manejo de argumentos
case "${1:-}" in
    "start")
        docker-compose up -d
        log_success "Servicios iniciados"
        ;;
    "stop")
        docker-compose down
        log_success "Servicios detenidos"
        ;;
    "restart")
        docker-compose restart
        log_success "Servicios reiniciados"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "clean")
        docker-compose down -v
        docker system prune -f
        log_success "Sistema limpiado"
        ;;
    *)
        main
        ;;
esac
