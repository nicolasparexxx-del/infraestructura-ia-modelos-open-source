#!/usr/bin/env python3
"""
API Principal para Infraestructura de IA con 2000+ Modelos
FastAPI server con endpoints para gestión e inferencia de modelos
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
import uuid
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, Field
from PIL import Image
import io
import base64

# Importar nuestro gestor de modelos
from model_manager import get_model_manager, ModelInfo, InferenceResult

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Esquemas Pydantic
class ModelRequest(BaseModel):
    model_id: str = Field(..., description="ID del modelo a usar")
    input_data: Union[str, Dict, List] = Field(..., description="Datos de entrada")
    parameters: Optional[Dict] = Field(default={}, description="Parámetros adicionales")

class TextGenerationRequest(BaseModel):
    model_id: str = Field(..., description="ID del modelo de generación de texto")
    prompt: str = Field(..., description="Texto de entrada")
    max_length: int = Field(default=512, ge=1, le=4096, description="Longitud máxima")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Temperatura")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling")
    do_sample: bool = Field(default=True, description="Usar sampling")

class ImageGenerationRequest(BaseModel):
    model_id: str = Field(..., description="ID del modelo de generación de imágenes")
    prompt: str = Field(..., description="Descripción de la imagen")
    negative_prompt: Optional[str] = Field(default="", description="Prompt negativo")
    width: int = Field(default=512, ge=256, le=1024, description="Ancho de la imagen")
    height: int = Field(default=512, ge=256, le=1024, description="Alto de la imagen")
    num_inference_steps: int = Field(default=50, ge=10, le=100, description="Pasos de inferencia")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Escala de guía")

class ClassificationRequest(BaseModel):
    model_id: str = Field(..., description="ID del modelo de clasificación")
    text: str = Field(..., description="Texto a clasificar")

class ModelResponse(BaseModel):
    request_id: str
    model_id: str
    task: str
    output: Any
    inference_time: float
    timestamp: float
    gpu_memory_used: float

class ModelInfoResponse(BaseModel):
    model_id: str
    task: str
    downloads: int
    tags: List[str]
    size_gb: float
    gpu_memory_gb: float
    status: str
    loaded_at: Optional[float]

class SystemStatsResponse(BaseModel):
    loaded_models: int
    available_models: int
    cpu_percent: float
    memory_percent: float
    gpu_memory_gb: float
    cache_memory_gb: float
    timestamp: float

class BatchRequest(BaseModel):
    requests: List[ModelRequest] = Field(..., description="Lista de solicitudes")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Máximo concurrente")

# Gestor de autenticación simple
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Autenticación simple con token"""
    if not credentials:
        return None

    # Aquí puedes implementar tu lógica de autenticación
    # Por ahora, aceptamos cualquier token
    return {"user_id": "anonymous", "token": credentials.credentials}

# Contexto de aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    logger.info("🚀 Iniciando API de Modelos de IA...")

    # Inicializar gestor de modelos
    manager = get_model_manager()

    # Esperar a que se inicialice el catálogo
    await asyncio.sleep(2)

    logger.info(f"✅ API iniciada con {len(manager.available_models)} modelos disponibles")

    yield

    # Cleanup
    logger.info("🔄 Limpiando recursos...")
    await manager.cleanup()
    logger.info("✅ Recursos limpiados")

# Crear aplicación FastAPI
app = FastAPI(
    title="AI Models Infrastructure API",
    description="API para gestión e inferencia de más de 2000 modelos de IA open-source",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Cache para respuestas
response_cache: Dict[str, Any] = {}

# Rutas principales
@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "AI Models Infrastructure API",
        "version": "1.0.0",
        "description": "API para gestión e inferencia de más de 2000 modelos de IA open-source",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Verificación de salud del servicio"""
    manager = get_model_manager()
    stats = manager.get_system_stats()

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system_stats": stats
    }

# Endpoints de gestión de modelos
@app.get("/models/available", response_model=List[ModelInfoResponse])
async def get_available_models(
    task: Optional[str] = None,
    min_downloads: int = 0,
    max_size_gb: Optional[float] = None,
    limit: int = 100,
    offset: int = 0
):
    """Obtiene lista de modelos disponibles con filtros"""
    try:
        manager = get_model_manager()
        models = manager.list_available_models(task, min_downloads, max_size_gb)

        # Paginación
        total = len(models)
        models = models[offset:offset + limit]

        response_models = [
            ModelInfoResponse(
                model_id=model.model_id,
                task=model.task,
                downloads=model.downloads,
                tags=model.tags,
                size_gb=model.size_gb,
                gpu_memory_gb=model.gpu_memory_gb,
                status=model.status,
                loaded_at=model.loaded_at
            )
            for model in models
        ]

        return response_models

    except Exception as e:
        logger.error(f"Error obteniendo modelos disponibles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/loaded")
async def get_loaded_models():
    """Obtiene lista de modelos cargados en memoria"""
    try:
        manager = get_model_manager()
        loaded_models = manager.list_loaded_models()

        return {
            "loaded_models": loaded_models,
            "count": len(loaded_models),
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error obteniendo modelos cargados: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load/{model_id}")
async def load_model(model_id: str, background_tasks: BackgroundTasks):
    """Carga un modelo específico en memoria"""
    try:
        manager = get_model_manager()

        # Verificar que el modelo existe
        if model_id not in manager.available_models:
            raise HTTPException(status_code=404, detail=f"Modelo {model_id} no encontrado")

        # Cargar modelo en background
        background_tasks.add_task(manager.load_model, model_id)

        return {
            "status": "loading",
            "message": f"Cargando modelo {model_id}...",
            "model_id": model_id,
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cargando modelo {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/unload/{model_id}")
async def unload_model(model_id: str):
    """Descarga un modelo de memoria"""
    try:
        manager = get_model_manager()
        success = await manager.unload_model(model_id)

        if success:
            return {
                "status": "success",
                "message": f"Modelo {model_id} descargado de memoria",
                "model_id": model_id,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Modelo {model_id} no estaba cargado")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando modelo {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info/{model_id}", response_model=ModelInfoResponse)
async def get_model_info(model_id: str):
    """Obtiene información detallada de un modelo"""
    try:
        manager = get_model_manager()
        model_info = manager.get_model_info(model_id)

        if not model_info:
            raise HTTPException(status_code=404, detail=f"Modelo {model_id} no encontrado")

        return ModelInfoResponse(
            model_id=model_info.model_id,
            task=model_info.task,
            downloads=model_info.downloads,
            tags=model_info.tags,
            size_gb=model_info.size_gb,
            gpu_memory_gb=model_info.gpu_memory_gb,
            status=model_info.status,
            loaded_at=model_info.loaded_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de inferencia
@app.post("/inference/text-generation", response_model=ModelResponse)
async def generate_text(request: TextGenerationRequest):
    """Genera texto usando un modelo de lenguaje"""
    try:
        manager = get_model_manager()

        # Verificar que el modelo existe y es del tipo correcto
        model_info = manager.get_model_info(request.model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Modelo {request.model_id} no encontrado")

        if model_info.task != "text-generation":
            raise HTTPException(
                status_code=400,
                detail=f"Modelo {request.model_id} no es de generación de texto"
            )

        # Ejecutar inferencia
        result = await manager.run_inference(
            model_id=request.model_id,
            input_data=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )

        if not result:
            raise HTTPException(status_code=500, detail="Error en la inferencia")

        return ModelResponse(
            request_id=str(uuid.uuid4()),
            model_id=result.model_id,
            task=result.task,
            output=result.output,
            inference_time=result.inference_time,
            timestamp=result.timestamp,
            gpu_memory_used=result.gpu_memory_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en generación de texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/image-generation")
async def generate_image(request: ImageGenerationRequest):
    """Genera imágenes usando modelos de difusión"""
    try:
        manager = get_model_manager()

        # Verificar modelo
        model_info = manager.get_model_info(request.model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Modelo {request.model_id} no encontrado")

        if model_info.task != "text-to-image":
            raise HTTPException(
                status_code=400,
                detail=f"Modelo {request.model_id} no es de generación de imágenes"
            )

        # Ejecutar inferencia
        result = await manager.run_inference(
            model_id=request.model_id,
            input_data=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale
        )

        if not result:
            raise HTTPException(status_code=500, detail="Error en la inferencia")

        # Convertir imagen a base64
        image = result.output
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "request_id": str(uuid.uuid4()),
            "model_id": result.model_id,
            "task": result.task,
            "output": {
                "image_base64": image_base64,
                "format": "PNG",
                "width": image.width,
                "height": image.height
            },
            "inference_time": result.inference_time,
            "timestamp": result.timestamp,
            "gpu_memory_used": result.gpu_memory_used
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en generación de imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/classification", response_model=ModelResponse)
async def classify_text(request: ClassificationRequest):
    """Clasifica texto usando modelos de clasificación"""
    try:
        manager = get_model_manager()

        # Verificar modelo
        model_info = manager.get_model_info(request.model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Modelo {request.model_id} no encontrado")

        if model_info.task != "text-classification":
            raise HTTPException(
                status_code=400,
                detail=f"Modelo {request.model_id} no es de clasificación de texto"
            )

        # Ejecutar inferencia
        result = await manager.run_inference(
            model_id=request.model_id,
            input_data=request.text
        )

        if not result:
            raise HTTPException(status_code=500, detail="Error en la inferencia")

        return ModelResponse(
            request_id=str(uuid.uuid4()),
            model_id=result.model_id,
            task=result.task,
            output=result.output,
            inference_time=result.inference_time,
            timestamp=result.timestamp,
            gpu_memory_used=result.gpu_memory_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clasificación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/image-classification")
async def classify_image(
    model_id: str = Form(...),
    image: UploadFile = File(...)
):
    """Clasifica imágenes usando modelos de visión"""
    try:
        manager = get_model_manager()

        # Verificar modelo
        model_info = manager.get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Modelo {model_id} no encontrado")

        if model_info.task != "image-classification":
            raise HTTPException(
                status_code=400,
                detail=f"Modelo {model_id} no es de clasificación de imágenes"
            )

        # Procesar imagen
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Ejecutar inferencia
        result = await manager.run_inference(
            model_id=model_id,
            input_data=pil_image
        )

        if not result:
            raise HTTPException(status_code=500, detail="Error en la inferencia")

        return {
            "request_id": str(uuid.uuid4()),
            "model_id": result.model_id,
            "task": result.task,
            "output": result.output,
            "inference_time": result.inference_time,
            "timestamp": result.timestamp,
            "gpu_memory_used": result.gpu_memory_used
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clasificación de imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/batch")
async def batch_inference(request: BatchRequest):
    """Ejecuta inferencia en lote para múltiples solicitudes"""
    try:
        manager = get_model_manager()

        # Crear semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def process_request(req: ModelRequest):
            async with semaphore:
                return await manager.run_inference(
                    model_id=req.model_id,
                    input_data=req.input_data,
                    **req.parameters
                )

        # Ejecutar todas las solicitudes
        tasks = [process_request(req) for req in request.requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Procesar resultados
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses.append({
                    "request_index": i,
                    "error": str(result),
                    "status": "error"
                })
            elif result:
                responses.append({
                    "request_index": i,
                    "request_id": str(uuid.uuid4()),
                    "model_id": result.model_id,
                    "task": result.task,
                    "output": result.output,
                    "inference_time": result.inference_time,
                    "timestamp": result.timestamp,
                    "gpu_memory_used": result.gpu_memory_used,
                    "status": "success"
                })
            else:
                responses.append({
                    "request_index": i,
                    "error": "Inference failed",
                    "status": "error"
                })

        return {
            "batch_id": str(uuid.uuid4()),
            "total_requests": len(request.requests),
            "successful": len([r for r in responses if r.get("status") == "success"]),
            "failed": len([r for r in responses if r.get("status") == "error"]),
            "results": responses,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error en inferencia en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de monitorización
@app.get("/stats/system", response_model=SystemStatsResponse)
async def get_system_stats():
    """Obtiene estadísticas del sistema"""
    try:
        manager = get_model_manager()
        stats = manager.get_system_stats()

        return SystemStatsResponse(
            loaded_models=stats["loaded_models"],
            available_models=stats["available_models"],
            cpu_percent=stats["cpu_percent"],
            memory_percent=stats["memory_percent"],
            gpu_memory_gb=stats["gpu_memory_gb"],
            cache_memory_gb=stats["cache_memory_gb"],
            timestamp=time.time()
        )

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/models")
async def get_model_stats():
    """Obtiene estadísticas de uso de modelos"""
    try:
        manager = get_model_manager()

        # Estadísticas por tarea
        task_stats = {}
        for model in manager.available_models.values():
            task = model.task
            if task not in task_stats:
                task_stats[task] = {"count": 0, "total_downloads": 0}
            task_stats[task]["count"] += 1
            task_stats[task]["total_downloads"] += model.downloads

        # Modelos más populares
        popular_models = sorted(
            manager.available_models.values(),
            key=lambda x: x.downloads,
            reverse=True
        )[:10]

        return {
            "total_models": len(manager.available_models),
            "loaded_models": len(manager.model_cache.loaded_models),
            "task_distribution": task_stats,
            "popular_models": [
                {
                    "model_id": model.model_id,
                    "task": model.task,
                    "downloads": model.downloads,
                    "status": model.status
                }
                for model in popular_models
            ],
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de streaming para generación de texto
@app.post("/inference/text-generation/stream")
async def stream_text_generation(request: TextGenerationRequest):
    """Genera texto con streaming en tiempo real"""
    try:
        # Nota: Esta es una implementación simplificada
        # En una implementación real, necesitarías modificar el model_manager
        # para soportar streaming

        async def generate_stream():
            manager = get_model_manager()

            # Por ahora, simulamos streaming dividiendo la respuesta
            result = await manager.run_inference(
                model_id=request.model_id,
                input_data=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample
            )

            if result:
                # Simular streaming dividiendo el texto
                text = result.output
                words = text.split()

                for i, word in enumerate(words):
                    chunk = {
                        "chunk": word + " ",
                        "index": i,
                        "finished": i == len(words) - 1
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.1)  # Simular delay

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )

    except Exception as e:
        logger.error(f"Error en streaming de texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )
