#!/usr/bin/env python3
"""
Gestor de Modelos de IA Open Source
Soporta más de 2000 modelos de Hugging Face Hub
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import time
import psutil
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import aiofiles

# Hugging Face imports
from huggingface_hub import HfApi, ModelFilter, snapshot_download, login
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoTokenizer,
    AutoProcessor,
    pipeline,
    BitsAndBytesConfig
)

# Diffusers for image generation
from diffusers import StableDiffusionPipeline, DiffusionPipeline

# Audio processing
import librosa
import soundfile as sf

# Computer vision
import cv2
from PIL import Image

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Información de un modelo"""
    model_id: str
    task: str
    downloads: int
    tags: List[str]
    size_gb: float
    gpu_memory_gb: float
    status: str = "available"
    loaded_at: Optional[float] = None

@dataclass
class InferenceResult:
    """Resultado de inferencia"""
    model_id: str
    task: str
    input_data: Any
    output: Any
    inference_time: float
    gpu_memory_used: float
    timestamp: float

class ModelCache:
    """Cache inteligente para modelos"""

    def __init__(self, max_memory_gb: float = 32.0):
        self.max_memory_gb = max_memory_gb
        self.loaded_models: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}

    def can_load_model(self, model_info: ModelInfo) -> bool:
        """Verifica si se puede cargar un modelo"""
        current_memory = sum(
            model["memory_gb"] for model in self.loaded_models.values()
        )
        return current_memory + model_info.gpu_memory_gb <= self.max_memory_gb

    def evict_lru_model(self) -> Optional[str]:
        """Expulsa el modelo menos recientemente usado"""
        if not self.access_times:
            return None

        lru_model = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.unload_model(lru_model)
        return lru_model

    def load_model(self, model_id: str, model_data: Dict, memory_gb: float):
        """Carga un modelo en cache"""
        self.loaded_models[model_id] = {
            **model_data,
            "memory_gb": memory_gb
        }
        self.access_times[model_id] = time.time()

    def unload_model(self, model_id: str):
        """Descarga un modelo del cache"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            del self.access_times[model_id]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

class OpenSourceModelManager:
    """Gestor principal para más de 2000 modelos de código abierto"""

    def __init__(self,
                 cache_dir: str = "./models",
                 max_memory_gb: float = 32.0,
                 hf_token: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.hf_api = HfApi()
        self.hf_token = hf_token
        if hf_token:
            login(token=hf_token)

        self.model_cache = ModelCache(max_memory_gb)
        self.available_models: Dict[str, ModelInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Configuración de cuantización para modelos grandes
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Inicializar catálogo de modelos
        asyncio.create_task(self._initialize_model_catalog())

    async def _initialize_model_catalog(self):
        """Inicializa el catálogo de modelos disponibles"""
        logger.info("Inicializando catálogo de modelos...")

        # Categorías de modelos a buscar
        tasks = [
            "text-generation",
            "text-classification",
            "image-classification",
            "image-to-text",
            "text-to-image",
            "automatic-speech-recognition",
            "translation",
            "summarization",
            "question-answering",
            "fill-mask",
            "token-classification",
            "zero-shot-classification"
        ]

        all_models = []

        for task in tasks:
            try:
                models = self.hf_api.list_models(
                    filter=ModelFilter(
                        task=task,
                        library="transformers",
                    ),
                    sort="downloads",
                    direction=-1,
                    limit=200  # Top 200 por categoría
                )

                for model in models:
                    if model.downloads and model.downloads > 100:  # Filtro mínimo
                        model_info = ModelInfo(
                            model_id=model.modelId,
                            task=task,
                            downloads=model.downloads or 0,
                            tags=model.tags or [],
                            size_gb=self._estimate_model_size(model.modelId),
                            gpu_memory_gb=self._estimate_gpu_memory(model.modelId)
                        )
                        all_models.append(model_info)

            except Exception as e:
                logger.error(f"Error obteniendo modelos para tarea {task}: {e}")

        # Agregar modelos de diffusers para generación de imágenes
        diffusion_models = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "dreamlike-art/dreamlike-diffusion-1.0",
            "wavymulder/Analog-Diffusion"
        ]

        for model_id in diffusion_models:
            model_info = ModelInfo(
                model_id=model_id,
                task="text-to-image",
                downloads=50000,  # Estimado
                tags=["diffusion", "image-generation"],
                size_gb=4.0,
                gpu_memory_gb=8.0
            )
            all_models.append(model_info)

        # Convertir a diccionario
        self.available_models = {
            model.model_id: model for model in all_models
        }

        logger.info(f"Catálogo inicializado con {len(self.available_models)} modelos")

    def _estimate_model_size(self, model_id: str) -> float:
        """Estima el tamaño de un modelo en GB"""
        # Estimaciones basadas en patrones comunes
        if "7b" in model_id.lower():
            return 14.0
        elif "13b" in model_id.lower():
            return 26.0
        elif "30b" in model_id.lower():
            return 60.0
        elif "70b" in model_id.lower():
            return 140.0
        elif "base" in model_id.lower():
            return 0.5
        elif "large" in model_id.lower():
            return 1.5
        else:
            return 2.0  # Valor por defecto

    def _estimate_gpu_memory(self, model_id: str) -> float:
        """Estima la memoria GPU requerida en GB"""
        size_gb = self._estimate_model_size(model_id)
        # Aproximadamente 2x el tamaño del modelo para inferencia
        return min(size_gb * 2, 80.0)  # Máximo 80GB

    async def download_model(self, model_id: str, revision: str = "main") -> Optional[str]:
        """Descarga un modelo de Hugging Face Hub"""
        try:
            logger.info(f"Descargando modelo {model_id}...")

            # Ejecutar descarga en thread separado
            loop = asyncio.get_event_loop()
            model_path = await loop.run_in_executor(
                self.executor,
                lambda: snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=str(self.cache_dir),
                    token=self.hf_token
                )
            )

            logger.info(f"Modelo {model_id} descargado en: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Error descargando modelo {model_id}: {e}")
            return None

    async def load_model(self, model_id: str, device: str = None) -> bool:
        """Carga un modelo en memoria"""
        if model_id in self.model_cache.loaded_models:
            self.model_cache.access_times[model_id] = time.time()
            return True

        if model_id not in self.available_models:
            logger.error(f"Modelo {model_id} no encontrado en catálogo")
            return False

        model_info = self.available_models[model_id]

        # Verificar memoria disponible
        while not self.model_cache.can_load_model(model_info):
            evicted = self.model_cache.evict_lru_model()
            if not evicted:
                logger.error(f"No se puede cargar {model_id}: memoria insuficiente")
                return False

        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Descargar modelo si no existe
            model_path = await self.download_model(model_id)
            if not model_path:
                return False

            # Cargar según el tipo de tarea
            model_data = await self._load_model_by_task(model_id, model_info.task, device)

            if model_data:
                self.model_cache.load_model(
                    model_id,
                    model_data,
                    model_info.gpu_memory_gb
                )
                model_info.status = "loaded"
                model_info.loaded_at = time.time()
                logger.info(f"Modelo {model_id} cargado exitosamente")
                return True

        except Exception as e:
            logger.error(f"Error cargando modelo {model_id}: {e}")

        return False

    async def _load_model_by_task(self, model_id: str, task: str, device: str) -> Optional[Dict]:
        """Carga un modelo según su tarea específica"""
        try:
            if task == "text-generation":
                return await self._load_text_generation_model(model_id, device)
            elif task == "text-classification":
                return await self._load_classification_model(model_id, device)
            elif task == "text-to-image":
                return await self._load_diffusion_model(model_id, device)
            elif task == "image-classification":
                return await self._load_image_classification_model(model_id, device)
            elif task == "automatic-speech-recognition":
                return await self._load_speech_model(model_id, device)
            else:
                # Modelo genérico con pipeline
                return await self._load_generic_model(model_id, task, device)

        except Exception as e:
            logger.error(f"Error cargando modelo {model_id} para tarea {task}: {e}")
            return None

    async def _load_text_generation_model(self, model_id: str, device: str) -> Dict:
        """Carga modelo de generación de texto"""
        loop = asyncio.get_event_loop()

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token
            )

            # Configurar padding token si no existe
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {
                "cache_dir": str(self.cache_dir),
                "token": self.hf_token,
                "device_map": "auto" if device == "cuda" else None,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32
            }

            # Usar cuantización para modelos grandes
            if self._estimate_model_size(model_id) > 10.0 and device == "cuda":
                model_kwargs["quantization_config"] = self.quantization_config

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "type": "text-generation"
            }

        return await loop.run_in_executor(self.executor, _load)

    async def _load_classification_model(self, model_id: str, device: str) -> Dict:
        """Carga modelo de clasificación"""
        loop = asyncio.get_event_loop()

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token
            )

            if device == "cuda":
                model = model.cuda()

            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "type": "classification"
            }

        return await loop.run_in_executor(self.executor, _load)

    async def _load_diffusion_model(self, model_id: str, device: str) -> Dict:
        """Carga modelo de difusión para generación de imágenes"""
        loop = asyncio.get_event_loop()

        def _load():
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )

            if device == "cuda":
                pipe = pipe.to("cuda")
                pipe.enable_memory_efficient_attention()

            return {
                "pipeline": pipe,
                "device": device,
                "type": "diffusion"
            }

        return await loop.run_in_executor(self.executor, _load)

    async def _load_image_classification_model(self, model_id: str, device: str) -> Dict:
        """Carga modelo de clasificación de imágenes"""
        loop = asyncio.get_event_loop()

        def _load():
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token
            )

            model = AutoModelForImageClassification.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token
            )

            if device == "cuda":
                model = model.cuda()

            return {
                "model": model,
                "processor": processor,
                "device": device,
                "type": "image-classification"
            }

        return await loop.run_in_executor(self.executor, _load)

    async def _load_speech_model(self, model_id: str, device: str) -> Dict:
        """Carga modelo de reconocimiento de voz"""
        loop = asyncio.get_event_loop()

        def _load():
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token,
                device=0 if device == "cuda" else -1
            )

            return {
                "pipeline": pipe,
                "device": device,
                "type": "speech-recognition"
            }

        return await loop.run_in_executor(self.executor, _load)

    async def _load_generic_model(self, model_id: str, task: str, device: str) -> Dict:
        """Carga modelo genérico usando pipeline"""
        loop = asyncio.get_event_loop()

        def _load():
            pipe = pipeline(
                task=task,
                model=model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token,
                device=0 if device == "cuda" else -1
            )

            return {
                "pipeline": pipe,
                "device": device,
                "type": "generic"
            }

        return await loop.run_in_executor(self.executor, _load)

    async def run_inference(self,
                          model_id: str,
                          input_data: Any,
                          **kwargs) -> Optional[InferenceResult]:
        """Ejecuta inferencia en un modelo"""
        if model_id not in self.model_cache.loaded_models:
            success = await self.load_model(model_id)
            if not success:
                return None

        model_data = self.model_cache.loaded_models[model_id]
        model_info = self.available_models[model_id]

        start_time = time.time()
        gpu_memory_before = self._get_gpu_memory_usage()

        try:
            # Ejecutar inferencia según el tipo de modelo
            if model_data["type"] == "text-generation":
                output = await self._run_text_generation(model_data, input_data, **kwargs)
            elif model_data["type"] == "classification":
                output = await self._run_classification(model_data, input_data, **kwargs)
            elif model_data["type"] == "diffusion":
                output = await self._run_diffusion(model_data, input_data, **kwargs)
            elif model_data["type"] == "image-classification":
                output = await self._run_image_classification(model_data, input_data, **kwargs)
            elif model_data["type"] in ["speech-recognition", "generic"]:
                output = await self._run_pipeline(model_data, input_data, **kwargs)
            else:
                raise ValueError(f"Tipo de modelo no soportado: {model_data['type']}")

            inference_time = time.time() - start_time
            gpu_memory_after = self._get_gpu_memory_usage()

            # Actualizar tiempo de acceso
            self.model_cache.access_times[model_id] = time.time()

            return InferenceResult(
                model_id=model_id,
                task=model_info.task,
                input_data=input_data,
                output=output,
                inference_time=inference_time,
                gpu_memory_used=gpu_memory_after - gpu_memory_before,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Error en inferencia para {model_id}: {e}")
            return None

    async def _run_text_generation(self, model_data: Dict, prompt: str, **kwargs) -> str:
        """Ejecuta generación de texto"""
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        device = model_data["device"]

        # Parámetros por defecto
        max_length = kwargs.get("max_length", 512)
        temperature = kwargs.get("temperature", 0.7)
        do_sample = kwargs.get("do_sample", True)
        top_p = kwargs.get("top_p", 0.9)

        loop = asyncio.get_event_loop()

        def _generate():
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    **kwargs
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remover el prompt original
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text

        return await loop.run_in_executor(self.executor, _generate)

    async def _run_classification(self, model_data: Dict, text: str, **kwargs) -> Dict:
        """Ejecuta clasificación de texto"""
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        device = model_data["device"]

        loop = asyncio.get_event_loop()

        def _classify():
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Obtener etiquetas si están disponibles
            labels = getattr(model.config, 'id2label', {})

            results = []
            for i, score in enumerate(predictions[0]):
                label = labels.get(i, f"LABEL_{i}")
                results.append({
                    "label": label,
                    "score": float(score)
                })

            # Ordenar por score descendente
            results.sort(key=lambda x: x["score"], reverse=True)

            return results

        return await loop.run_in_executor(self.executor, _classify)

    async def _run_diffusion(self, model_data: Dict, prompt: str, **kwargs) -> Image.Image:
        """Ejecuta generación de imágenes con difusión"""
        pipeline = model_data["pipeline"]

        # Parámetros por defecto
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)

        loop = asyncio.get_event_loop()

        def _generate():
            with torch.no_grad():
                image = pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    **kwargs
                ).images[0]

            return image

        return await loop.run_in_executor(self.executor, _generate)

    async def _run_image_classification(self, model_data: Dict, image: Image.Image, **kwargs) -> Dict:
        """Ejecuta clasificación de imágenes"""
        model = model_data["model"]
        processor = model_data["processor"]
        device = model_data["device"]

        loop = asyncio.get_event_loop()

        def _classify():
            inputs = processor(images=image, return_tensors="pt")

            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Obtener etiquetas
            labels = getattr(model.config, 'id2label', {})

            results = []
            for i, score in enumerate(predictions[0]):
                label = labels.get(i, f"LABEL_{i}")
                results.append({
                    "label": label,
                    "score": float(score)
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        return await loop.run_in_executor(self.executor, _classify)

    async def _run_pipeline(self, model_data: Dict, input_data: Any, **kwargs) -> Any:
        """Ejecuta pipeline genérico"""
        pipeline = model_data["pipeline"]

        loop = asyncio.get_event_loop()

        def _run():
            return pipeline(input_data, **kwargs)

        return await loop.run_in_executor(self.executor, _run)

    def _get_gpu_memory_usage(self) -> float:
        """Obtiene el uso actual de memoria GPU en GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Obtiene información de un modelo"""
        return self.available_models.get(model_id)

    def list_available_models(self,
                            task: Optional[str] = None,
                            min_downloads: int = 0,
                            max_size_gb: Optional[float] = None) -> List[ModelInfo]:
        """Lista modelos disponibles con filtros"""
        models = list(self.available_models.values())

        if task:
            models = [m for m in models if m.task == task]

        if min_downloads > 0:
            models = [m for m in models if m.downloads >= min_downloads]

        if max_size_gb:
            models = [m for m in models if m.size_gb <= max_size_gb]

        return sorted(models, key=lambda x: x.downloads, reverse=True)

    def list_loaded_models(self) -> List[str]:
        """Lista modelos cargados en memoria"""
        return list(self.model_cache.loaded_models.keys())

    def get_system_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        return {
            "loaded_models": len(self.model_cache.loaded_models),
            "available_models": len(self.available_models),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory_gb": self._get_gpu_memory_usage(),
            "cache_memory_gb": sum(
                model["memory_gb"] for model in self.model_cache.loaded_models.values()
            )
        }

    async def unload_model(self, model_id: str) -> bool:
        """Descarga un modelo de memoria"""
        if model_id in self.model_cache.loaded_models:
            self.model_cache.unload_model(model_id)
            if model_id in self.available_models:
                self.available_models[model_id].status = "available"
                self.available_models[model_id].loaded_at = None
            logger.info(f"Modelo {model_id} descargado de memoria")
            return True
        return False

    async def cleanup(self):
        """Limpia recursos"""
        for model_id in list(self.model_cache.loaded_models.keys()):
            await self.unload_model(model_id)

        self.executor.shutdown(wait=True)
        logger.info("Recursos limpiados")

# Instancia global del gestor
model_manager = None

def get_model_manager() -> OpenSourceModelManager:
    """Obtiene la instancia global del gestor de modelos"""
    global model_manager
    if model_manager is None:
        model_manager = OpenSourceModelManager()
    return model_manager
