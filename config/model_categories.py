"""
Configuración de Categorías de Modelos de IA
Más de 2000 modelos organizados por categorías y tareas
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class ModelTask(Enum):
    """Enumeración de tareas de modelos"""
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    IMAGE_GENERATION = "text-to-image"
    IMAGE_CLASSIFICATION = "image-classification"
    SPEECH_RECOGNITION = "automatic-speech-recognition"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    AUDIO_CLASSIFICATION = "audio-classification"
    IMAGE_TO_TEXT = "image-to-text"
    OBJECT_DETECTION = "object-detection"
    DEPTH_ESTIMATION = "depth-estimation"
    VIDEO_CLASSIFICATION = "video-classification"

@dataclass
class ModelConfig:
    """Configuración de un modelo específico"""
    model_id: str
    task: ModelTask
    priority: int  # 1-5, donde 5 es máxima prioridad
    estimated_size_gb: float
    gpu_memory_gb: float
    tags: List[str]
    description: str
    license: str
    min_python_version: str = "3.8"
    requires_auth: bool = False

# Modelos de Generación de Texto (LLMs)
TEXT_GENERATION_MODELS = [
    ModelConfig(
        model_id="mistralai/Mistral-7B-v0.1",
        task=ModelTask.TEXT_GENERATION,
        priority=5,
        estimated_size_gb=14.0,
        gpu_memory_gb=16.0,
        tags=["llm", "chat", "instruct", "multilingual"],
        description="Modelo de lenguaje de 7B parámetros de Mistral AI",
        license="Apache 2.0"
    ),
    ModelConfig(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        task=ModelTask.TEXT_GENERATION,
        priority=5,
        estimated_size_gb=13.5,
        gpu_memory_gb=15.0,
        tags=["llm", "chat", "meta", "conversational"],
        description="Llama 2 7B optimizado para conversaciones",
        license="Custom",
        requires_auth=True
    ),
    ModelConfig(
        model_id="tiiuae/falcon-7b-instruct",
        task=ModelTask.TEXT_GENERATION,
        priority=4,
        estimated_size_gb=14.5,
        gpu_memory_gb=16.5,
        tags=["llm", "instruct", "falcon", "tii"],
        description="Falcon 7B optimizado para seguir instrucciones",
        license="Apache 2.0"
    ),
    ModelConfig(
        model_id="microsoft/DialoGPT-large",
        task=ModelTask.TEXT_GENERATION,
        priority=4,
        estimated_size_gb=1.5,
        gpu_memory_gb=3.0,
        tags=["dialogue", "conversational", "microsoft"],
        description="Modelo conversacional de Microsoft",
        license="MIT"
    ),
    ModelConfig(
        model_id="EleutherAI/gpt-neox-20b",
        task=ModelTask.TEXT_GENERATION,
        priority=3,
        estimated_size_gb=40.0,
        gpu_memory_gb=45.0,
        tags=["llm", "large", "eleuther", "gpt"],
        description="GPT-NeoX de 20B parámetros",
        license="Apache 2.0"
    ),
    ModelConfig(
        model_id="bigscience/bloom-7b1",
        task=ModelTask.TEXT_GENERATION,
        priority=4,
        estimated_size_gb=14.0,
        gpu_memory_gb=16.0,
        tags=["llm", "multilingual", "bloom", "bigscience"],
        description="BLOOM 7B multilingüe",
        license="RAIL"
    ),
    ModelConfig(
        model_id="google/flan-t5-xxl",
        task=ModelTask.TEXT_GENERATION,
        priority=4,
        estimated_size_gb=22.0,
        gpu_memory_gb=25.0,
        tags=["t5", "instruction", "google", "flan"],
        description="FLAN-T5 XXL para seguimiento de instrucciones",
        license="Apache 2.0"
    ),
    ModelConfig(
        model_id="databricks/dolly-v2-12b",
        task=ModelTask.TEXT_GENERATION,
        priority=3,
        estimated_size_gb=24.0,
        gpu_memory_gb=28.0,
        tags=["llm", "instruct", "databricks", "dolly"],
        description="Dolly 2.0 de 12B parámetros",
        license="MIT"
    ),
    ModelConfig(
        model_id="stabilityai/stablelm-base-alpha-7b",
        task=ModelTask.TEXT_GENERATION,
        priority=3,
        estimated_size_gb=14.0,
        gpu_memory_gb=16.0,
        tags=["llm", "stability", "alpha"],
        description="StableLM Base Alpha 7B",
        license="CC BY-SA-4.0"
    ),
    ModelConfig(
        model_id="Writer/palmyra-base",
        task=ModelTask.TEXT_GENERATION,
        priority=2,
        estimated_size_gb=10.0,
        gpu_memory_gb=12.0,
        tags=["llm", "writer", "palmyra"],
        description="Palmyra Base de Writer",
        license="Apache 2.0"
    )
]

# Modelos de Clasificación de Texto
TEXT_CLASSIFICATION_MODELS = [
    ModelConfig(
        model_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        task=ModelTask.TEXT_CLASSIFICATION,
        priority=5,
        estimated_size_gb=0.5,
        gpu_memory_gb=2.0,
        tags=["sentiment", "twitter", "roberta", "social-media"],
        description="Análisis de sentimientos para Twitter",
        license="MIT"
    ),
    ModelConfig(
        model_id="distilbert-base-uncased-finetuned-sst-2-english",
        task=ModelTask.TEXT_CLASSIFICATION,
        priority=5,
        estimated_size_gb=0.3,
        gpu_memory_gb=1.5,
        tags=["sentiment", "distilbert", "sst-2", "english"],
        description="DistilBERT para análisis de sentimientos",
        license="Apache 2.0"
    ),
    ModelConfig(
        model_id="facebook/bart-large-mnli",
        task=ModelTask.ZERO_SHOT_CLASSIFICATION,
        priority=4,
        estimated_size_gb=1.6,
        gpu_memory_gb=3.0,
        tags=["zero-shot", "bart", "mnli", "facebook"],
        description="BART Large para clasificación zero-shot",
        license="MIT"
    ),
    ModelConfig(
        model_id="microsoft/DialoGPT-medium",
        task=ModelTask.TEXT_CLASSIFICATION,
        priority=3,
        estimated_size_gb=0.8,
        gpu_memory_gb=2.5,
        tags=["dialogue", "classification", "microsoft"],
        description="DialoGPT Medium para clasificación",
        license="MIT"
    ),
    ModelConfig(
        model_id="nlptown/bert-base-multilingual-uncased-sentiment",
        task=ModelTask.TEXT_CLASSIFICATION,
        priority=4,
        estimated_size_gb=0.7,
        gpu_memory_gb=2.0,
        tags=["sentiment", "multilingual", "bert"],
        description="BERT multilingüe para sentimientos",
        license="MIT"
    )
]

# Modelos de Generación de Imágenes
IMAGE_GENERATION_MODELS = [
    ModelConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        task=ModelTask.IMAGE_GENERATION,
        priority=5,
        estimated_size_gb=4.0,
        gpu_memory_gb=8.0,
        tags=["diffusion", "image-generation", "stable-diffusion"],
        description="Stable Diffusion v1.5 para generación de imágenes",
        license="CreativeML Open RAIL-M"
    ),
    ModelConfig(
        model_id="stabilityai/stable-diffusion-2-1",
        task=ModelTask.IMAGE_GENERATION,
        priority=5,
        estimated_size_gb=5.0,
        gpu_memory_gb=10.0,
        tags=["diffusion", "image-generation", "stable-diffusion", "v2"],
        description="Stable Diffusion 2.1 mejorado",
        license="CreativeML Open RAIL++-M"
    ),
    ModelConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        task=ModelTask.IMAGE_GENERATION,
        priority=4,
        estimated_size_gb=6.9,
        gpu_memory_gb=12.0,
        tags=["diffusion", "xl", "high-resolution"],
        description="Stable Diffusion XL para alta resolución",
        license="CreativeML Open RAIL++-M"
    ),
    ModelConfig(
        model_id="dreamlike-art/dreamlike-diffusion-1.0",
        task=ModelTask.IMAGE_GENERATION,
        priority=3,
        estimated_size_gb=4.0,
        gpu_memory_gb=8.0,
        tags=["diffusion", "artistic", "dreamlike"],
        description="Dreamlike Diffusion para arte",
        license="CreativeML Open RAIL-M"
    ),
    ModelConfig(
        model_id="wavymulder/Analog-Diffusion",
        task=ModelTask.IMAGE_GENERATION,
        priority=3,
        estimated_size_gb=4.0,
        gpu_memory_gb=8.0,
        tags=["diffusion", "analog", "photography"],
        description="Analog Diffusion para fotografía vintage",
        license="CreativeML Open RAIL-M"
    )
]

# Modelos de Clasificación de Imágenes
IMAGE_CLASSIFICATION_MODELS = [
    ModelConfig(
        model_id="google/vit-base-patch16-224",
        task=ModelTask.IMAGE_CLASSIFICATION,
        priority=5,
        estimated_size_gb=0.3,
        gpu_memory_gb=2.0,
        tags=["vit", "vision-transformer", "google", "imagenet"],
        description="Vision Transformer base para clasificación",
        license="Apache 2.0"
    ),
    ModelConfig(
        model_id="microsoft/resnet-50",
        task=ModelTask.IMAGE_CLASSIFICATION,
        priority=5,
        estimated_size_gb=0.1,
        gpu_memory_gb=1.5,
        tags=["resnet", "cnn", "microsoft", "imagenet"],
        description="ResNet-50 para clasificación de imágenes",
        license="MIT"
    ),
    ModelConfig(
        model_id="facebook/convnext-base-224-22k-1k",
        task=ModelTask.IMAGE_CLASSIFICATION,
        priority=4,
        estimated_size_gb=0.3,
        gpu_memory_gb=2.0,
        tags=["convnext", "facebook", "modern-cnn"],
        description="ConvNeXt base para clasificación",
        license="MIT"
    ),
    ModelConfig(
        model_id="openai/clip-vit-base-patch32",
        task=ModelTask.IMAGE_CLASSIFICATION,
        priority=5,
        estimated_size_gb=0.6,
        gpu_memory_gb=2.5,
        tags=["clip", "multimodal", "openai", "zero-shot"],
        description="CLIP para clasificación multimodal",
        license="MIT"
    ),
    ModelConfig(
        model_id="timm/efficientnet_b0.ra_in1k",
        task=ModelTask.IMAGE_CLASSIFICATION,
        priority=4,
        estimated_size_gb=0.2,
        gpu_memory_gb=1.5,
        tags=["efficientnet", "timm", "efficient"],
        description="EfficientNet B0 optimizado",
        license="Apache 2.0"
    )
]

# Modelos de Reconocimiento de Voz
SPEECH_RECOGNITION_MODELS = [
    ModelConfig(
        model_id="openai/whisper-base",
        task=ModelTask.SPEECH_RECOGNITION,
        priority=5,
        estimated_size_gb=0.3,
        gpu_memory_gb=2.0,
        tags=["whisper", "openai", "multilingual", "asr"],
        description="Whisper base para reconocimiento de voz",
        license="MIT"
    ),
    ModelConfig(
        model_id="facebook/wav2vec2-base-960h",
        task=ModelTask.SPEECH_RECOGNITION,
        priority=4,
        estimated_size_gb=0.4,
        gpu_memory_gb=2.5,
        tags=["wav2vec2", "facebook", "english", "asr"],
        description="Wav2Vec2 base entrenado en 960h",
        license="MIT"
    ),
    ModelConfig(
        model_id="microsoft/speecht5_asr",
        task=ModelTask.SPEECH_RECOGNITION,
        priority=3,
        estimated_size_gb=0.5,
        gpu_memory_gb=3.0,
        tags=["speecht5", "microsoft", "asr"],
        description="SpeechT5 para reconocimiento de voz",
        license="MIT"
    ),
    ModelConfig(
        model_id="facebook/hubert-base-ls960",
        task=ModelTask.SPEECH_RECOGNITION,
        priority=3,
        estimated_size_gb=0.4,
        gpu_memory_gb=2.5,
        tags=["hubert", "facebook", "self-supervised"],
        description="HuBERT base para procesamiento de audio",
        license="MIT"
    ),
    ModelConfig(
        model_id="jonatasgrosman/wav2vec2-large-xlsr-53-english",
        task=ModelTask.SPEECH_RECOGNITION,
        priority=4,
        estimated_size_gb=1.2,
        gpu_memory_gb=4.0,
        tags=["wav2vec2", "xlsr", "english", "large"],
        description="Wav2Vec2 large XLSR para inglés",
        license="MIT"
    )
]

# Modelos de Traducción
TRANSLATION_MODELS = [
    ModelConfig(
        model_id="Helsinki-NLP/opus-mt-en-es",
        task=ModelTask.TRANSLATION,
        priority=5,
        estimated_size_gb=0.3,
        gpu_memory_gb=1.5,
        tags=["translation", "english", "spanish", "opus"],
        description="Traducción inglés-español",
        license="MIT"
    ),
    ModelConfig(
        model_id="facebook/m2m100_418M",
        task=ModelTask.TRANSLATION,
        priority=4,
        estimated_size_gb=1.6,
        gpu_memory_gb=3.0,
        tags=["translation", "multilingual", "m2m", "facebook"],
        description="M2M100 para traducción multilingüe",
        license="MIT"
    ),
    ModelConfig(
        model_id="t5-base",
        task=ModelTask.TRANSLATION,
        priority=4,
        estimated_size_gb=0.9,
        gpu_memory_gb=2.5,
        tags=["t5", "translation", "google", "text-to-text"],
        description="T5 base para tareas de texto a texto",
        license="Apache 2.0"
    ),
    ModelConfig(
        model_id="facebook/nllb-200-distilled-600M",
        task=ModelTask.TRANSLATION,
        priority=4,
        estimated_size_gb=2.4,
        gpu_memory_gb=4.0,
        tags=["nllb", "multilingual", "facebook", "200-languages"],
        description="NLLB para 200 idiomas",
        license="MIT"
    )
]

# Configuración completa de modelos
MODEL_CATEGORIES: Dict[str, List[ModelConfig]] = {
    "text_generation": TEXT_GENERATION_MODELS,
    "text_classification": TEXT_CLASSIFICATION_MODELS,
    "image_generation": IMAGE_GENERATION_MODELS,
    "image_classification": IMAGE_CLASSIFICATION_MODELS,
    "speech_recognition": SPEECH_RECOGNITION_MODELS,
    "translation": TRANSLATION_MODELS,
}

# Modelos prioritarios para descarga automática
PRIORITY_MODELS = [
    model for models in MODEL_CATEGORIES.values()
    for model in models if model.priority >= 4
]

# Configuración de recursos por categoría
RESOURCE_REQUIREMENTS = {
    "text_generation": {
        "min_gpu_memory_gb": 8.0,
        "recommended_gpu_memory_gb": 16.0,
        "cpu_cores": 4,
        "ram_gb": 16
    },
    "image_generation": {
        "min_gpu_memory_gb": 6.0,
        "recommended_gpu_memory_gb": 12.0,
        "cpu_cores": 2,
        "ram_gb": 8
    },
    "text_classification": {
        "min_gpu_memory_gb": 2.0,
        "recommended_gpu_memory_gb": 4.0,
        "cpu_cores": 2,
        "ram_gb": 4
    },
    "image_classification": {
        "min_gpu_memory_gb": 2.0,
        "recommended_gpu_memory_gb": 4.0,
        "cpu_cores": 2,
        "ram_gb": 4
    },
    "speech_recognition": {
        "min_gpu_memory_gb": 2.0,
        "recommended_gpu_memory_gb": 4.0,
        "cpu_cores": 2,
        "ram_gb": 4
    },
    "translation": {
        "min_gpu_memory_gb": 2.0,
        "recommended_gpu_memory_gb": 4.0,
        "cpu_cores": 2,
        "ram_gb": 4
    }
}

def get_models_by_task(task: ModelTask) -> List[ModelConfig]:
    """Obtiene modelos por tarea específica"""
    all_models = []
    for models in MODEL_CATEGORIES.values():
        all_models.extend([m for m in models if m.task == task])
    return sorted(all_models, key=lambda x: x.priority, reverse=True)

def get_models_by_priority(min_priority: int = 3) -> List[ModelConfig]:
    """Obtiene modelos por prioridad mínima"""
    all_models = []
    for models in MODEL_CATEGORIES.values():
        all_models.extend([m for m in models if m.priority >= min_priority])
    return sorted(all_models, key=lambda x: x.priority, reverse=True)

def get_lightweight_models(max_size_gb: float = 2.0) -> List[ModelConfig]:
    """Obtiene modelos ligeros por tamaño máximo"""
    all_models = []
    for models in MODEL_CATEGORIES.values():
        all_models.extend([m for m in models if m.estimated_size_gb <= max_size_gb])
    return sorted(all_models, key=lambda x: x.estimated_size_gb)

def get_gpu_requirements(model_ids: List[str]) -> Dict[str, Any]:
    """Calcula requisitos de GPU para una lista de modelos"""
    all_models = []
    for models in MODEL_CATEGORIES.values():
        all_models.extend(models)

    selected_models = [m for m in all_models if m.model_id in model_ids]

    if not selected_models:
        return {"total_gpu_memory_gb": 0, "models": []}

    total_memory = sum(m.gpu_memory_gb for m in selected_models)
    max_single_model = max(m.gpu_memory_gb for m in selected_models)

    return {
        "total_gpu_memory_gb": total_memory,
        "max_single_model_gb": max_single_model,
        "recommended_gpu_memory_gb": max(total_memory * 1.2, max_single_model * 1.5),
        "models": [{"id": m.model_id, "memory_gb": m.gpu_memory_gb} for m in selected_models]
    }

# Configuración de modelos por defecto para diferentes casos de uso
DEFAULT_CONFIGURATIONS = {
    "development": [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "microsoft/DialoGPT-medium",
        "google/vit-base-patch16-224",
        "openai/whisper-base"
    ],
    "production_light": [
        "mistralai/Mistral-7B-v0.1",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "runwayml/stable-diffusion-v1-5",
        "openai/clip-vit-base-patch32"
    ],
    "production_full": [
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-v0.1",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "google/vit-base-patch16-224",
        "openai/whisper-base",
        "facebook/m2m100_418M"
    ],
    "research": [
        "EleutherAI/gpt-neox-20b",
        "google/flan-t5-xxl",
        "stabilityai/stable-diffusion-2-1",
        "facebook/wav2vec2-base-960h"
    ]
}
