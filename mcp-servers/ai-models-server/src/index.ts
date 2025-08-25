#!/usr/bin/env node
/**
 * Servidor MCP para Gestión de Modelos de IA Open-Source
 * Proporciona herramientas y recursos para interactuar con más de 2000 modelos
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import axios from 'axios';

// Configuración del servidor API
const API_BASE_URL = process.env.AI_API_URL || 'http://localhost:8000';
const API_TOKEN = process.env.AI_API_TOKEN;

interface ModelInfo {
  model_id: string;
  task: string;
  downloads: number;
  tags: string[];
  size_gb: number;
  gpu_memory_gb: number;
  status: string;
  loaded_at?: number;
}

interface InferenceRequest {
  model_id: string;
  input_data: any;
  parameters?: Record<string, any>;
}

class AIModelsServer {
  private server: Server;
  private axiosInstance;

  constructor() {
    this.server = new Server(
      {
        name: 'ai-models-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Configurar cliente HTTP
    this.axiosInstance = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: API_TOKEN ? { 'Authorization': `Bearer ${API_TOKEN}` } : {},
    });

    this.setupResourceHandlers();
    this.setupToolHandlers();

    // Manejo de errores
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupResourceHandlers() {
    // Recursos estáticos
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: [
        {
          uri: 'ai-models://catalog/all',
          name: 'Catálogo completo de modelos de IA',
          mimeType: 'application/json',
          description: 'Lista completa de todos los modelos disponibles con metadatos'
        },
        {
          uri: 'ai-models://stats/system',
          name: 'Estadísticas del sistema',
          mimeType: 'application/json',
          description: 'Estadísticas en tiempo real del sistema de IA'
        },
        {
          uri: 'ai-models://models/loaded',
          name: 'Modelos cargados',
          mimeType: 'application/json',
          description: 'Lista de modelos actualmente cargados en memoria'
        }
      ],
    }));

    // Plantillas de recursos dinámicos
    this.server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({
      resourceTemplates: [
        {
          uriTemplate: 'ai-models://model/{model_id}/info',
          name: 'Información detallada del modelo',
          mimeType: 'application/json',
          description: 'Información completa de un modelo específico'
        },
        {
          uriTemplate: 'ai-models://task/{task}/models',
          name: 'Modelos por tarea',
          mimeType: 'application/json',
          description: 'Lista de modelos filtrados por tarea específica'
        },
        {
          uriTemplate: 'ai-models://inference/{model_id}/history',
          name: 'Historial de inferencias',
          mimeType: 'application/json',
          description: 'Historial de inferencias para un modelo específico'
        }
      ],
    }));

    // Manejo de lectura de recursos
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const uri = request.params.uri;

      try {
        if (uri === 'ai-models://catalog/all') {
          return await this.getCatalogResource();
        } else if (uri === 'ai-models://stats/system') {
          return await this.getSystemStatsResource();
        } else if (uri === 'ai-models://models/loaded') {
          return await this.getLoadedModelsResource();
        } else if (uri.startsWith('ai-models://model/')) {
          return await this.getModelInfoResource(uri);
        } else if (uri.startsWith('ai-models://task/')) {
          return await this.getTaskModelsResource(uri);
        } else {
          throw new McpError(ErrorCode.InvalidRequest, `Recurso no encontrado: ${uri}`);
        }
      } catch (error) {
        if (error instanceof McpError) throw error;
        throw new McpError(ErrorCode.InternalError, `Error accediendo al recurso: ${error}`);
      }
    });
  }

  private setupToolHandlers() {
    // Lista de herramientas disponibles
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'list_models',
          description: 'Lista modelos disponibles con filtros opcionales',
          inputSchema: {
            type: 'object',
            properties: {
              task: {
                type: 'string',
                description: 'Filtrar por tarea específica (text-generation, image-classification, etc.)'
              },
              min_downloads: {
                type: 'number',
                description: 'Número mínimo de descargas',
                default: 0
              },
              max_size_gb: {
                type: 'number',
                description: 'Tamaño máximo en GB'
              },
              limit: {
                type: 'number',
                description: 'Número máximo de resultados',
                default: 50
              }
            }
          }
        },
        {
          name: 'load_model',
          description: 'Carga un modelo específico en memoria',
          inputSchema: {
            type: 'object',
            properties: {
              model_id: {
                type: 'string',
                description: 'ID del modelo a cargar'
              }
            },
            required: ['model_id']
          }
        },
        {
          name: 'unload_model',
          description: 'Descarga un modelo de memoria',
          inputSchema: {
            type: 'object',
            properties: {
              model_id: {
                type: 'string',
                description: 'ID del modelo a descargar'
              }
            },
            required: ['model_id']
          }
        },
        {
          name: 'generate_text',
          description: 'Genera texto usando un modelo de lenguaje',
          inputSchema: {
            type: 'object',
            properties: {
              model_id: {
                type: 'string',
                description: 'ID del modelo de generación de texto'
              },
              prompt: {
                type: 'string',
                description: 'Texto de entrada para generar'
              },
              max_length: {
                type: 'number',
                description: 'Longitud máxima del texto generado',
                default: 512
              },
              temperature: {
                type: 'number',
                description: 'Temperatura para la generación (0.1-2.0)',
                default: 0.7
              },
              top_p: {
                type: 'number',
                description: 'Top-p sampling (0.1-1.0)',
                default: 0.9
              }
            },
            required: ['model_id', 'prompt']
          }
        },
        {
          name: 'generate_image',
          description: 'Genera imágenes usando modelos de difusión',
          inputSchema: {
            type: 'object',
            properties: {
              model_id: {
                type: 'string',
                description: 'ID del modelo de generación de imágenes'
              },
              prompt: {
                type: 'string',
                description: 'Descripción de la imagen a generar'
              },
              negative_prompt: {
                type: 'string',
                description: 'Prompt negativo (opcional)'
              },
              width: {
                type: 'number',
                description: 'Ancho de la imagen',
                default: 512
              },
              height: {
                type: 'number',
                description: 'Alto de la imagen',
                default: 512
              },
              num_inference_steps: {
                type: 'number',
                description: 'Número de pasos de inferencia',
                default: 50
              }
            },
            required: ['model_id', 'prompt']
          }
        },
        {
          name: 'classify_text',
          description: 'Clasifica texto usando modelos de clasificación',
          inputSchema: {
            type: 'object',
            properties: {
              model_id: {
                type: 'string',
                description: 'ID del modelo de clasificación'
              },
              text: {
                type: 'string',
                description: 'Texto a clasificar'
              }
            },
            required: ['model_id', 'text']
          }
        },
        {
          name: 'batch_inference',
          description: 'Ejecuta inferencia en lote para múltiples solicitudes',
          inputSchema: {
            type: 'object',
            properties: {
              requests: {
                type: 'array',
                description: 'Lista de solicitudes de inferencia',
                items: {
                  type: 'object',
                  properties: {
                    model_id: { type: 'string' },
                    input_data: {},
                    parameters: { type: 'object' }
                  },
                  required: ['model_id', 'input_data']
                }
              },
              max_concurrent: {
                type: 'number',
                description: 'Número máximo de solicitudes concurrentes',
                default: 5
              }
            },
            required: ['requests']
          }
        },
        {
          name: 'get_system_stats',
          description: 'Obtiene estadísticas del sistema de IA',
          inputSchema: {
            type: 'object',
            properties: {}
          }
        },
        {
          name: 'search_models',
          description: 'Busca modelos por nombre, descripción o tags',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Término de búsqueda'
              },
              task: {
                type: 'string',
                description: 'Filtrar por tarea específica'
              },
              limit: {
                type: 'number',
                description: 'Número máximo de resultados',
                default: 20
              }
            },
            required: ['query']
          }
        }
      ],
    }));

    // Manejo de llamadas a herramientas
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'list_models':
            return await this.listModels(args);
          case 'load_model':
            return await this.loadModel(args);
          case 'unload_model':
            return await this.unloadModel(args);
          case 'generate_text':
            return await this.generateText(args);
          case 'generate_image':
            return await this.generateImage(args);
          case 'classify_text':
            return await this.classifyText(args);
          case 'batch_inference':
            return await this.batchInference(args);
          case 'get_system_stats':
            return await this.getSystemStats(args);
          case 'search_models':
            return await this.searchModels(args);
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Herramienta desconocida: ${name}`);
        }
      } catch (error) {
        if (error instanceof McpError) throw error;
        return {
          content: [
            {
              type: 'text',
              text: `Error ejecutando ${name}: ${error}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  // Implementación de recursos
  private async getCatalogResource() {
    const response = await this.axiosInstance.get('/models/available?limit=1000');
    return {
      contents: [
        {
          uri: 'ai-models://catalog/all',
          mimeType: 'application/json',
          text: JSON.stringify({
            total_models: response.data.length,
            models: response.data,
            timestamp: new Date().toISOString()
          }, null, 2),
        },
      ],
    };
  }

  private async getSystemStatsResource() {
    const response = await this.axiosInstance.get('/stats/system');
    return {
      contents: [
        {
          uri: 'ai-models://stats/system',
          mimeType: 'application/json',
          text: JSON.stringify(response.data, null, 2),
        },
      ],
    };
  }

  private async getLoadedModelsResource() {
    const response = await this.axiosInstance.get('/models/loaded');
    return {
      contents: [
        {
          uri: 'ai-models://models/loaded',
          mimeType: 'application/json',
          text: JSON.stringify(response.data, null, 2),
        },
      ],
    };
  }

  private async getModelInfoResource(uri: string) {
    const modelId = uri.split('/')[2];
    const response = await this.axiosInstance.get(`/models/info/${modelId}`);
    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(response.data, null, 2),
        },
      ],
    };
  }

  private async getTaskModelsResource(uri: string) {
    const task = uri.split('/')[2];
    const response = await this.axiosInstance.get(`/models/available?task=${task}&limit=100`);
    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify({
            task,
            models: response.data,
            count: response.data.length
          }, null, 2),
        },
      ],
    };
  }

  // Implementación de herramientas
  private async listModels(args: any) {
    const params = new URLSearchParams();
    if (args.task) params.append('task', args.task);
    if (args.min_downloads) params.append('min_downloads', args.min_downloads.toString());
    if (args.max_size_gb) params.append('max_size_gb', args.max_size_gb.toString());
    if (args.limit) params.append('limit', args.limit.toString());

    const response = await this.axiosInstance.get(`/models/available?${params}`);

    return {
      content: [
        {
          type: 'text',
          text: `Encontrados ${response.data.length} modelos:\n\n` +
                response.data.map((model: ModelInfo) =>
                  `• ${model.model_id}\n  Tarea: ${model.task}\n  Descargas: ${model.downloads.toLocaleString()}\n  Tamaño: ${model.size_gb}GB\n  Estado: ${model.status}\n`
                ).join('\n'),
        },
      ],
    };
  }

  private async loadModel(args: any) {
    const response = await this.axiosInstance.post(`/models/load/${args.model_id}`);

    return {
      content: [
        {
          type: 'text',
          text: `Modelo ${args.model_id} cargándose en memoria...\nEstado: ${response.data.status}\nMensaje: ${response.data.message}`,
        },
      ],
    };
  }

  private async unloadModel(args: any) {
    const response = await this.axiosInstance.delete(`/models/unload/${args.model_id}`);

    return {
      content: [
        {
          type: 'text',
          text: `Modelo ${args.model_id} descargado de memoria.\nEstado: ${response.data.status}\nMensaje: ${response.data.message}`,
        },
      ],
    };
  }

  private async generateText(args: any) {
    const response = await this.axiosInstance.post('/inference/text-generation', {
      model_id: args.model_id,
      prompt: args.prompt,
      max_length: args.max_length || 512,
      temperature: args.temperature || 0.7,
      top_p: args.top_p || 0.9,
      do_sample: true
    });

    return {
      content: [
        {
          type: 'text',
          text: `Texto generado por ${args.model_id}:\n\n${response.data.output}\n\n` +
                `Tiempo de inferencia: ${response.data.inference_time.toFixed(2)}s\n` +
                `Memoria GPU usada: ${response.data.gpu_memory_used.toFixed(2)}GB`,
        },
      ],
    };
  }

  private async generateImage(args: any) {
    const response = await this.axiosInstance.post('/inference/image-generation', {
      model_id: args.model_id,
      prompt: args.prompt,
      negative_prompt: args.negative_prompt || '',
      width: args.width || 512,
      height: args.height || 512,
      num_inference_steps: args.num_inference_steps || 50,
      guidance_scale: 7.5
    });

    return {
      content: [
        {
          type: 'text',
          text: `Imagen generada por ${args.model_id}:\n\n` +
                `Prompt: "${args.prompt}"\n` +
                `Dimensiones: ${response.data.output.width}x${response.data.output.height}\n` +
                `Tiempo de inferencia: ${response.data.inference_time.toFixed(2)}s\n` +
                `Memoria GPU usada: ${response.data.gpu_memory_used.toFixed(2)}GB\n\n` +
                `Imagen en base64: ${response.data.output.image_base64.substring(0, 100)}...`,
        },
      ],
    };
  }

  private async classifyText(args: any) {
    const response = await this.axiosInstance.post('/inference/classification', {
      model_id: args.model_id,
      text: args.text
    });

    const results = response.data.output.slice(0, 5); // Top 5 resultados
    const resultText = results.map((r: any) =>
      `• ${r.label}: ${(r.score * 100).toFixed(2)}%`
    ).join('\n');

    return {
      content: [
        {
          type: 'text',
          text: `Clasificación de texto por ${args.model_id}:\n\n` +
                `Texto: "${args.text}"\n\n` +
                `Resultados:\n${resultText}\n\n` +
                `Tiempo de inferencia: ${response.data.inference_time.toFixed(2)}s`,
        },
      ],
    };
  }

  private async batchInference(args: any) {
    const response = await this.axiosInstance.post('/inference/batch', {
      requests: args.requests,
      max_concurrent: args.max_concurrent || 5
    });

    return {
      content: [
        {
          type: 'text',
          text: `Inferencia en lote completada:\n\n` +
                `Total de solicitudes: ${response.data.total_requests}\n` +
                `Exitosas: ${response.data.successful}\n` +
                `Fallidas: ${response.data.failed}\n\n` +
                `Resultados disponibles en la respuesta completa.`,
        },
      ],
    };
  }

  private async getSystemStats(args: any) {
    const response = await this.axiosInstance.get('/stats/system');
    const stats = response.data;

    return {
      content: [
        {
          type: 'text',
          text: `Estadísticas del Sistema de IA:\n\n` +
                `• Modelos cargados: ${stats.loaded_models}\n` +
                `• Modelos disponibles: ${stats.available_models}\n` +
                `• CPU: ${stats.cpu_percent.toFixed(1)}%\n` +
                `• Memoria RAM: ${stats.memory_percent.toFixed(1)}%\n` +
                `• Memoria GPU: ${stats.gpu_memory_gb.toFixed(2)}GB\n` +
                `• Cache de modelos: ${stats.cache_memory_gb.toFixed(2)}GB`,
        },
      ],
    };
  }

  private async searchModels(args: any) {
    const params = new URLSearchParams();
    params.append('limit', (args.limit || 20).toString());
    if (args.task) params.append('task', args.task);

    const response = await this.axiosInstance.get(`/models/available?${params}`);

    // Filtrar por query en el lado cliente
    const query = args.query.toLowerCase();
    const filteredModels = response.data.filter((model: ModelInfo) =>
      model.model_id.toLowerCase().includes(query) ||
      model.tags.some(tag => tag.toLowerCase().includes(query))
    );

    return {
      content: [
        {
          type: 'text',
          text: `Búsqueda: "${args.query}" - Encontrados ${filteredModels.length} modelos:\n\n` +
                filteredModels.map((model: ModelInfo) =>
                  `• ${model.model_id}\n  Tarea: ${model.task}\n  Descargas: ${model.downloads.toLocaleString()}\n`
                ).join('\n'),
        },
      ],
    };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Servidor MCP de Modelos de IA ejecutándose en stdio');
  }
}

const server = new AIModelsServer();
server.run().catch(console.error);
