#!/usr/bin/env node
/**
 * Servidor MCP para Gestión de Kubernetes
 * Proporciona herramientas para desplegar y gestionar aplicaciones en Kubernetes
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
import * as k8s from '@kubernetes/client-node';
import * as yaml from 'yaml';

class KubernetesServer {
  private server: Server;
  private k8sApi: k8s.CoreV1Api;
  private k8sAppsApi: k8s.AppsV1Api;
  private k8sNetworkingApi: k8s.NetworkingV1Api;
  private k8sConfig: k8s.KubeConfig;

  constructor() {
    this.server = new Server(
      {
        name: 'kubernetes-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Configurar cliente de Kubernetes
    this.k8sConfig = new k8s.KubeConfig();
    this.k8sConfig.loadFromDefault();

    this.k8sApi = this.k8sConfig.makeApiClient(k8s.CoreV1Api);
    this.k8sAppsApi = this.k8sConfig.makeApiClient(k8s.AppsV1Api);
    this.k8sNetworkingApi = this.k8sConfig.makeApiClient(k8s.NetworkingV1Api);

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
          uri: 'k8s://cluster/info',
          name: 'Información del cluster',
          mimeType: 'application/json',
          description: 'Información general del cluster de Kubernetes'
        },
        {
          uri: 'k8s://nodes/all',
          name: 'Todos los nodos',
          mimeType: 'application/json',
          description: 'Lista de todos los nodos del cluster'
        },
        {
          uri: 'k8s://namespaces/all',
          name: 'Todos los namespaces',
          mimeType: 'application/json',
          description: 'Lista de todos los namespaces'
        }
      ],
    }));

    // Plantillas de recursos dinámicos
    this.server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({
      resourceTemplates: [
        {
          uriTemplate: 'k8s://namespace/{namespace}/pods',
          name: 'Pods por namespace',
          mimeType: 'application/json',
          description: 'Lista de pods en un namespace específico'
        },
        {
          uriTemplate: 'k8s://namespace/{namespace}/services',
          name: 'Servicios por namespace',
          mimeType: 'application/json',
          description: 'Lista de servicios en un namespace específico'
        },
        {
          uriTemplate: 'k8s://namespace/{namespace}/deployments',
          name: 'Deployments por namespace',
          mimeType: 'application/json',
          description: 'Lista de deployments en un namespace específico'
        },
        {
          uriTemplate: 'k8s://pod/{namespace}/{name}/logs',
          name: 'Logs de pod',
          mimeType: 'text/plain',
          description: 'Logs de un pod específico'
        }
      ],
    }));

    // Manejo de lectura de recursos
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const uri = request.params.uri;

      try {
        if (uri === 'k8s://cluster/info') {
          return await this.getClusterInfoResource();
        } else if (uri === 'k8s://nodes/all') {
          return await this.getNodesResource();
        } else if (uri === 'k8s://namespaces/all') {
          return await this.getNamespacesResource();
        } else if (uri.includes('/pods')) {
          return await this.getPodsResource(uri);
        } else if (uri.includes('/services')) {
          return await this.getServicesResource(uri);
        } else if (uri.includes('/deployments')) {
          return await this.getDeploymentsResource(uri);
        } else if (uri.includes('/logs')) {
          return await this.getPodLogsResource(uri);
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
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'get_pods',
          description: 'Lista pods en un namespace',
          inputSchema: {
            type: 'object',
            properties: {
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              }
            }
          }
        },
        {
          name: 'get_services',
          description: 'Lista servicios en un namespace',
          inputSchema: {
            type: 'object',
            properties: {
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              }
            }
          }
        },
        {
          name: 'get_deployments',
          description: 'Lista deployments en un namespace',
          inputSchema: {
            type: 'object',
            properties: {
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              }
            }
          }
        },
        {
          name: 'create_deployment',
          description: 'Crea un nuevo deployment',
          inputSchema: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
                description: 'Nombre del deployment'
              },
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              },
              image: {
                type: 'string',
                description: 'Imagen del contenedor'
              },
              replicas: {
                type: 'number',
                description: 'Número de réplicas',
                default: 1
              },
              port: {
                type: 'number',
                description: 'Puerto del contenedor'
              },
              env_vars: {
                type: 'object',
                description: 'Variables de entorno'
              }
            },
            required: ['name', 'image']
          }
        },
        {
          name: 'create_service',
          description: 'Crea un nuevo servicio',
          inputSchema: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
                description: 'Nombre del servicio'
              },
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              },
              selector: {
                type: 'object',
                description: 'Selector para los pods'
              },
              port: {
                type: 'number',
                description: 'Puerto del servicio'
              },
              target_port: {
                type: 'number',
                description: 'Puerto del contenedor'
              },
              service_type: {
                type: 'string',
                description: 'Tipo de servicio (ClusterIP, NodePort, LoadBalancer)',
                default: 'ClusterIP'
              }
            },
            required: ['name', 'selector', 'port']
          }
        },
        {
          name: 'scale_deployment',
          description: 'Escala un deployment',
          inputSchema: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
                description: 'Nombre del deployment'
              },
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              },
              replicas: {
                type: 'number',
                description: 'Número de réplicas'
              }
            },
            required: ['name', 'replicas']
          }
        },
        {
          name: 'delete_deployment',
          description: 'Elimina un deployment',
          inputSchema: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
                description: 'Nombre del deployment'
              },
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              }
            },
            required: ['name']
          }
        },
        {
          name: 'get_pod_logs',
          description: 'Obtiene logs de un pod',
          inputSchema: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
                description: 'Nombre del pod'
              },
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              },
              lines: {
                type: 'number',
                description: 'Número de líneas (default: 100)',
                default: 100
              }
            },
            required: ['name']
          }
        },
        {
          name: 'apply_yaml',
          description: 'Aplica un manifiesto YAML',
          inputSchema: {
            type: 'object',
            properties: {
              yaml_content: {
                type: 'string',
                description: 'Contenido del manifiesto YAML'
              },
              namespace: {
                type: 'string',
                description: 'Namespace (default: default)',
                default: 'default'
              }
            },
            required: ['yaml_content']
          }
        },
        {
          name: 'get_cluster_info',
          description: 'Obtiene información del cluster',
          inputSchema: {
            type: 'object',
            properties: {}
          }
        }
      ],
    }));

    // Manejo de llamadas a herramientas
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'get_pods':
            return await this.getPods(args);
          case 'get_services':
            return await this.getServices(args);
          case 'get_deployments':
            return await this.getDeployments(args);
          case 'create_deployment':
            return await this.createDeployment(args);
          case 'create_service':
            return await this.createService(args);
          case 'scale_deployment':
            return await this.scaleDeployment(args);
          case 'delete_deployment':
            return await this.deleteDeployment(args);
          case 'get_pod_logs':
            return await this.getPodLogs(args);
          case 'apply_yaml':
            return await this.applyYaml(args);
          case 'get_cluster_info':
            return await this.getClusterInfo(args);
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
  private async getClusterInfoResource() {
    const nodes = await this.k8sApi.listNode();
    const namespaces = await this.k8sApi.listNamespace();

    const clusterInfo = {
      nodes: nodes.body.items.length,
      namespaces: namespaces.body.items.length,
      kubernetes_version: nodes.body.items[0]?.status?.nodeInfo?.kubeletVersion || 'unknown',
      timestamp: new Date().toISOString()
    };

    return {
      contents: [
        {
          uri: 'k8s://cluster/info',
          mimeType: 'application/json',
          text: JSON.stringify(clusterInfo, null, 2),
        },
      ],
    };
  }

  private async getNodesResource() {
    const response = await this.k8sApi.listNode();
    const nodes = response.body.items.map(node => ({
      name: node.metadata?.name,
      status: node.status?.conditions?.find(c => c.type === 'Ready')?.status,
      version: node.status?.nodeInfo?.kubeletVersion,
      os: node.status?.nodeInfo?.osImage,
      capacity: node.status?.capacity
    }));

    return {
      contents: [
        {
          uri: 'k8s://nodes/all',
          mimeType: 'application/json',
          text: JSON.stringify(nodes, null, 2),
        },
      ],
    };
  }

  private async getNamespacesResource() {
    const response = await this.k8sApi.listNamespace();
    const namespaces = response.body.items.map(ns => ({
      name: ns.metadata?.name,
      status: ns.status?.phase,
      created: ns.metadata?.creationTimestamp
    }));

    return {
      contents: [
        {
          uri: 'k8s://namespaces/all',
          mimeType: 'application/json',
          text: JSON.stringify(namespaces, null, 2),
        },
      ],
    };
  }

  private async getPodsResource(uri: string) {
    const namespace = uri.split('/')[2];
    const response = await this.k8sApi.listNamespacedPod(namespace);

    const pods = response.body.items.map(pod => ({
      name: pod.metadata?.name,
      status: pod.status?.phase,
      ready: pod.status?.containerStatuses?.every(c => c.ready) || false,
      restarts: pod.status?.containerStatuses?.reduce((sum, c) => sum + c.restartCount, 0) || 0,
      created: pod.metadata?.creationTimestamp
    }));

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(pods, null, 2),
        },
      ],
    };
  }

  private async getServicesResource(uri: string) {
    const namespace = uri.split('/')[2];
    const response = await this.k8sApi.listNamespacedService(namespace);

    const services = response.body.items.map(svc => ({
      name: svc.metadata?.name,
      type: svc.spec?.type,
      cluster_ip: svc.spec?.clusterIP,
      ports: svc.spec?.ports,
      selector: svc.spec?.selector
    }));

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(services, null, 2),
        },
      ],
    };
  }

  private async getDeploymentsResource(uri: string) {
    const namespace = uri.split('/')[2];
    const response = await this.k8sAppsApi.listNamespacedDeployment(namespace);

    const deployments = response.body.items.map(dep => ({
      name: dep.metadata?.name,
      replicas: dep.spec?.replicas,
      ready_replicas: dep.status?.readyReplicas || 0,
      available_replicas: dep.status?.availableReplicas || 0,
      created: dep.metadata?.creationTimestamp
    }));

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(deployments, null, 2),
        },
      ],
    };
  }

  private async getPodLogsResource(uri: string) {
    const parts = uri.split('/');
    const namespace = parts[2];
    const podName = parts[3];

    const response = await this.k8sApi.readNamespacedPodLog(
      podName,
      namespace,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      100 // últimas 100 líneas
    );

    return {
      contents: [
        {
          uri,
          mimeType: 'text/plain',
          text: response.body,
        },
      ],
    };
  }

  // Implementación de herramientas
  private async getPods(args: any) {
    const namespace = args.namespace || 'default';
    const response = await this.k8sApi.listNamespacedPod(namespace);

    const pods = response.body.items.map(pod => ({
      name: pod.metadata?.name,
      status: pod.status?.phase,
      ready: pod.status?.containerStatuses?.every(c => c.ready) || false,
      restarts: pod.status?.containerStatuses?.reduce((sum, c) => sum + c.restartCount, 0) || 0,
      age: this.calculateAge(pod.metadata?.creationTimestamp)
    }));

    return {
      content: [
        {
          type: 'text',
          text: `Pods en namespace '${namespace}':\n\n` +
                pods.map(pod =>
                  `• ${pod.name}\n  Estado: ${pod.status}\n  Listo: ${pod.ready}\n  Reinicios: ${pod.restarts}\n  Edad: ${pod.age}\n`
                ).join('\n'),
        },
      ],
    };
  }

  private async getServices(args: any) {
    const namespace = args.namespace || 'default';
    const response = await this.k8sApi.listNamespacedService(namespace);

    const services = response.body.items.map(svc => ({
      name: svc.metadata?.name,
      type: svc.spec?.type,
      cluster_ip: svc.spec?.clusterIP,
      ports: svc.spec?.ports?.map(p => `${p.port}:${p.targetPort}/${p.protocol}`).join(', ')
    }));

    return {
      content: [
        {
          type: 'text',
          text: `Servicios en namespace '${namespace}':\n\n` +
                services.map(svc =>
                  `• ${svc.name}\n  Tipo: ${svc.type}\n  IP: ${svc.cluster_ip}\n  Puertos: ${svc.ports}\n`
                ).join('\n'),
        },
      ],
    };
  }

  private async getDeployments(args: any) {
    const namespace = args.namespace || 'default';
    const response = await this.k8sAppsApi.listNamespacedDeployment(namespace);

    const deployments = response.body.items.map(dep => ({
      name: dep.metadata?.name,
      replicas: dep.spec?.replicas,
      ready: dep.status?.readyReplicas || 0,
      available: dep.status?.availableReplicas || 0,
      age: this.calculateAge(dep.metadata?.creationTimestamp)
    }));

    return {
      content: [
        {
          type: 'text',
          text: `Deployments en namespace '${namespace}':\n\n` +
                deployments.map(dep =>
                  `• ${dep.name}\n  Réplicas: ${dep.ready}/${dep.replicas}\n  Disponibles: ${dep.available}\n  Edad: ${dep.age}\n`
                ).join('\n'),
        },
      ],
    };
  }

  private async createDeployment(args: any) {
    const namespace = args.namespace || 'default';

    const deployment = {
      apiVersion: 'apps/v1',
      kind: 'Deployment',
      metadata: {
        name: args.name,
        namespace: namespace
      },
      spec: {
        replicas: args.replicas || 1,
        selector: {
          matchLabels: {
            app: args.name
          }
        },
        template: {
          metadata: {
            labels: {
              app: args.name
            }
          },
          spec: {
            containers: [{
              name: args.name,
              image: args.image,
              ports: args.port ? [{ containerPort: args.port }] : undefined,
              env: args.env_vars ? Object.entries(args.env_vars).map(([key, value]) => ({
                name: key,
                value: String(value)
              })) : undefined
            }]
          }
        }
      }
    };

    const response = await this.k8sAppsApi.createNamespacedDeployment(namespace, deployment);

    return {
      content: [
        {
          type: 'text',
          text: `Deployment '${args.name}' creado exitosamente en namespace '${namespace}'.\n\n` +
                `Imagen: ${args.image}\n` +
                `Réplicas: ${args.replicas || 1}\n` +
                `Estado: ${response.body.status?.conditions?.[0]?.type || 'Creando'}`,
        },
      ],
    };
  }

  private async createService(args: any) {
    const namespace = args.namespace || 'default';

    const service = {
      apiVersion: 'v1',
      kind: 'Service',
      metadata: {
        name: args.name,
        namespace: namespace
      },
      spec: {
        selector: args.selector,
        ports: [{
          port: args.port,
          targetPort: args.target_port || args.port,
          protocol: 'TCP'
        }],
        type: args.service_type || 'ClusterIP'
      }
    };

    const response = await this.k8sApi.createNamespacedService(namespace, service);

    return {
      content: [
        {
          type: 'text',
          text: `Servicio '${args.name}' creado exitosamente en namespace '${namespace}'.\n\n` +
                `Tipo: ${args.service_type || 'ClusterIP'}\n` +
                `Puerto: ${args.port}\n` +
                `IP del cluster: ${response.body.spec?.clusterIP}`,
        },
      ],
    };
  }

  private async scaleDeployment(args: any) {
    const namespace = args.namespace || 'default';

    const patch = {
      spec: {
        replicas: args.replicas
      }
    };

    await this.k8sAppsApi.patchNamespacedDeployment(
      args.name,
      namespace,
      patch,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      {
        headers: {
          'Content-Type': 'application/merge-patch+json'
        }
      }
    );

    return {
      content: [
        {
          type: 'text',
          text: `Deployment '${args.name}' escalado a ${args.replicas} réplicas en namespace '${namespace}'.`,
        },
      ],
    };
  }

  private async deleteDeployment(args: any) {
    const namespace = args.namespace || 'default';

    await this.k8sAppsApi.deleteNamespacedDeployment(args.name, namespace);

    return {
      content: [
        {
          type: 'text',
          text: `Deployment '${args.name}' eliminado del namespace '${namespace}'.`,
        },
      ],
    };
  }

  private async getPodLogs(args: any) {
    const namespace = args.namespace || 'default';
    const lines = args.lines || 100;

    const response = await this.k8sApi.readNamespacedPodLog(
      args.name,
      namespace,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      lines
    );

    return {
      content: [
        {
          type: 'text',
          text: `Logs del pod '${args.name}' (últimas ${lines} líneas):\n\n${response.body}`,
        },
      ],
    };
  }

  private async applyYaml(args: any) {
    const namespace = args.namespace || 'default';

    try {
      const manifest = yaml.parse(args.yaml_content);

      // Determinar el tipo de recurso y aplicarlo
      const kind = manifest.kind;
      const apiVersion = manifest.apiVersion;

      // Asegurar que el namespace esté configurado
      if (!manifest.metadata) manifest.metadata = {};
      if (!manifest.metadata.namespace) manifest.metadata.namespace = namespace;

      let result;

      switch (kind) {
        case 'Deployment':
          result = await this.k8sAppsApi.createNamespacedDeployment(namespace, manifest);
          break;
        case 'Service':
          result = await this.k8sApi.createNamespacedService(namespace, manifest);
          break;
        case 'Pod':
          result = await this.k8sApi.createNamespacedPod(namespace, manifest);
          break;
        default:
          throw new Error(`Tipo de recurso no soportado: ${kind}`);
      }

      return {
        content: [
          {
            type: 'text',
            text: `Manifiesto YAML aplicado exitosamente.\n\n` +
                  `Tipo: ${kind}\n` +
                  `Nombre: ${manifest.metadata.name}\n` +
                  `Namespace: ${namespace}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error aplicando manifiesto YAML: ${error}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async getClusterInfo(args: any) {
    const nodes = await this.k8sApi.listNode();
    const namespaces = await this.k8sApi.listNamespace();
    const pods = await this.k8sApi.listPodForAllNamespaces();

    const nodeInfo = nodes.body.items.map(node => ({
      name: node.metadata?.name,
      status: node.status?.conditions?.find(c => c.type === 'Ready')?.status,
      version: node.status?.nodeInfo?.kubeletVersion,
      capacity: {
        cpu: node.status?.capacity?.cpu,
        memory: node.status?.capacity?.memory,
        pods: node.status?.capacity?.pods
      }
    }));

    return {
      content: [
        {
          type: 'text',
          text: `Información del Cluster de Kubernetes:\n\n` +
                `• Nodos: ${nodes.body.items.length}\n` +
                `• Namespaces: ${namespaces.body.items.length}\n` +
                `• Pods totales: ${pods.body.items.length}\n` +
                `• Versión: ${nodeInfo[0]?.version || 'unknown'}\n\n` +
                `Nodos:\n` +
                nodeInfo.map(node =>
                  `  - ${node.name}: ${node.status} (CPU: ${node.capacity.cpu}, RAM: ${node.capacity.memory})`
                ).join('\n'),
        },
      ],
    };
  }

  private calculateAge(creationTimestamp?: string): string {
    if (!creationTimestamp) return 'unknown';

    const created = new Date(creationTimestamp);
    const now = new Date();
    const diffMs = now.getTime() - created.getTime();

    const days = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

    if (days > 0) return `${days}d`;
    if (hours > 0) return `${hours}h`;
    return `${minutes}m`;
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Servidor MCP de Kubernetes ejecutándose en stdio');
  }
}

const server = new KubernetesServer();
server.run().catch(console.error);
