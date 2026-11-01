import { createAnthropic } from '@ai-sdk/anthropic';
import type { LanguageModel, EmbeddingModel } from 'ai';
import type { AIProvider, AISettings, AIProviderName } from '../types';
import { DEFAULT_AI_SETTINGS } from '../constants';
import { aiLogger } from '../logger';
import { AI_TIMEOUTS } from '../utils/retry';
import { getAIFetch } from '../utils/httpFetch';

const API_BASE = 'https://api.anthropic.com/v1';
const API_VERSION = '2023-06-01';

const headersFor = (apiKey: string): Record<string, string> => ({
  'x-api-key': apiKey,
  'anthropic-version': API_VERSION,
  // Only matters on the web build (no Tauri HTTP transport): lets the browser
  // call the API directly with the reader's own key.
  'anthropic-dangerous-direct-browser-access': 'true',
});

/**
 * Claude through the Anthropic Messages API with the reader's own API key
 * (console.anthropic.com, billed per token). This is not a Claude.ai
 * subscription login. Requests go through {@link getAIFetch} so the Tauri app
 * sends them from Rust (no CORS).
 *
 * Anthropic has no embeddings API; see `providerHasEmbeddings`.
 */
export class AnthropicProvider implements AIProvider {
  id: AIProviderName = 'anthropic';
  name = 'Claude (Anthropic API)';
  requiresAuth = true;

  private settings: AISettings;
  private apiKey: string;
  private client: ReturnType<typeof createAnthropic>;
  private httpFetch: typeof fetch;

  constructor(settings: AISettings) {
    if (!settings.anthropicApiKey) {
      throw new Error('Anthropic API key required');
    }
    this.settings = settings;
    this.apiKey = settings.anthropicApiKey;
    this.httpFetch = getAIFetch();
    this.client = createAnthropic({
      apiKey: this.apiKey,
      headers: { 'anthropic-dangerous-direct-browser-access': 'true' },
      fetch: this.httpFetch,
    });
    aiLogger.provider.init('anthropic', this.modelId());
  }

  private modelId(): string {
    return this.settings.anthropicModel || DEFAULT_AI_SETTINGS.anthropicModel!;
  }

  getModel(): LanguageModel {
    return this.client(this.modelId());
  }

  getEmbeddingModel(): EmbeddingModel {
    throw new Error('Claude has no embeddings API');
  }

  async isAvailable(): Promise<boolean> {
    return !!this.apiKey;
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.httpFetch(`${API_BASE}/models`, {
        method: 'GET',
        headers: headersFor(this.apiKey),
        signal: AbortSignal.timeout(AI_TIMEOUTS.HEALTH_CHECK),
      });
      if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
      return true;
    } catch (e) {
      aiLogger.provider.error('anthropic', `healthCheck failed: ${(e as Error).message}`);
      return false;
    }
  }
}

export interface AnthropicModelInfo {
  id: string;
  display_name?: string;
}

/** The models this key can use, newest first (GET /v1/models). */
export async function fetchAnthropicModels(
  apiKey: string,
  signal?: AbortSignal,
): Promise<AnthropicModelInfo[]> {
  const response = await getAIFetch()(`${API_BASE}/models?limit=100`, {
    method: 'GET',
    headers: headersFor(apiKey),
    signal,
  });
  if (!response.ok) throw new Error(`Failed to fetch models: ${response.status}`);
  const json = (await response.json()) as { data?: AnthropicModelInfo[] };
  return Array.isArray(json.data) ? json.data : [];
}
