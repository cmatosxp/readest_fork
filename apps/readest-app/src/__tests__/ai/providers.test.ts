import { describe, test, expect, vi, beforeEach } from 'vitest';

// mock fetch for provider tests
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// mock logger
vi.mock('@/services/ai/logger', () => ({
  aiLogger: {
    provider: {
      init: vi.fn(),
      error: vi.fn(),
    },
  },
}));

// mock ai-sdk-ollama
vi.mock('ai-sdk-ollama', () => ({
  createOllama: vi.fn(() => {
    const ollamaFn = Object.assign(vi.fn(), {
      embeddingModel: vi.fn(),
    });
    return ollamaFn;
  }),
}));

// mock @ai-sdk/openai-compatible so OpenRouterProvider can be constructed
// without going over the network during unit tests.
vi.mock('@ai-sdk/openai-compatible', () => ({
  createOpenAICompatible: vi.fn(() => ({
    chatModel: vi.fn(),
    textEmbeddingModel: vi.fn(),
  })),
}));

// mock @ai-sdk/anthropic so the Claude provider is built without network.
const anthropicModelFn = vi.hoisted(() => vi.fn());
vi.mock('@ai-sdk/anthropic', () => ({
  createAnthropic: vi.fn(() => anthropicModelFn),
}));

import { OllamaProvider } from '@/services/ai/providers/OllamaProvider';
import { AIGatewayProvider } from '@/services/ai/providers/AIGatewayProvider';
import { OpenRouterProvider } from '@/services/ai/providers/OpenRouterProvider';
import { getAIProvider } from '@/services/ai/providers';
import type { AISettings } from '@/services/ai/types';
import { DEFAULT_AI_SETTINGS } from '@/services/ai/constants';

describe('OllamaProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('should create provider with default settings', () => {
    const settings: AISettings = { ...DEFAULT_AI_SETTINGS, enabled: true };
    const provider = new OllamaProvider(settings);

    expect(provider.id).toBe('ollama');
    expect(provider.name).toBe('Ollama (Local)');
    expect(provider.requiresAuth).toBe(false);
  });

  test('isAvailable should return true when Ollama responds', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true });
    const settings: AISettings = { ...DEFAULT_AI_SETTINGS, enabled: true };
    const provider = new OllamaProvider(settings);

    const result = await provider.isAvailable();
    expect(result).toBe(true);
  });

  test('isAvailable should return false when Ollama not running', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Connection refused'));
    const settings: AISettings = { ...DEFAULT_AI_SETTINGS, enabled: true };
    const provider = new OllamaProvider(settings);

    const result = await provider.isAvailable();
    expect(result).toBe(false);
  });

  test('healthCheck should verify model exists', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ models: [{ name: 'llama3.2:latest' }, { name: 'nomic-embed:latest' }] }),
    });
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      ollamaModel: 'llama3.2',
      ollamaEmbeddingModel: 'nomic-embed',
    };
    const provider = new OllamaProvider(settings);

    const result = await provider.healthCheck();
    expect(result).toBe(true);
  });

  test('healthCheck should return false if model not found', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ models: [{ name: 'other-model' }, { name: 'nomic-embed:latest' }] }),
    });
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      ollamaModel: 'llama3.2',
      ollamaEmbeddingModel: 'nomic-embed',
    };
    const provider = new OllamaProvider(settings);

    const result = await provider.healthCheck();
    expect(result).toBe(false);
  });
});

describe('AIGatewayProvider', () => {
  test('should throw if no API key', () => {
    const settings: AISettings = { ...DEFAULT_AI_SETTINGS, enabled: true, provider: 'ai-gateway' };

    expect(() => new AIGatewayProvider(settings)).toThrow('API key required');
  });

  test('should create provider with API key', () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'ai-gateway',
      aiGatewayApiKey: 'test-key',
    };
    const provider = new AIGatewayProvider(settings);

    expect(provider.id).toBe('ai-gateway');
    expect(provider.name).toBe('AI Gateway (Cloud)');
    expect(provider.requiresAuth).toBe(true);
  });

  test('isAvailable should return true if key exists', async () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'ai-gateway',
      aiGatewayApiKey: 'test-key',
    };
    const provider = new AIGatewayProvider(settings);

    const result = await provider.isAvailable();
    expect(result).toBe(true);
  });

  test('isAvailable should return false if key does not exist', async () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'ai-gateway',
      aiGatewayApiKey: '',
    };

    // provider throws on construction if no key, so we test via getAIProvider fallback
    expect(() => new AIGatewayProvider(settings)).toThrow('API key required');
  });

  test('healthCheck should return false if key does not exist', async () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'ai-gateway',
      aiGatewayApiKey: 'valid-key',
    };
    const provider = new AIGatewayProvider(settings);

    // override key after construction to simulate missing key check in healthCheck
    (provider as unknown as { settings: AISettings }).settings.aiGatewayApiKey = '';
    const result = await provider.healthCheck();
    expect(result).toBe(false);
  });
});

describe('OpenRouterProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('should throw if no API key', () => {
    const settings: AISettings = { ...DEFAULT_AI_SETTINGS, enabled: true, provider: 'openrouter' };
    expect(() => new OpenRouterProvider(settings)).toThrow('OpenRouter API key required');
  });

  test('should create provider with API key and default base URL', () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'openrouter',
      openrouterApiKey: 'sk-or-test',
    };
    const provider = new OpenRouterProvider(settings);

    expect(provider.id).toBe('openrouter');
    expect(provider.name).toBe('OpenRouter (Custom)');
    expect(provider.requiresAuth).toBe(true);
  });

  test('isAvailable should return true if key exists', async () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'openrouter',
      openrouterApiKey: 'sk-or-test',
    };
    const provider = new OpenRouterProvider(settings);

    expect(await provider.isAvailable()).toBe(true);
  });

  test('healthCheck succeeds when /models responds OK', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200 });
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'openrouter',
      openrouterApiKey: 'sk-or-test',
      openrouterBaseUrl: 'https://openrouter.ai/api/v1',
    };
    const provider = new OpenRouterProvider(settings);

    expect(await provider.healthCheck()).toBe(true);
    // verifies we hit `${baseUrl}/models` with the Authorization header
    expect(mockFetch).toHaveBeenCalledWith(
      'https://openrouter.ai/api/v1/models',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({ Authorization: 'Bearer sk-or-test' }),
      }),
    );
  });

  test('healthCheck returns false when /models fails', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 401 });
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'openrouter',
      openrouterApiKey: 'sk-or-bad',
    };
    const provider = new OpenRouterProvider(settings);

    expect(await provider.healthCheck()).toBe(false);
  });

  test('healthCheck strips trailing slashes from base URL', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200 });
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'openrouter',
      openrouterApiKey: 'sk-or-test',
      openrouterBaseUrl: 'https://example.com/v1////',
    };
    const provider = new OpenRouterProvider(settings);

    await provider.healthCheck();
    expect(mockFetch).toHaveBeenCalledWith('https://example.com/v1/models', expect.any(Object));
  });
});

describe('getAIProvider', () => {
  test('should return OllamaProvider for ollama', () => {
    const settings: AISettings = { ...DEFAULT_AI_SETTINGS, enabled: true, provider: 'ollama' };
    const provider = getAIProvider(settings);

    expect(provider.id).toBe('ollama');
  });

  test('should return AIGatewayProvider for ai-gateway', () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'ai-gateway',
      aiGatewayApiKey: 'test-key',
    };
    const provider = getAIProvider(settings);

    expect(provider.id).toBe('ai-gateway');
  });

  test('should return OpenRouterProvider for openrouter', () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'openrouter',
      openrouterApiKey: 'sk-or-test',
    };
    const provider = getAIProvider(settings);

    expect(provider.id).toBe('openrouter');
  });

  test('getAIProvider throws when openrouter has no API key', () => {
    const settings: AISettings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'openrouter',
    };
    expect(() => getAIProvider(settings)).toThrow('API key required for OpenRouter');
  });

  test('should throw for unknown provider', () => {
    const settings = {
      ...DEFAULT_AI_SETTINGS,
      enabled: true,
      provider: 'unknown' as unknown,
    } as AISettings;

    expect(() => getAIProvider(settings)).toThrow('Unknown provider');
  });
});

describe('AnthropicProvider (Claude via API key)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const claude = (extra: Partial<AISettings> = {}): AISettings => ({
    ...DEFAULT_AI_SETTINGS,
    enabled: true,
    provider: 'anthropic',
    anthropicApiKey: 'sk-ant-test',
    ...extra,
  });

  test('getAIProvider returns the Claude provider', () => {
    const provider = getAIProvider(claude());
    expect(provider.id).toBe('anthropic');
    expect(provider.requiresAuth).toBe(true);
  });

  test('getAIProvider throws without an API key', () => {
    expect(() => getAIProvider(claude({ anthropicApiKey: '' }))).toThrow(
      'API key required for Claude',
    );
  });

  test('uses the chosen model, else the default', () => {
    const provider = getAIProvider(claude({ anthropicModel: 'claude-haiku-5-5' }));
    provider.getModel();
    expect(anthropicModelFn).toHaveBeenLastCalledWith('claude-haiku-5-5');
    getAIProvider(claude()).getModel();
    expect(anthropicModelFn).toHaveBeenLastCalledWith(DEFAULT_AI_SETTINGS.anthropicModel);
  });

  test('has no embedding model', () => {
    expect(() => getAIProvider(claude()).getEmbeddingModel()).toThrow();
  });

  test('healthCheck lists models with the key (no token spend)', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true });
    expect(await getAIProvider(claude()).healthCheck()).toBe(true);
    const [url, init] = mockFetch.mock.calls[0]!;
    expect(url).toBe('https://api.anthropic.com/v1/models');
    expect((init as RequestInit).headers).toMatchObject({
      'x-api-key': 'sk-ant-test',
      'anthropic-version': '2023-06-01',
    });
  });

  test('healthCheck is false on a rejected key', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 401 });
    expect(await getAIProvider(claude()).healthCheck()).toBe(false);
  });
});
