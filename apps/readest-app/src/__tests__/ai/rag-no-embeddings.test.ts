import { beforeEach, describe, expect, test, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  embed: vi.fn(),
  embedMany: vi.fn(),
  store: {
    isIndexed: vi.fn(async () => false),
    saveChunks: vi.fn(async () => {}),
    saveBM25Index: vi.fn(async () => {}),
    saveMeta: vi.fn(async () => {}),
    hybridSearch: vi.fn(async () => []),
  },
}));

vi.mock('ai', () => ({ embed: mocks.embed, embedMany: mocks.embedMany }));
vi.mock('@/services/ai/storage/aiStore', () => ({ aiStore: mocks.store }));
vi.mock('@/services/ai/utils/chunker', () => ({
  extractTextFromDocument: () => 'x'.repeat(200),
  chunkSection: (_doc: unknown, i: number, chapterTitle: string, bookHash: string) => [
    { id: `${bookHash}-${i}`, bookHash, sectionIndex: i, chapterTitle, text: 'x', pageNumber: 1 },
  ],
}));
// A provider with no embeddings API (Claude): getEmbeddingModel must never run.
vi.mock('@/services/ai/providers', () => ({
  getAIProvider: () => ({
    id: 'anthropic',
    getEmbeddingModel: () => {
      throw new Error('no embeddings');
    },
  }),
}));

import { hybridSearch, indexBook } from '@/services/ai/ragService';
import { DEFAULT_AI_SETTINGS } from '@/services/ai/constants';

const settings = { ...DEFAULT_AI_SETTINGS, enabled: true, provider: 'anthropic' as const };
const bookDoc = {
  sections: [{ id: 's0', size: 200, linear: 'yes', createDocument: async () => ({}) as Document }],
  metadata: { title: 'T', author: 'A' },
};

describe('RAG with a provider that has no embeddings', () => {
  beforeEach(() => vi.clearAllMocks());

  test('indexes BM25-only instead of failing', async () => {
    await indexBook(bookDoc, 'hash', settings);
    expect(mocks.embedMany).not.toHaveBeenCalled();
    expect(mocks.store.saveChunks).toHaveBeenCalledTimes(1);
    expect(mocks.store.saveBM25Index).toHaveBeenCalledTimes(1);
    expect(mocks.store.saveMeta).toHaveBeenCalledWith(
      expect.objectContaining({ bookHash: 'hash', embeddingModel: 'none (keyword search)' }),
    );
  });

  test('searches with keywords only, without retrying an embedding call', async () => {
    await hybridSearch('hash', 'query', settings);
    expect(mocks.embed).not.toHaveBeenCalled();
    expect(mocks.store.hybridSearch).toHaveBeenCalledWith('hash', null, 'query', 10, undefined);
  });
});
