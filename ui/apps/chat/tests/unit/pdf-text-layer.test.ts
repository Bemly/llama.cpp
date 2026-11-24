import { beforeAll, describe, expect, it } from 'vitest';

beforeAll(() => {
	const store = new Map<string, string>();
	(globalThis as unknown as { localStorage: Storage }).localStorage = {
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		get length() {
			return store.size;
		},
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};
});

import { AttachmentType } from '#lib/enums';
import { ChatService } from '#lib/services/chat.service';

const pdf = (textLayer?: string) => ({
	base64Data: '',
	content: 'PDF file with 1 pages',
	images: ['data:image/jpeg;base64,AAAA'],
	name: 'deed.pdf',
	processedAsImages: true,
	textLayer,
	type: AttachmentType.PDF
});

describe('a PDF sent as page images', () => {
	it('carries its own text beside the images', async () => {
		const layer = '[Page 1]\nThence North 10°00\'00" East 100.00 feet\n[End of text layer]';
		const message = { content: 'Compute it', role: 'user', extra: [pdf(layer)] };
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const data = await ChatService.convertDbMessageToApiChatMessageData(message as any);
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const parts = data.content as any[];
		expect(parts.filter((p) => p.type === 'image_url')).toHaveLength(1);
		const text = parts.find((p) => p.type === 'text' && p.text.includes('(text layer)'));
		expect(text.text).toContain('--- PDF File: deed.pdf (text layer) ---\n[Page 1]');
		expect(text.text.endsWith('[End of text layer]')).toBe(true);
	});

	it('sends the images alone when the PDF has no text layer', async () => {
		const message = { content: 'Compute it', role: 'user', extra: [pdf()] };
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const data = await ChatService.convertDbMessageToApiChatMessageData(message as any);
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		expect((data.content as any[]).some((p) => p.type === 'text' && p.text.includes('(text layer)'))).toBe(false);
	});
});
