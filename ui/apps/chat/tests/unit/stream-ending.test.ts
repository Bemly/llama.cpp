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

import { ChatService } from '#lib/services/chat.service';
import type { ChatMessageTimings } from '#lib/types/chat';

function streamOf(lines: string[]): Response {
	const encoder = new TextEncoder();
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			for (const line of lines) controller.enqueue(encoder.encode(line + '\n\n'));
			controller.close();
		}
	});
	return new Response(body, { status: 200 });
}

async function read(lines: string[], maxTokens?: number) {
	const seen = {
		content: '',
		completed: undefined as undefined | { text: string; timings?: ChatMessageTimings },
		errors: [] as Error[]
	};
	await ChatService.handleStreamResponse(
		streamOf(lines),
		(chunk) => (seen.content += chunk),
		(text, _reasoning, timings) => (seen.completed = { text, timings }),
		(error) => seen.errors.push(error),
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		maxTokens
	);
	return seen;
}

const chunk = (value: unknown) => 'data: ' + JSON.stringify(value);

describe('the end of a stream', () => {
	it('is an error when it stops with nothing to resume from, never an answer left loading', async () => {
		const seen = await read([chunk({ choices: [{ delta: { content: 'Half an' } }] })]);
		expect(seen.completed).toBeUndefined();
		expect(seen.errors).toHaveLength(1);
		expect(seen.content).toContain('so it is incomplete._');
	});

	it("takes an agent's count of the context for its meter", async () => {
		const seen = await read([
			chunk({ choices: [{ delta: { content: 'Done.' } }] }),
			chunk({
				choices: [{ delta: {}, finish_reason: 'stop' }],
				usage: { prompt_tokens: 9000, completion_tokens: 40 },
				tosh: { version: 1, outcome: 'answered', context: { used: 3120, n_ctx: 8192, left_out: 0 } }
			}),
			'data: [DONE]'
		]);
		expect(seen.completed?.timings).toEqual({ prompt_n: 3120, cache_n: 0, predicted_n: 0 });
	});

	it('says whether the context or the limit of tokens cut an answer', async () => {
		const cut = (predicted: number) => [
			chunk({ choices: [{ delta: { content: 'Once upon' } }] }),
			chunk({
				choices: [{ delta: {}, finish_reason: 'length' }],
				timings: { prompt_n: 100, cache_n: 0, predicted_n: predicted }
			}),
			'data: [DONE]'
		];
		expect((await read(cut(300), 1024)).completed?.text).toContain('filled the model\'s context');
		expect((await read(cut(1024), 1024)).completed?.text).toContain('limit of tokens per answer');
		expect((await read(cut(300))).completed?.text).toContain('limit of tokens per answer');
	});
});
