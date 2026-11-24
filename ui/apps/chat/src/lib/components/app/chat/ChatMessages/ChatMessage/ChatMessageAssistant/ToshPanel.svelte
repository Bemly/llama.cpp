<script lang="ts">
	import ToshMap from './ToshMap.svelte';
	import ToshSketches from './ToshSketches.svelte';
	import Waypoints from '@lucide/svelte/icons/waypoints';
	import { CollapsibleContentBlock } from '#lib/components/app';
	import { DatabaseService } from '#lib/services';
	import { conversationsStore } from '#lib/stores';
	import { fetchMap } from '#lib/tosh/map';
	import type { DatabaseMessage, ToshCall, ToshMapLayer } from '#lib/types';

	interface Props {
		message: DatabaseMessage;
	}

	let { message }: Props = $props();

	let stepsOpen = $state(false);
	let loading = $state(false);
	let failed = $state(false);

	let tosh = $derived(message.tosh);
	let steps = $derived(tosh?.steps ?? []);

	// an agent reports a call's state, or only its status
	function stepFailed(step: ToshCall): boolean {
		return step.state ? step.state !== 'completed' : !!step.status && step.status !== 'ok';
	}

	function question(): string {
		const parent = conversationsStore.activeMessages.find((m) => m.id === message.parent);

		return parent?.content ?? '';
	}

	async function loadMap() {
		if (!tosh?.mapRequest || loading) return;

		loading = true;
		failed = false;
		const layers: ToshMapLayer[] | null = await fetchMap(
			$state.snapshot(tosh.mapRequest),
			question()
		);

		loading = false;

		if (!layers) {
			failed = true;

			return;
		}

		const updated = { ...$state.snapshot(tosh), map: { layers } };

		conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(message.id), {
			tosh: updated
		});
		await DatabaseService.updateMessage(message.id, { tosh: updated });
	}

	$effect(() => {
		if (tosh?.mapRequest?.show && !tosh.map && !loading && !failed) loadMap();
	});
</script>

{#if tosh}
	<div class="mt-3 grid gap-3">
		{#if steps.length}
			<CollapsibleContentBlock
				icon={Waypoints}
				onToggle={() => (stepsOpen = !stepsOpen)}
				open={stepsOpen}
				subtitle={steps[steps.length - 1]?.summary ?? ''}
				title={`Agent steps (${steps.length})`}
			>
				<ol class="grid gap-1.5 text-xs text-muted-foreground">
					{#each steps as step, index (step.id ?? index)}
						<li class="flex gap-2">
							<span class="tabular-nums">{index + 1}.</span>

							<span class={stepFailed(step) ? 'text-destructive' : ''}>
								{step.summary || step.tool}
							</span>
						</li>
					{/each}
				</ol>
			</CollapsibleContentBlock>
		{/if}

		{#if tosh.progress}
			<p class="animate-pulse text-xs text-muted-foreground">{tosh.progress}…</p>
		{/if}

		{#if tosh.sketches?.length}
			<ToshSketches sketches={tosh.sketches} />
		{/if}

		{#if tosh.map}
			<ToshMap layers={tosh.map.layers} />
		{:else if tosh.mapRequest}
			<button
				class="w-fit text-xs text-primary underline-offset-2 hover:underline disabled:opacity-50"
				disabled={loading}
				onclick={loadMap}
			>
				{loading
					? 'Loading the map…'
					: failed
						? 'The map could not be loaded. Retry'
						: 'Show on a map'}
			</button>
		{/if}
	</div>
{/if}
