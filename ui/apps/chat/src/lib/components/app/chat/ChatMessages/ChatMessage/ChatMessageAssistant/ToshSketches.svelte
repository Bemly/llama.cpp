<script lang="ts">
	import ToshSketch from './ToshSketch.svelte';
	import type { ToshSketch as Sketch } from '#lib/types';

	interface Props {
		sketches: Sketch[];
	}

	let { sketches }: Props = $props();

	// small and several to a row; one at a time enlarged above the rest
	let enlarged = $state<number | null>(null);
</script>

<div class="grid gap-3">
	{#if enlarged !== null && sketches[enlarged]}
		<ToshSketch enlarged onResize={() => (enlarged = null)} sketch={sketches[enlarged]} />
	{/if}

	<div
		class={sketches.length > 1
			? 'grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3'
			: 'grid'}
	>
		{#each sketches as sketch, index (index)}
			{#if index !== enlarged}
				<ToshSketch enlarged={false} onResize={() => (enlarged = index)} {sketch} />
			{/if}
		{/each}
	</div>
</div>
