<script lang="ts">
	import Maximize2 from '@lucide/svelte/icons/maximize-2';
	import Minimize2 from '@lucide/svelte/icons/minimize-2';

	interface Props {
		enlarged: boolean;
		onclick: () => void;
	}

	let { enlarged, onclick }: Props = $props();
</script>

<button
	class="text-muted-foreground hover:text-foreground"
	{onclick}
	title={enlarged ? 'Make smaller' : 'Enlarge'}
>
	{#if enlarged}
		<Minimize2 class="h-3.5 w-3.5" />
	{:else}
		<Maximize2 class="h-3.5 w-3.5" />
	{/if}
</button>
