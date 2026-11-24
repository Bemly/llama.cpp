<script lang="ts">
	import ToshResize from './ToshResize.svelte';
	import { download, fileStem } from '#lib/tosh/download';
	import { csv, dxf, figure, gap, isOpen, notice, png, svg } from '#lib/tosh/sketch';
	import { shapefile } from '#lib/tosh/shapefile';
	import type { ToshSketch } from '#lib/types';

	interface Props {
		sketch: ToshSketch;
		enlarged: boolean;
		onResize: () => void;
	}

	let { enlarged, onResize, sketch }: Props = $props();

	// drawn at about the size it shows, so its numbers keep their size
	let drawing = $derived(
		svg(sketch, enlarged ? 640 : 320, enlarged ? 360 : 220, {
			line: '#3b82f6',
			text: 'currentColor'
		})
	);
	let stem = $derived(fileStem(sketch.title, 'sketch'));

	async function savePng() {
		const blob = await png(sketch);

		if (blob) download(`${stem}.png`, blob, 'image/png');
	}
</script>

<div class="grid gap-1.5">
	<div class="flex items-center justify-between gap-2">
		<div class="text-xs font-medium {enlarged ? '' : 'truncate'}">{sketch.title ?? ''}</div>

		<ToshResize {enlarged} onclick={onResize} />
	</div>

	<div
		class="overflow-hidden rounded-lg bg-muted/40 text-foreground [&_svg]:h-full [&_svg]:w-full {enlarged
			? 'h-96'
			: 'h-44'}"
		title="Sketch from the answer's data: point 1 is the start and each leg runs to the next number."
	>
		<!-- eslint-disable-next-line svelte/no-at-html-tags -->
		{@html drawing}
	</div>

	<div class="flex flex-wrap items-start justify-between gap-2 text-[11px] text-muted-foreground">
		<span>
			{sketch.plane && sketch.origin
				? `Sketch in ${sketch.plane}`
				: 'Sketch, not placed on a map'}{#if isOpen(sketch)}{` · in red, what it lacks to close: ${figure(
					gap(sketch)
				)} ${sketch.unit}`}{/if}{#if notice(sketch)}{` · ${notice(sketch)}`}{/if}
		</span>

		<span class="flex gap-3">
			<button class="hover:text-foreground" onclick={savePng}>PNG</button>

			<button
				class="hover:text-foreground"
				onclick={() => download(`${stem}.dxf`, dxf(sketch), 'application/dxf')}
				title="For CAD and GIS applications, from point 1">DXF</button
			>

			<button
				class="hover:text-foreground"
				onclick={() => download(`${stem}.csv`, csv(sketch), 'text/csv')}>Corners CSV</button
			>

			<button
				class="hover:text-foreground"
				onclick={() => download(`${stem}.zip`, shapefile(sketch, stem), 'application/zip')}
				title="Shapefile in a ZIP, with its coordinate system when the figure has one"
				>Shapefile</button
			>
		</span>
	</div>
</div>
