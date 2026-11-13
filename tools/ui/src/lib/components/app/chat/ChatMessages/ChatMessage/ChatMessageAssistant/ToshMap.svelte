<script lang="ts">
	import 'maplibre-gl/dist/maplibre-gl.css';
	import ToshResize from './ToshResize.svelte';
	import { download } from '$lib/tosh/download';
	import { baseStyle, bounds, bundle, geojsonFiles, ground } from '$lib/tosh/map';
	import { serverStore } from '$lib/stores/server.svelte';
	import type { ToshMapLayer } from '$lib/types';
	import type { Map as MapLibreMap } from 'maplibre-gl';
	import { onDestroy, onMount, tick } from 'svelte';

	interface Props {
		layers: ToshMapLayer[];
	}

	let { layers }: Props = $props();

	const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

	let element: HTMLDivElement;
	let map: MapLibreMap | null = null;
	let destroyed = false;
	let enlarged = $state(false);

	async function resize() {
		enlarged = !enlarged;
		await tick();
		map?.resize();
	}

	function escape(value: unknown): string {
		return String(value ?? '').replace(
			/[<>&"]/g,
			(c) => ({ '"': '&quot;', '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c] ?? c
		);
	}

	/** The attributes of a feature in the layer's own order, with the names a GIS shows for them. */
	function popup(properties: Record<string, unknown>, layer: ToshMapLayer): string {
		const names = layer.fields?.length
			? layer.fields
			: Object.keys(properties).map((name) => ({ alias: name, name }));
		const rows = names
			.filter(
				(f) =>
					properties[f.name] !== null &&
					properties[f.name] !== undefined &&
					properties[f.name] !== ''
			)
			.slice(0, 40)
			.map(
				(f) =>
					`<tr><th style="text-align:left;padding-right:8px">${escape(f.alias ?? f.name)}</th><td>${escape(properties[f.name])}</td></tr>`
			);

		return `<div style="max-height:220px;overflow:auto;font-size:12px;color:#111"><b>${escape(layer.title)}</b><table>${rows.join('')}</table></div>`;
	}

	// MapLibre touches window when imported, so it loads in the browser only.
	onMount(async () => {
		const maplibregl = await import('maplibre-gl');
		// the worker comes out of the build as a file of its own, which MapLibre cannot find on its own
		const worker = (await import('maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url')).default;

		maplibregl.setWorkerUrl(worker);
		const dark = document.documentElement.classList.contains('dark');
		const style = await baseStyle(maplibregl, serverStore.uiSettings ?? {}, dark);

		if (destroyed) return;

		const created = new maplibregl.Map({
			attributionControl: { compact: true },
			container: element,
			scrollZoom: false,
			style
		});
		map = created;
		created.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

		// without internet and without a region the base map cannot load: the layers still show on a plain ground
		let fellBack = false;

		created.on('error', () => {
			if (fellBack || created.isStyleLoaded()) return;

			fellBack = true;
			created.setStyle(ground(dark));
		});

		created.on('style.load', () => {
			layers.forEach((layer, index) => {
				const id = `tosh-${index}`;
				const color = COLORS[index % COLORS.length];
				const polygons = ['in', ['geometry-type'], ['literal', ['Polygon', 'MultiPolygon']]];
				const lines = [
					'in',
					['geometry-type'],
					['literal', ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']]
				];
				const points = ['in', ['geometry-type'], ['literal', ['Point', 'MultiPoint']]];

				if (created.getSource(id)) return;

				created.addSource(id, {
					data: {
						features: layer.features,
						type: 'FeatureCollection'
					} as GeoJSON.FeatureCollection,
					type: 'geojson'
				});
				created.addLayer({
					filter: polygons as never,
					id: `${id}-fill`,
					paint: { 'fill-color': color, 'fill-opacity': 0.15 },
					source: id,
					type: 'fill'
				});
				created.addLayer({
					filter: lines as never,
					id: `${id}-line`,
					paint: {
						'line-color': color,
						'line-width': 2,
						...(layer.drawn ? { 'line-dasharray': [3, 2] } : {})
					},
					source: id,
					type: 'line'
				});
				created.addLayer({
					filter: points as never,
					id: `${id}-point`,
					paint: {
						'circle-color': color,
						'circle-opacity': 0.85,
						'circle-radius': 5,
						'circle-stroke-color': '#ffffff',
						'circle-stroke-width': 1
					},
					source: id,
					type: 'circle'
				});

				for (const part of ['fill', 'line', 'point']) {
					created.on('click', `${id}-${part}`, (event) => {
						const feature = event.features?.[0];

						if (!feature) return;

						new maplibregl.Popup({ maxWidth: '320px' })
							.setLngLat(event.lngLat)
							.setHTML(popup((feature.properties ?? {}) as Record<string, unknown>, layer))
							.addTo(created);
					});
					created.on(
						'mouseenter',
						`${id}-${part}`,
						() => (created.getCanvas().style.cursor = 'pointer')
					);
					created.on('mouseleave', `${id}-${part}`, () => (created.getCanvas().style.cursor = ''));
				}
			});
		});

		const box = bounds(layers);

		if (box) created.fitBounds(box, { duration: 0, maxZoom: 17, padding: 20 });
		else created.jumpTo({ center: [0, 0], zoom: 1 });
	});

	onDestroy(() => {
		destroyed = true;
		map?.remove();
	});

	function exportGeojson() {
		const file = bundle(geojsonFiles(layers));

		if (file) download(file.name, file.data, file.type);
	}
</script>

<div class="grid gap-1.5">
	<div class="w-full overflow-hidden rounded-lg {enlarged ? 'h-[460px]' : 'h-56'}">
		<div bind:this={element} class="h-full w-full"></div>
	</div>

	<div class="flex flex-wrap items-start justify-between gap-2 text-[11px] text-muted-foreground">
		<span class="flex flex-wrap gap-3">
			{#each layers as layer, index (index)}
				<span class="inline-flex items-center gap-1">
					<span
						class="inline-block h-2 w-2 rounded-full"
						style="background:{COLORS[index % COLORS.length]}"
					></span>

					{#if layer.link?.url && /^https?:\/\//.test(layer.link.url)}
						<a
							class="underline-offset-2 hover:underline"
							href={layer.link.url}
							rel="noopener noreferrer"
							target="_blank">{layer.title ?? layer.link.title ?? 'Layer'}</a
						>
					{:else}
						{layer.title ?? 'Layer'}
					{/if}{layer.complete === false ? ' (partial)' : ''}{layer.drawn ? ' (approximate)' : ''}
					{#if layer.subtitle}<span class="opacity-75">{layer.subtitle}</span>{/if}
				</span>
			{/each}
		</span>

		<span class="flex items-center gap-3">
			<ToshResize {enlarged} onclick={resize} />

			<button
				class="hover:text-foreground"
				onclick={exportGeojson}
				title="One .geojson file per layer and kind of geometry, as GIS import tools take them"
				>GeoJSON</button
			>
		</span>
	</div>
</div>
