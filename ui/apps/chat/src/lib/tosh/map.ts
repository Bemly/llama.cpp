import { fileStem } from './download';
import { API_CHAT } from '#lib/constants';
import type { ToshMapLayer, ToshMapRequest } from '#lib/types';
import type { LngLatBoundsLike, StyleSpecification } from 'maplibre-gl';
import { getJsonHeaders } from '#lib/utils/api-headers';
import { strToU8, zipSync } from 'fflate';

/** The features of an answer's map, fetched from the agent with the calls it kept; the model is not involved. */
export async function fetchMap(
	request: ToshMapRequest,
	question: string
): Promise<ToshMapLayer[] | null> {
	try {
		const response = await fetch(API_CHAT.COMPLETIONS, {
			body: JSON.stringify({
				messages: [{ content: question, role: 'user' }],
				stream: false,
				tosh: { map_request: request }
			}),
			headers: { ...getJsonHeaders(), 'X-Tosh-Agent': 'on' },
			method: 'POST'
		});

		if (!response.ok) return null;

		const layers = (await response.json())?.tosh?.map?.layers;

		return Array.isArray(layers) && layers.length ? layers : null;
	} catch {
		return null;
	}
}

const KINDS: [string, string[]][] = [
	['polygons', ['Polygon', 'MultiPolygon']],
	['lines', ['LineString', 'MultiLineString']],
	['points', ['Point']],
	['multipoints', ['MultiPoint']]
];

/** One GeoJSON file per layer and kind of geometry, as GIS import tools take one kind per run. */
export function geojsonFiles(layers: ToshMapLayer[]): [string, string][] {
	const files: [string, string][] = [];
	const taken = new Set<string>();

	layers.forEach((layer, index) => {
		const title = layer.title ?? 'map';
		const features = (layer.features ?? []) as {
			geometry?: { type?: string };
			properties?: object;
		}[];
		const groups = KINDS.map(([, types]) =>
			features.filter((f) => types.includes(f.geometry?.type ?? ''))
		);
		const used = groups.filter((g) => g.length).length;

		groups.forEach((group, k) => {
			if (!group.length) return;

			const collection = {
				features: group.map((f) => ({
					...f,
					properties: { ...(f.properties ?? {}), layer: title }
				})),
				type: 'FeatureCollection'
			};

			let stem = fileStem(title, 'map') + (used > 1 ? ` - ${KINDS[k][0]}` : '');

			if (taken.has(stem)) stem += ` ${index + 1}`;

			taken.add(stem);
			files.push([`${stem}.geojson`, JSON.stringify(collection, null, 2)]);
		});
	});

	return files;
}

/** The files as one download: the file itself when there is one, else a ZIP of them all. */
export function bundle(
	files: [string, string][]
): { name: string; data: BlobPart; type: string } | null {
	if (!files.length) return null;

	if (files.length === 1)
		return { data: files[0][1], name: files[0][0], type: 'application/geo+json' };

	const zipped = zipSync(Object.fromEntries(files.map(([name, text]) => [name, strToU8(text)])));

	return { data: zipped, name: 'map.zip', type: 'application/zip' };
}

const OPEN_FREE_MAP = 'https://tiles.openfreemap.org/styles/liberty';
const PROTOMAPS_ATTRIBUTION =
	'<a href="https://protomaps.com">Protomaps</a> © <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>';

let pmtilesProtocol = false;

/** A plain ground, for a map with no base map or one that could not load. */
export function ground(dark: boolean): StyleSpecification {
	return {
		layers: [
			{
				id: 'ground',
				paint: { 'background-color': dark ? '#1f2937' : '#f3f4f6' },
				type: 'background'
			}
		],
		sources: {},
		version: 8
	};
}

/**
 * The base map the server names in its ui settings: its PMTiles regions, drawn with the Protomaps style and the
 * fonts and sprites next to them, which need no internet (without labels when mapLabels is false, for a folder that
 * lacks them); a style of its own, or none for a plain ground; or OpenFreeMap online. An entry ending in .json is
 * an index the server keeps, {"regions": [files next to it], "labels": bool}, read each time a map opens so regions
 * added while it runs show without a restart.
 */
export async function baseStyle(
	maplibregl: typeof import('maplibre-gl'),
	settings: Record<string, string | number | boolean>,
	dark: boolean
): Promise<string | StyleSpecification> {
	const listed = String(settings.mapRegions ?? '')
		.split(',')
		.map((r) => r.trim())
		.filter(Boolean);
	const regions: string[] = [];
	let labels = settings.mapLabels !== false;

	for (const entry of listed) {
		const url = new URL(entry, location.href);

		if (!/\.json$/i.test(url.pathname)) {
			regions.push(url.href);
			continue;
		}

		try {
			const index = await (await fetch(url, { cache: 'no-store' })).json();

			for (const file of index.regions ?? []) regions.push(new URL(String(file), url).href);
			if (index.labels === false) labels = false;
		} catch {
			// an index that cannot be read adds no region
		}
	}

	if (!regions.length) {
		if (settings.mapStyle === 'none') return ground(dark);

		return String(settings.mapStyle || OPEN_FREE_MAP);
	}

	const { Protocol } = await import('pmtiles');
	const { layers, namedFlavor } = await import('@protomaps/basemaps');

	if (!pmtilesProtocol) {
		maplibregl.addProtocol('pmtiles', new Protocol().tile);
		pmtilesProtocol = true;
	}

	const assets = new URL(String(settings.mapAssets || '/tiles/assets'), location.href).href.replace(
		/\/$/,
		''
	);
	const flavor = dark ? 'dark' : 'light';
	const lang = (navigator.language || 'en').slice(0, 2);
	const sources: StyleSpecification['sources'] = {};
	const drawn: StyleSpecification['layers'] = [];

	regions.forEach((region, index) => {
		const name = `region${index}`;

		sources[name] = {
			attribution: PROTOMAPS_ATTRIBUTION,
			type: 'vector',
			url: 'pmtiles://' + region
		};
		// one ground for all the regions, under every one of them
		for (const layer of layers(name, namedFlavor(flavor), { lang })) {
			if (index > 0 && layer.type === 'background') continue;
			if (!labels && layer.type === 'symbol') continue;
			drawn.push({ ...layer, id: `${name}-${layer.id}` } as StyleSpecification['layers'][number]);
		}
	});

	if (!labels) return { layers: drawn, sources, version: 8 };

	return {
		glyphs: `${assets}/fonts/{fontstack}/{range}.pbf`,
		layers: drawn,
		sources,
		sprite: `${assets}/sprites/v4/${flavor}`,
		version: 8
	};
}

/** The box around every feature of the layers, or null when they hold none. */
export function bounds(layers: ToshMapLayer[]): LngLatBoundsLike | null {
	let west = Infinity,
		south = Infinity,
		east = -Infinity,
		north = -Infinity;

	const visit = (value: unknown) => {
		if (!Array.isArray(value)) return;

		if (typeof value[0] === 'number' && typeof value[1] === 'number') {
			west = Math.min(west, value[0]);
			east = Math.max(east, value[0]);
			south = Math.min(south, value[1]);
			north = Math.max(north, value[1]);
		} else value.forEach(visit);
	};

	for (const layer of layers) {
		for (const feature of (layer.features ?? []) as { geometry?: { coordinates?: unknown } }[]) {
			visit(feature.geometry?.coordinates);
		}
	}

	return west === Infinity
		? null
		: [
				[west, south],
				[east, north]
			];
}
