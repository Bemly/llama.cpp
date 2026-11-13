/** A call the server's Tosh agent made while it answered, as its progress events report it. */
export interface ToshCall {
	id?: string;
	tool?: string;
	arguments?: unknown;
	status?: string;
	state?: string;
	result?: string;
	summary?: string;
}

/** A figure the agent attached to an answer in a plane of its own, such as a tract from a deed's calls. */
export interface ToshSketch {
	title?: string;
	unit: string;
	points: number[][];
	corners?: number[][];
	closed: boolean;
	/** Where the first point lies in the coordinate system `plane` names: exports put the figure there. */
	origin?: number[];
	plane?: string;
	/** A line every export carries, such as a disclaimer that the figure is not a survey. */
	notice?: string;
	/** The coordinate system of `origin` as a shapefile's .prj holds it. */
	prj?: string;
}

export interface ToshMapRequest {
	calls: unknown[];
	show?: boolean;
}

export interface ToshMapLayer {
	title?: string;
	type?: string;
	features: unknown[];
	link?: { title?: string; url?: string };
	complete?: boolean;
	subtitle?: string;
	fields?: { name: string; alias?: string }[];
	drawn?: boolean;
}

/** The `tosh` object of a chunk: a progress event while the agent works, or the answer's metadata at the end. */
export interface ToshChunk {
	version?: number;
	event?: { type: string; call?: ToshCall; [key: string]: unknown };
	sketches?: ToshSketch[];
	map_request?: ToshMapRequest;
	map?: { layers: ToshMapLayer[] };
	/** How full the context is after the turn, as the engine counted it. */
	context?: { used?: number | null; n_ctx?: number | null; left_out?: number };
	[key: string]: unknown;
}

/** What an assistant message keeps of the agent's work. */
export interface ToshMessageData {
	steps?: ToshCall[];
	/** What the agent is doing while it works, as it says it; gone once it answers. */
	progress?: string;
	sketches?: ToshSketch[];
	mapRequest?: ToshMapRequest;
	map?: { layers: ToshMapLayer[] };
}
