import type { ToshSketch } from '#lib/types';

/** How far the last point ends from the first, in the sketch's unit. */
export function gap(sketch: ToshSketch): number {
	const first = sketch.points[0];
	const last = sketch.points[sketch.points.length - 1];

	return Math.hypot(last[0] - first[0], last[1] - first[1]);
}

export function span(sketch: ToshSketch): number {
	const xs = sketch.points.map((p) => p[0]);
	const ys = sketch.points.map((p) => p[1]);

	return Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
}

export function notice(sketch: ToshSketch): string | null {
	return sketch.notice?.trim() || null;
}

export function isOpen(sketch: ToshSketch): boolean {
	return sketch.closed && gap(sketch) > span(sketch) * 1e-4;
}

/** The points to number: the corners, without the last one of a closed figure, which is its first again. */
/** A point where an export writes it: at the figure's coordinates when it has an origin, else from its first point. */
export function placed(sketch: ToshSketch, point: number[]): number[] {
	const origin = sketch.origin?.length === 2 ? sketch.origin : null;

	return origin ? [point[0] + origin[0], point[1] + origin[1]] : point;
}

export function corners(sketch: ToshSketch): number[][] {
	const all = sketch.corners ?? sketch.points;

	return sketch.closed ? all.slice(0, -1) : all;
}

/** A round length near a quarter of the figure, for the scale bar. */
export function scaleLength(sketch: ToshSketch): number {
	const target = Math.max(span(sketch) / 4, 1e-9);
	const power = 10 ** Math.floor(Math.log10(target));

	return (
		[1, 2, 5, 10]
			.map((m) => m * power)
			.filter((v) => v <= target)
			.pop() ?? power
	);
}

export function figure(value: number): string {
	return value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

/** The figure as an SVG drawing of the given size: north up, numbered corners, the gap left open in red. */
export function svg(
	sketch: ToshSketch,
	width: number,
	height: number,
	colors: { line: string; text: string }
): string {
	const marked = sketch.corners ?? sketch.points;
	const all = [...sketch.points, ...marked];
	const xs = all.map((p) => p[0]);
	const ys = all.map((p) => p[1]);
	const [minX, maxX, minY, maxY] = [
		Math.min(...xs),
		Math.max(...xs),
		Math.min(...ys),
		Math.max(...ys)
	];
	const inset = 34;
	const scale = Math.min(
		(width - 2 * inset) / Math.max(maxX - minX, 1e-9),
		(height - 2 * inset) / Math.max(maxY - minY, 1e-9)
	);
	const offsetX = (width - (maxX - minX) * scale) / 2;
	const offsetY = (height - (maxY - minY) * scale) / 2;
	const place = (p: number[]) => [
		offsetX + (p[0] - minX) * scale,
		height - offsetY - (p[1] - minY) * scale
	];
	const path = sketch.points.map(
		(p, i) => `${i ? 'L' : 'M'}${place(p)[0].toFixed(1)},${place(p)[1].toFixed(1)}`
	);
	const parts = [
		`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" font-family="system-ui, sans-serif">`,
		`<path d="${path.join('')}" fill="none" stroke="${colors.line}" stroke-width="2"/>`
	];

	if (isOpen(sketch)) {
		const [x1, y1] = place(sketch.points[sketch.points.length - 1]);
		const [x2, y2] = place(sketch.points[0]);

		parts.push(
			`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#d33" stroke-width="2" stroke-dasharray="5 4"/>`,
			`<circle cx="${x1}" cy="${y1}" r="4.5" fill="#d33"/>`
		);
	}

	const labelled: number[][] = [];

	corners(sketch).forEach((corner, index) => {
		const [x, y] = place(corner);

		parts.push(`<circle cx="${x}" cy="${y}" r="3.5" fill="${colors.line}"/>`);

		if (index === 0) {
			parts.push(
				`<circle cx="${x}" cy="${y}" r="7.5" fill="none" stroke="${colors.text}" stroke-width="1.5"/>`
			);
		}

		// a number that would cover another is left out; the figure enlarged has room for more
		if (index > 0 && labelled.some(([lx, ly]) => Math.hypot(lx - x, ly - y) < 16)) return;

		labelled.push([x, y]);
		parts.push(
			`<text x="${x + 9}" y="${y - 6}" font-size="11" font-weight="600" fill="${colors.text}">${index + 1}</text>`
		);
	});

	const length = scaleLength(sketch) * scale;

	parts.push(
		`<path d="M${width - 18},${34} L${width - 18},${16} M${width - 22},${22} L${width - 18},${16} L${width - 14},${22}" fill="none" stroke="${colors.text}" stroke-width="1.2" opacity="0.7"/>`,
		`<text x="${width - 33}" y="${24}" font-size="11" font-weight="700" fill="${colors.text}" opacity="0.7">N</text>`,
		`<line x1="14" y1="${height - 18}" x2="${14 + length}" y2="${height - 18}" stroke="${colors.text}" stroke-width="2" opacity="0.7"/>`,
		`<text x="${14 + length / 2}" y="${height - 5}" font-size="10" text-anchor="middle" fill="${colors.text}" opacity="0.7">${figure(scaleLength(sketch))} ${escape(sketch.unit)}</text>`,
		'</svg>'
	);

	return parts.join('');
}

function escape(text: string): string {
	return text.replace(
		/[<>&"]/g,
		(c) => ({ '"': '&quot;', '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c] ?? c
	);
}

function number(value: number): string {
	return value.toFixed(3);
}

/** The figure as an R12 DXF in its own unit, the first point at the origin, as ToshLLM exports it. */
export function dxf(sketch: ToshSketch): string {
	const out = [
		'0',
		'SECTION',
		'2',
		'HEADER',
		'9',
		'$ACADVER',
		'1',
		'AC1009',
		'9',
		'$INSUNITS',
		'70',
		sketch.unit.toLowerCase().startsWith('m') ? '6' : '2',
		'0',
		'ENDSEC',
		'0',
		'SECTION',
		'2',
		'ENTITIES',
		'0',
		'POLYLINE',
		'8',
		'FIGURE',
		'66',
		'1',
		'70',
		'0',
		'10',
		'0.0',
		'20',
		'0.0',
		'30',
		'0.0'
	];

	for (const p of sketch.points.map((point) => placed(sketch, point))) {
		out.push('0', 'VERTEX', '8', 'FIGURE', '10', number(p[0]), '20', number(p[1]), '30', '0.0');
	}

	out.push('0', 'SEQEND', '8', 'FIGURE');

	if (isOpen(sketch)) {
		const first = placed(sketch, sketch.points[0]);
		const last = placed(sketch, sketch.points[sketch.points.length - 1]);

		out.push(
			'0',
			'LINE',
			'8',
			'MISCLOSURE',
			'10',
			number(last[0]),
			'20',
			number(last[1]),
			'30',
			'0.0',
			'11',
			number(first[0]),
			'21',
			number(first[1]),
			'31',
			'0.0'
		);
	}

	const size = Math.max(span(sketch) / 60, 1e-3);

	corners(sketch)
		.map((corner) => placed(sketch, corner))
		.forEach((c, i) => {
			out.push(
				'0',
				'POINT',
				'8',
				'CORNERS',
				'10',
				number(c[0]),
				'20',
				number(c[1]),
				'30',
				'0.0',
				'0',
				'TEXT',
				'8',
				'LABELS',
				'10',
				number(c[0] + size / 2),
				'20',
				number(c[1] + size / 2),
				'30',
				'0.0',
				'40',
				number(size),
				'1',
				String(i + 1)
			);
		});

	const note = notice(sketch);

	if (note) {
		const placedAll = [...sketch.points, ...corners(sketch)].map((p) => placed(sketch, p));
		const left = Math.min(...placedAll.map((p) => p[0]));
		const bottom = Math.min(...placedAll.map((p) => p[1]));

		out.push(
			'0',
			'TEXT',
			'8',
			'NOTICE',
			'10',
			number(left),
			'20',
			number(bottom - 3 * size),
			'30',
			'0.0',
			'40',
			number(size),
			'1',
			note.replace(/\n/g, ' ')
		);
	}

	out.push('0', 'ENDSEC', '0', 'EOF');

	return out.join('\n') + '\n';
}

/** The numbered corners, east and north of the first one. */
export function csv(sketch: ToshSketch): string {
	const suffix = sketch.unit ? `_${sketch.unit}` : '';
	// a column rather than a comment line, which some table imports reject
	const note = notice(sketch);
	const column = note ? `,"${note.replace(/"/g, '""')}"` : '';
	const rows = corners(sketch).map((corner, i) => {
		const c = placed(sketch, corner);

		return `${i + 1},${number(c[0])},${number(c[1])}${column}`;
	});
	const header =
		(sketch.origin ? `point,x${suffix},y${suffix}` : `point,east${suffix},north${suffix}`) +
		(note ? ',note' : '');

	return [header, ...rows].join('\n') + '\n';
}

/** A PNG of the drawing on white, twice its size, as a report wants it. */
export async function png(sketch: ToshSketch): Promise<Blob | null> {
	const drawing = svg(sketch, 1200, 800, { line: '#2563eb', text: '#111' });
	const image = new Image();

	image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(drawing);
	await image.decode();
	const canvas = document.createElement('canvas');

	const note = notice(sketch);

	canvas.width = 2400;
	canvas.height = note ? 1680 : 1600;
	const context = canvas.getContext('2d');

	if (!context) return null;

	context.fillStyle = '#fff';
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.drawImage(image, 0, 0, 2400, 1600);

	if (note) {
		context.fillStyle = '#111';
		context.font = '40px system-ui, sans-serif';
		context.textAlign = 'center';
		context.fillText(note, 1200, 1640);
	}

	return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
}
