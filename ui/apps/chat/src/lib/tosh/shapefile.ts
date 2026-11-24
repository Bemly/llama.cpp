import type { ToshSketch } from '#lib/types';
import { notice, placed } from '#lib/tosh/sketch';

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
	let c = n;

	for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;

	return c >>> 0;
});

export function crc32(data: Uint8Array): number {
	let c = 0xffffffff;

	for (const byte of data) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);

	return (c ^ 0xffffffff) >>> 0;
}

class Bytes {
	private parts: number[] = [];

	get length(): number {
		return this.parts.length;
	}

	u8(...values: number[]): this {
		this.parts.push(...values.map((v) => v & 0xff));

		return this;
	}

	le16(value: number): this {
		return this.u8(value, value >>> 8);
	}

	le32(value: number): this {
		return this.u8(value, value >>> 8, value >>> 16, value >>> 24);
	}

	be32(value: number): this {
		return this.u8(value >>> 24, value >>> 16, value >>> 8, value);
	}

	double(value: number): this {
		const view = new DataView(new ArrayBuffer(8));

		view.setFloat64(0, value, true);

		return this.u8(...new Uint8Array(view.buffer));
	}

	bytes(data: Uint8Array | Bytes): this {
		const values = data instanceof Bytes ? data.parts : Array.from(data);

		for (const v of values) this.parts.push(v);

		return this;
	}

	array(): Uint8Array<ArrayBuffer> {
		return Uint8Array.from(this.parts);
	}
}

/** A ZIP of files stored without compression, which every unzip tool reads. */
export function storedZip(files: { name: string; data: Uint8Array }[]): Uint8Array<ArrayBuffer> {
	const out = new Bytes();
	const directory = new Bytes();

	for (const file of files) {
		const name = new TextEncoder().encode(file.name);
		const offset = out.length;
		// UTF-8 names, and 1 January 1980 as the date
		const shared = new Bytes()
			.le16(20)
			.le16(0x0800)
			.le16(0)
			.le16(0)
			.le16(0x21)
			.le32(crc32(file.data))
			.le32(file.data.length)
			.le32(file.data.length)
			.le16(name.length)
			.le16(0);

		out.le32(0x04034b50).bytes(shared).bytes(name).bytes(file.data);
		directory
			.le32(0x02014b50)
			.le16(20)
			.bytes(shared)
			.le16(0)
			.le16(0)
			.le16(0)
			.le32(0)
			.le32(offset)
			.bytes(name);
	}

	const start = out.length;

	out
		.bytes(directory)
		.le32(0x06054b50)
		.le16(0)
		.le16(0)
		.le16(files.length)
		.le16(files.length)
		.le32(directory.length)
		.le32(start)
		.le16(0);

	return out.array();
}

/** A dBase table of one record with a text field per value, cut at a character boundary to fit. */
function table(fields: [string, string][]): Uint8Array {
	const width = 254;
	const today = new Date();
	const out = new Bytes()
		.u8(0x03, today.getFullYear() - 1900, today.getMonth() + 1, today.getDate())
		.le32(1)
		.le16(32 + 32 * fields.length + 1)
		.le16(1 + width * fields.length)
		.bytes(new Uint8Array(20));

	for (const [name] of fields) {
		const encoded = new TextEncoder().encode(name).slice(0, 10);

		out
			.bytes(encoded)
			.bytes(new Uint8Array(11 - encoded.length))
			.u8('C'.charCodeAt(0), 0, 0, 0, 0, width, 0)
			.bytes(new Uint8Array(14));
	}

	out.u8(0x0d, 0x20);

	for (const [, value] of fields) {
		const kept: number[] = [];

		for (const character of value) {
			const next = new TextEncoder().encode(character);

			if (kept.length + next.length > width) break;

			kept.push(...next);
		}

		out.u8(...kept).bytes(new Uint8Array(width - kept.length).fill(0x20));
	}

	return out.u8(0x1a).array();
}

/**
 * The figure as a shapefile in a ZIP: a polygon when it should close, else a line, at its coordinates and with its
 * coordinate system as the .prj when it has them, and its title and notice as attributes.
 */
export function shapefile(sketch: ToshSketch, stem: string): Uint8Array<ArrayBuffer> {
	const path = sketch.points.map((p) => placed(sketch, p));

	if (sketch.closed) {
		const first = path[0];
		const last = path[path.length - 1];

		if (first[0] !== last[0] || first[1] !== last[1]) path.push(first);

		// an outer ring runs clockwise
		let doubled = 0;

		for (let i = 0; i + 1 < path.length; i++) {
			doubled += path[i][0] * path[i + 1][1] - path[i + 1][0] * path[i][1];
		}

		if (doubled > 0) path.reverse();
	}

	const xs = path.map((p) => p[0]);
	const ys = path.map((p) => p[1]);
	const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
	const type = sketch.closed ? 5 : 3;
	const record = new Bytes().le32(type);

	box.forEach((v) => record.double(v));
	record.le32(1).le32(path.length).le32(0);
	path.forEach((p) => record.double(p[0]).double(p[1]));

	const header = (bytes: number) => {
		const out = new Bytes().be32(9994);

		for (let i = 0; i < 5; i++) out.be32(0);

		out
			.be32(bytes / 2)
			.le32(1000)
			.le32(type);
		[...box, 0, 0, 0, 0].forEach((v) => out.double(v));

		return out;
	};
	const shp = header(100 + 8 + record.length)
		.be32(1)
		.be32(record.length / 2)
		.bytes(record);
	const shx = header(100 + 8)
		.be32(50)
		.be32(record.length / 2);
	const note = notice(sketch);
	const fields: [string, string][] = [['NAME', sketch.title ?? '']];

	if (note) fields.push(['NOTE', note]);

	const files = [
		{ name: `${stem}.shp`, data: shp.array() },
		{ name: `${stem}.shx`, data: shx.array() },
		{ name: `${stem}.dbf`, data: table(fields) },
		{ name: `${stem}.cpg`, data: new TextEncoder().encode('UTF-8') }
	];

	if (sketch.prj && sketch.origin) {
		files.push({ name: `${stem}.prj`, data: new TextEncoder().encode(sketch.prj) });
	}

	return storedZip(files);
}
