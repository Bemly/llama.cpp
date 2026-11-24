export function download(name: string, data: BlobPart, type: string): void {
	const url = URL.createObjectURL(new Blob([data], { type }));
	const link = document.createElement('a');

	link.href = url;
	link.download = name;
	document.body.appendChild(link);
	link.click();
	link.remove();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A name for a file from a title, without characters a file system refuses. */
export function fileStem(title: string | undefined, fallback: string): string {
	const cleaned = (title ?? '').replace(/[/:\\]/g, '-').trim();

	return (cleaned || fallback).slice(0, 80);
}
