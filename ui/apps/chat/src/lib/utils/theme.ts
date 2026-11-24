import { browser } from '$app/env';
import { ColorMode } from '#lib/enums/ui.enums';
import { setMode } from 'mode-watcher';

const PALETTE_MODES = {
	forest: ColorMode.DARK,
	lavender: ColorMode.LIGHT,
	ocean: ColorMode.DARK,
	sepia: ColorMode.LIGHT
} as const;

export function applyTheme(theme: unknown): void {
	if (!browser) return;

	if (typeof theme === 'string' && Object.hasOwn(PALETTE_MODES, theme)) {
		document.documentElement.dataset.colorTheme = theme;
		setMode(PALETTE_MODES[theme as keyof typeof PALETTE_MODES]);
	} else {
		delete document.documentElement.dataset.colorTheme;
		setMode(theme === ColorMode.LIGHT || theme === ColorMode.DARK ? theme : ColorMode.SYSTEM);
	}
}
