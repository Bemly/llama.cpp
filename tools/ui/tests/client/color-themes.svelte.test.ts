import { CONFIG_LOCALSTORAGE_KEY, SETTING_CONFIG_DEFAULT } from '$lib/constants';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { applyTheme } from '$lib/utils/theme';
import { userPrefersMode } from 'mode-watcher';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

function mockDefaults(theme: string) {
	Object.defineProperty(serverStore, 'props', {
		configurable: true,
		get: () =>
			({
				default_generation_settings: { params: { temperature: 0.8 } },
				ui_settings: { theme }
			}) as unknown as typeof serverStore.props
	});
}

describe('color theme preferences', () => {
	beforeEach(() => {
		localStorage.clear();
		settingsStore.initialize();
	});

	afterEach(() => applyTheme('system'));

	it.each([
		['sepia', 'light'],
		['ocean', 'dark'],
		['forest', 'dark'],
		['lavender', 'light']
	])('restores saved %s colors and %s highlighting after reload', (theme, expectedMode) => {
		settingsStore.updateTheme(theme);
		applyTheme('system');
		settingsStore.initialize();

		expect(settingsStore.config.theme).toBe(theme);
		expect(document.documentElement.dataset.colorTheme).toBe(theme);
		expect(userPrefersMode.current).toBe(expectedMode);
	});

	it('previews without overwriting the saved choice', () => {
		settingsStore.updateTheme('sepia');
		applyTheme('ocean');

		expect(document.documentElement.dataset.colorTheme).toBe('ocean');
		expect(JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}').theme).toBe('sepia');

		applyTheme(settingsStore.config.theme);
		expect(document.documentElement.dataset.colorTheme).toBe('sepia');
	});

	it('applies an imported palette and clears it when reset', () => {
		settingsStore.updateTheme('forest');
		const exported = settingsStore.exportSettings();

		settingsStore.updateTheme('lavender');
		settingsStore.importSettings(exported);
		expect(document.documentElement.dataset.colorTheme).toBe('forest');

		settingsStore.resetAll();
		expect(settingsStore.config.theme).toBe(SETTING_CONFIG_DEFAULT.theme);
		expect(document.documentElement.dataset.colorTheme).toBeUndefined();
	});

	it('applies the server palette for a new user and on reset', () => {
		mockDefaults('ocean');
		settingsStore.syncWithServerDefaults();
		expect(document.documentElement.dataset.colorTheme).toBe('ocean');

		settingsStore.updateTheme('sepia');
		settingsStore.initialize();
		settingsStore.syncWithServerDefaults();
		expect(document.documentElement.dataset.colorTheme).toBe('sepia');

		settingsStore.forceSyncWithServerDefaults();
		expect(document.documentElement.dataset.colorTheme).toBe('ocean');
	});

	it.each(['light', 'dark', 'system'])('clears palette overrides for %s', (theme) => {
		applyTheme('forest');
		applyTheme(theme);
		expect(document.documentElement.dataset.colorTheme).toBeUndefined();
		expect(userPrefersMode.current).toBe(theme);
	});

	it('falls back to System for an unsupported theme', () => {
		applyTheme('ocean');
		applyTheme('unknown');
		expect(document.documentElement.dataset.colorTheme).toBeUndefined();
		expect(userPrefersMode.current).toBe('system');
	});
});
