/**
 * Saving files chosen by the user: native save dialog (`plugin-dialog`) and
 * `plugin-fs` `writeFile`, whose scope only contains the path picked in that dialog.
 */
import { isTauri } from "@tauri-apps/api/core";

export interface SaveFilter {
  name: string;
  extensions: string[];
}

export const PNG_FILTER: SaveFilter = { name: "Imagen PNG", extensions: ["png"] };
export const PYTHON_FILTER: SaveFilter = { name: "Código Python", extensions: ["py"] };

/** Returns the saved file name, or `null` if the user cancelled. */
export async function saveBytes(defaultName: string, bytes: Uint8Array, filter: SaveFilter): Promise<string | null> {
  if (!isTauri()) {
    const url = URL.createObjectURL(new Blob([bytes.slice().buffer]));
    const link = document.createElement("a");
    link.href = url;
    link.download = defaultName;
    link.click();
    URL.revokeObjectURL(url);
    return defaultName;
  }
  const [{ save }, { writeFile }] = await Promise.all([
    import("@tauri-apps/plugin-dialog"),
    import("@tauri-apps/plugin-fs"),
  ]);
  const path = await save({ defaultPath: defaultName, filters: [filter] });
  if (!path) {
    return null;
  }
  await writeFile(path, bytes);
  return path.split(/[\\/]/).pop() ?? path;
}

export function saveText(defaultName: string, text: string, filter: SaveFilter): Promise<string | null> {
  return saveBytes(defaultName, new TextEncoder().encode(text), filter);
}

export function dataUrlToBytes(dataUrl: string): Uint8Array {
  const base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/** Native "Abrir" dialog for a CSV; `null` if cancelled. */
export async function pickCsvFile(filterName: string): Promise<string | null> {
  if (!isTauri()) {
    return null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const path = await open({ multiple: false, directory: false, filters: [{ name: filterName, extensions: ["csv"] }] });
  return typeof path === "string" ? path : null;
}
