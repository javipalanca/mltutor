/**
 * Saving files chosen by the user. In the app the native save dialog and the writing
 * happen in Rust (`save_file`): the frontend has no filesystem or dialog permission.
 * In a plain browser (`npm run dev` without Tauri) it falls back to a download.
 */
import { isTauri } from "@tauri-apps/api/core";
import { saveFile, type SaveKind } from "./backend";

export type { SaveKind };

/** Returns the saved file name, or `null` if the user cancelled. */
export async function saveBytes(defaultName: string, bytes: Uint8Array, kind: SaveKind): Promise<string | null> {
  if (!isTauri()) {
    const url = URL.createObjectURL(new Blob([bytes.slice().buffer]));
    const link = document.createElement("a");
    link.href = url;
    link.download = defaultName;
    link.click();
    URL.revokeObjectURL(url);
    return defaultName;
  }
  return saveFile(defaultName, kind, bytes);
}

export function saveText(defaultName: string, text: string, kind: SaveKind): Promise<string | null> {
  return saveBytes(defaultName, new TextEncoder().encode(text), kind);
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
