// Centralised client for the PS32 Python backend (FastAPI). All screens go through here.
export const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
// Optional write token (server: ACCESSMAP_API_TOKEN). Only needed when the API is exposed beyond a trusted LAN.
const TOKEN = process.env.NEXT_PUBLIC_ACCESSMAP_API_TOKEN || "";
const authHeaders = (): Record<string, string> => (TOKEN ? { "x-accessmap-token": TOKEN } : {});

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** Absolute URL for an API path or a backend-served asset path (e.g. /vision/outputs/...). */
export function apiUrl(path?: string | null): string {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown })?.detail;
    const msg = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${res.status}`;
    throw new ApiError(res.status, msg);
  }
  return body as T;
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { signal, cache: "no-store" });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, `Cannot reach the AI backend at ${API_BASE}. Is it running?`);
  }
  return parse<T>(res);
}

export async function apiSend<T>(path: string, body?: unknown, method = "POST"): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      method,
      headers: { ...authHeaders(), ...(body === undefined ? {} : { "content-type": "application/json" }) },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, `Cannot reach the AI backend at ${API_BASE}.`);
  }
  return parse<T>(res);
}

/** Multipart upload with real progress events (fetch has no upload progress). */
export function apiUpload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl(path));
    for (const [k, v] of Object.entries(authHeaders())) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = xhr.responseText;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else {
        const d = (body as { detail?: unknown })?.detail;
        reject(new ApiError(xhr.status, typeof d === "string" ? d : `HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, `Cannot reach the AI backend at ${API_BASE}.`));
    xhr.send(form);
  });
}
