"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet } from "@/lib/api";

type State<T> = { key: string | null; data: T | null; error: string | null };

/** GET `path` (null = idle). Previous data is kept while a new path/reload is in flight; `loading` says
 *  whether the shown data belongs to the current request. */
export function useApi<T>(path: string | null, deps: unknown[] = []) {
  const [tick, setTick] = useState(0);
  const key = path ? `${path}#${tick}#${JSON.stringify(deps)}` : null;
  const [s, setS] = useState<State<T>>({ key: null, data: null, error: null });
  const reload = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    if (!path || !key) return;
    const ctl = new AbortController();
    apiGet<T>(path, ctl.signal)
      .then((d) => setS({ key, data: d, error: null }))
      .catch((e) => {
        if (e.name !== "AbortError") setS((p) => ({ key, data: p.data, error: e.message }));
      });
    return () => ctl.abort();
  }, [path, key]);
  const setData = useCallback((d: T | null) => setS((p) => ({ ...p, data: d })), []);
  return { data: s.data, error: s.key === key ? s.error : null, loading: !!key && s.key !== key, reload, setData };
}

/** Poll a URL every `ms` while `active(data)` is true. */
export function usePoll<T>(path: string | null, active: (d: T | null) => boolean, ms = 2000) {
  const r = useApi<T>(path);
  const activeRef = useRef(active);
  const dataRef = useRef(r.data);
  useEffect(() => {
    activeRef.current = active;
    dataRef.current = r.data;
  });
  const { reload } = r;
  useEffect(() => {
    if (!path) return;
    const id = setInterval(() => {
      if (activeRef.current(dataRef.current)) reload();
    }, ms);
    return () => clearInterval(id);
  }, [path, ms, reload]);
  return r;
}
