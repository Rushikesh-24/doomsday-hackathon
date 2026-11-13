"use client";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { PageHeader, Loading } from "@/components/app/common";
import { ImageWorkspace } from "@/components/studio/image-workspace";
import { VideoPanel } from "@/components/studio/video-panel";

function Studio() {
  const sp = useSearchParams();
  const [tab, setTab] = useState(sp.get("tab") === "video" ? "video" : "image");
  return (
    <>
      <PageHeader title="AI Studio" description="Run the real AI pipelines: facility detection, accessibility measurement, sign reading and video-to-3D reconstruction."
        actions={
          <div className="flex rounded-lg border bg-card p-1 text-sm">
            {[["image", "Photos"], ["video", "Walkthrough video"]].map(([k, l]) => (
              <button key={k} onClick={() => setTab(k)} className={`rounded-md px-3 py-1 ${tab === k ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}>{l}</button>
            ))}
          </div>
        } />
      {tab === "image" ? <ImageWorkspace initialId={sp.get("id")} /> : <div className="max-w-3xl"><VideoPanel /></div>}
    </>
  );
}

export default function Page() {
  return <Suspense fallback={<Loading />}><Studio /></Suspense>;
}
