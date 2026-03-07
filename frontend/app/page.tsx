"use client";

import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/navbar";

export default function LandingPage() {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleAnalyze = () => {
    if (!preview) return;
    setUploading(true);
    // Store image in sessionStorage to pass to prediction page
    sessionStorage.setItem("desertImage", preview);
    sessionStorage.setItem("desertFileName", fileName || "image.jpg");
    setTimeout(() => {
      router.push("/prediction");
    }, 1200);
  };

  return (
    <main className="min-h-screen bg-[#0f0b06] text-[#e8c98a] overflow-x-hidden">
      <Navbar />

      {/* Background texture / dunes */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {/* Radial glow */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-[#c9913e]/6 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 left-0 right-0 h-64 bg-gradient-to-t from-[#0f0b06] to-transparent" />
        {/* Grain overlay */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
            backgroundSize: "200px 200px",
          }}
        />
        {/* Desert dune silhouette */}
        <svg
          className="absolute bottom-0 left-0 w-full opacity-20"
          viewBox="0 0 1440 300"
          preserveAspectRatio="none"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M0 300 Q180 100 360 200 Q540 300 720 150 Q900 0 1080 180 Q1260 300 1440 200 L1440 300 Z"
            fill="#8b5a1a"
            fillOpacity="0.4"
          />
          <path
            d="M0 300 Q200 150 400 240 Q600 300 800 200 Q1000 100 1200 220 Q1350 280 1440 240 L1440 300 Z"
            fill="#c9913e"
            fillOpacity="0.15"
          />
        </svg>
      </div>

      {/* Hero */}
      <section className="relative z-10 min-h-screen flex flex-col items-center justify-center px-6 pt-24 pb-16">
        {/* Eyebrow */}
        <div className="mb-6 flex items-center gap-3">
          <span className="h-px w-12 bg-[#c9913e]/60" />
          <span className="font-['Cormorant_Garamond'] text-xs tracking-[0.35em] uppercase text-[#c9913e]/80">
            Desert Classification AI
          </span>
          <span className="h-px w-12 bg-[#c9913e]/60" />
        </div>

        {/* Heading */}
        <h1 className="font-['Cormorant_Garamond'] text-center leading-[1.05] mb-4">
          <span className="block text-5xl md:text-7xl font-light text-[#e8c98a] tracking-[-0.01em]">
            Read the
          </span>
          <span className="block text-6xl md:text-8xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-[#e8a94c] via-[#f0d080] to-[#c9913e] tracking-[-0.02em]">
            Desert
          </span>
        </h1>

        <p className="font-['EB_Garamond'] text-center text-lg md:text-xl text-[#b8946a]/70 max-w-lg mb-12 leading-relaxed">
          Upload a photograph of any desert landscape. Our model will identify
          the terrain type and return a confidence score.
        </p>

        {/* Upload Zone */}
        <div className="w-full max-w-xl">
          {!preview ? (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative cursor-pointer group transition-all duration-500 ${
                dragging ? "scale-[1.02]" : ""
              }`}
            >
              {/* Border frame */}
              <div
                className={`absolute inset-0 border transition-all duration-500 ${
                  dragging
                    ? "border-[#e8a94c] shadow-[0_0_40px_#c9913e30]"
                    : "border-[#c9913e]/30 group-hover:border-[#c9913e]/70"
                }`}
              />
              {/* Corner decorations */}
              {["top-0 left-0", "top-0 right-0", "bottom-0 left-0", "bottom-0 right-0"].map((pos, i) => (
                <div key={i} className={`absolute ${pos} w-4 h-4`}>
                  <div className={`absolute w-full h-px bg-[#e8a94c] ${pos.includes("bottom") ? "bottom-0" : "top-0"}`} />
                  <div className={`absolute h-full w-px bg-[#e8a94c] ${pos.includes("right") ? "right-0" : "left-0"}`} />
                </div>
              ))}

              <div className="p-16 flex flex-col items-center gap-5">
                {/* Upload icon */}
                <div className={`w-16 h-16 rounded-full border border-[#c9913e]/40 flex items-center justify-center transition-all duration-500 ${dragging ? "border-[#e8a94c] bg-[#c9913e]/10" : "group-hover:border-[#c9913e]/80 group-hover:bg-[#c9913e]/5"}`}>
                  <svg className="w-7 h-7 text-[#c9913e]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12M8 8l4-4 4 4" />
                  </svg>
                </div>

                <div className="text-center">
                  <p className="font-['Cormorant_Garamond'] text-xl text-[#e8c98a] mb-1">
                    {dragging ? "Release to upload" : "Drop your image here"}
                  </p>
                  <p className="font-['EB_Garamond'] text-sm text-[#b8946a]/60">
                    or click to browse · PNG, JPG, WEBP
                  </p>
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>
          ) : (
            /* Preview state */
            <div className="relative border border-[#c9913e]/40">
              {/* Corner decorations */}
              {["top-0 left-0", "top-0 right-0", "bottom-0 left-0", "bottom-0 right-0"].map((pos, i) => (
                <div key={i} className={`absolute ${pos} w-4 h-4 z-10`}>
                  <div className={`absolute w-full h-px bg-[#e8a94c] ${pos.includes("bottom") ? "bottom-0" : "top-0"}`} />
                  <div className={`absolute h-full w-px bg-[#e8a94c] ${pos.includes("right") ? "right-0" : "left-0"}`} />
                </div>
              ))}

              {/* Preview image */}
              <div className="relative overflow-hidden" style={{ aspectRatio: "16/9" }}>
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0f0b06]/60 to-transparent" />
                <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
                  <span className="font-['EB_Garamond'] text-xs text-[#e8c98a]/70 truncate max-w-[70%]">
                    {fileName}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setPreview(null);
                      setFileName(null);
                    }}
                    className="text-xs text-[#c9913e]/70 hover:text-[#e8a94c] transition-colors font-['EB_Garamond'] tracking-wide"
                  >
                    Change
                  </button>
                </div>
              </div>

              {/* Analyze button */}
              <button
                onClick={handleAnalyze}
                disabled={uploading}
                className={`w-full py-4 font-['Cormorant_Garamond'] text-base tracking-[0.2em] uppercase transition-all duration-500 ${
                  uploading
                    ? "bg-[#c9913e]/20 text-[#e8a94c]/60 cursor-wait"
                    : "bg-gradient-to-r from-[#8b5a1a] to-[#c9913e] text-[#0f0b06] hover:from-[#a06b22] hover:to-[#e8a94c] font-semibold"
                }`}
              >
                {uploading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border border-[#e8a94c]/40 border-t-[#e8a94c] rounded-full animate-spin" />
                    Analyzing terrain…
                  </span>
                ) : (
                  "Analyze Desert Terrain →"
                )}
              </button>
            </div>
          )}
        </div>

        {/* Feature pills */}
        <div className="mt-10 flex flex-wrap justify-center gap-3">
          {["Sand Dunes", "Rocky Plateaus", "Salt Flats", "Oasis Zones", "Gravel Plains"].map((tag) => (
            <span
              key={tag}
              className="font-['EB_Garamond'] text-xs tracking-[0.15em] uppercase px-4 py-1.5 border border-[#c9913e]/20 text-[#b8946a]/60"
            >
              {tag}
            </span>
          ))}
        </div>
      </section>
      <footer className="relative z-10 py-8 border-t border-[#c9913e]/10 text-center">
        <p className="font-['EB_Garamond'] text-xs tracking-[0.2em] uppercase text-[#b8946a]/40">
          © 2026 DuneLens · Desert Terrain Intelligence
        </p>
      </footer>
      
    </main>
  );
}