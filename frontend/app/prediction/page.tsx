"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "@/components/navbar";

// The classes and colors based on your Python VALUE_MAP
const SEGMENTATION_CLASSES = [
  { id: 1, name: "Trees", color: "#2ca02c" },
  { id: 2, name: "Lush Bushes", color: "#98df8a" },
  { id: 3, name: "Dry Grass", color: "#e8a94c" },
  { id: 4, name: "Dry Bushes", color: "#c9913e" },
  { id: 5, name: "Ground Clutter", color: "#8c564b" },
  { id: 6, name: "Logs", color: "#c49c94" },
  { id: 7, name: "Rocks", color: "#7f7f7f" },
  { id: 8, name: "Landscape", color: "#b8946a" },
  { id: 9, name: "Sky", color: "#1f77b4" },
];

// Mock API response — When your backend is ready, replace this with a real fetch()
function simulateSegmentation(imageData: string) {
  // Simulating the distribution of pixels the model found
  const distribution = [
    { ...SEGMENTATION_CLASSES[8], percentage: 42.5 }, // Sky
    { ...SEGMENTATION_CLASSES[7], percentage: 35.0 }, // Landscape
    { ...SEGMENTATION_CLASSES[3], percentage: 12.3 }, // Dry Bushes
    { ...SEGMENTATION_CLASSES[6], percentage: 8.2 },  // Rocks
    { ...SEGMENTATION_CLASSES[2], percentage: 2.0 },  // Dry Grass
  ];

  // For the mock, we will just return the original image as the mask. 
  // In reality, your API will return a base64 string of the painted segmentation mask.
  return {
    originalImage: imageData,
    maskImage: imageData, // Replace with base64 mask from backend later
    distribution: distribution.sort((a, b) => b.percentage - a.percentage),
  };
}

function AnimatedBar({ target, color }: { target: number; color: string }) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setWidth(target), 200);
    return () => clearTimeout(timer);
  }, [target]);

  return (
    <div className="h-1.5 w-full bg-[#c9913e]/10 rounded-full overflow-hidden flex-1">
      <div
        style={{ width: `${width}%`, backgroundColor: color, transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)" }}
        className="h-full rounded-full"
      />
    </div>
  );
}

export default function PredictionPage() {
  const router = useRouter();
  const [fileName, setFileName] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<ReturnType<typeof simulateSegmentation> | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [viewMode, setViewMode] = useState<"original" | "mask">("mask");

useEffect(() => {
    setMounted(true);
    const stored = sessionStorage.getItem("desertImage");
    const name = sessionStorage.getItem("desertFileName");

    if (!stored) {
      router.push("/");
      return;
    }

    setFileName(name || "Unknown Terrain");

    // NEW: Real API Call to your Python Backend
    const fetchPrediction = async () => {
      try {
        const response = await fetch("http://localhost:8000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_data: stored }),
        });

        if (!response.ok) throw new Error("Failed to process image");

        const data = await response.json();
        
        setResult(data);
        setLoading(false);
        setTimeout(() => setRevealed(true), 100);
      } catch (error) {
        console.error("Prediction Error:", error);
        alert("Failed to connect to the ML model. Is the Python server running?");
        router.push("/");
      }
    };

    fetchPrediction();
  }, [router]);

  if (!mounted) return null;

  return (
    <main className="min-h-screen bg-[#0f0b06] text-[#e8c98a] overflow-x-hidden">
      <Navbar />

      {/* Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-1/4 right-1/4 w-[600px] h-[400px] bg-[#c9913e]/5 rounded-full blur-[100px]" />
        <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-[#0f0b06] to-transparent" />
      </div>

      <section className="relative z-10 pt-32 pb-20 px-6 max-w-7xl mx-auto">
        <Link
          href="/"
          className="inline-flex items-center gap-2 font-['EB_Garamond'] text-xs tracking-[0.2em] uppercase text-[#b8946a]/60 hover:text-[#c9913e] transition-colors mb-10"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 19l-7-7 7-7" />
          </svg>
          New Analysis
        </Link>

        {loading ? (
          /* Loading State */
          <div className="flex flex-col items-center justify-center min-h-[60vh] gap-8">
            <div className="relative w-20 h-20">
              <div className="absolute inset-0 rounded-full border border-[#c9913e]/20 animate-ping" />
              <div className="absolute inset-2 rounded-full border border-[#c9913e]/40" />
              <div className="absolute inset-4 rounded-full border-t border-[#e8a94c] animate-spin" />
              <div className="absolute inset-6 rounded-full bg-[#c9913e]/10" />
            </div>
            <div className="text-center">
              <p className="font-['Cormorant_Garamond'] text-2xl text-[#e8c98a] mb-2">Segmenting Terrain</p>
              <p className="font-['EB_Garamond'] text-sm text-[#b8946a]/60">Generating pixel-wise classification map…</p>
            </div>
            <div className="flex flex-col gap-2 w-64">
              {["Extracting features", "Applying Segformer weights", "Calculating class distribution"].map((step, i) => (
                <div key={step} className="flex items-center gap-3">
                  <div
                    className="w-1.5 h-1.5 rounded-full transition-all duration-500"
                    style={{ backgroundColor: "#c9913e", opacity: 0.4 + i * 0.2 }}
                  />
                  <span className="font-['EB_Garamond'] text-xs text-[#b8946a]/50 tracking-[0.1em]">{step}</span>
                </div>
              ))}
            </div>
          </div>
        ) : result ? (
          /* Results State */
          <div className={`transition-all duration-700 ${revealed ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
            <div className="mb-12 flex items-center gap-3">
              <span className="h-px w-8 bg-[#c9913e]/60" />
              <span className="font-['Cormorant_Garamond'] text-xs tracking-[0.3em] uppercase text-[#c9913e]/80">Segmentation Complete</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
              
              {/* Left Column: Image Viewer (Takes up 7 columns on desktop) */}
              <div className="lg:col-span-7 flex flex-col gap-6">
                
                {/* Image / Mask Display */}
                <div className="relative border border-[#c9913e]/30 overflow-hidden bg-[#0a0704] aspect-[4/3] flex items-center justify-center">
                  {/* Decorative Corners */}
                  {["top-0 left-0", "top-0 right-0", "bottom-0 left-0", "bottom-0 right-0"].map((pos, i) => (
                    <div key={i} className={`absolute ${pos} w-4 h-4 z-20`}>
                      <div className={`absolute w-full h-px bg-[#e8a94c] ${pos.includes("bottom") ? "bottom-0" : "top-0"}`} />
                      <div className={`absolute h-full w-px bg-[#e8a94c] ${pos.includes("right") ? "right-0" : "left-0"}`} />
                    </div>
                  ))}

                  {/* The Image */}
                  <img
                    src={viewMode === "original" ? result.originalImage : result.maskImage}
                    alt={viewMode === "original" ? "Original Upload" : "Segmentation Mask"}
                    className={`w-full h-full object-contain transition-opacity duration-500 ${viewMode === "mask" && "opacity-80"}`}
                  />
                  
                  {/* Mock placeholder text for when you haven't wired the real backend yet */}
                  {viewMode === "mask" && (
                     <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-black/40">
                        <span className="font-['EB_Garamond'] text-sm tracking-[0.2em] text-[#e8c98a]/50 uppercase border border-[#e8c98a]/20 px-4 py-2 bg-black/50 backdrop-blur-sm">
                          Mask Overlay Renders Here
                        </span>
                     </div>
                  )}
                </div>

                {/* View Toggles */}
                <div className="flex justify-between items-center border border-[#c9913e]/20 p-2 bg-[#c9913e]/5">
                  <div className="flex gap-2">
                    <button
                      onClick={() => setViewMode("original")}
                      className={`px-6 py-2 font-['EB_Garamond'] text-xs tracking-[0.15em] uppercase transition-all duration-300 ${
                        viewMode === "original" ? "bg-[#c9913e]/20 text-[#e8a94c] border border-[#c9913e]/40" : "text-[#b8946a]/60 hover:text-[#e8a94c]"
                      }`}
                    >
                      Original
                    </button>
                    <button
                      onClick={() => setViewMode("mask")}
                      className={`px-6 py-2 font-['EB_Garamond'] text-xs tracking-[0.15em] uppercase transition-all duration-300 ${
                        viewMode === "mask" ? "bg-[#c9913e]/20 text-[#e8a94c] border border-[#c9913e]/40" : "text-[#b8946a]/60 hover:text-[#e8a94c]"
                      }`}
                    >
                      Segmentation Map
                    </button>
                  </div>
                  <span className="font-['EB_Garamond'] text-xs text-[#b8946a]/40 pr-4 hidden sm:block">
                    {fileName}
                  </span>
                </div>
              </div>

              {/* Right Column: Class Distribution (Takes up 5 columns on desktop) */}
              <div className="lg:col-span-5 flex flex-col gap-6">
                <div className="border border-[#c9913e]/30 p-8 relative overflow-hidden bg-[#c9913e]/5 h-full">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-[#c9913e]/5 rounded-full -translate-y-1/2 translate-x-1/3 blur-3xl pointer-events-none" />
                  
                  <div className="mb-8">
                    <p className="font-['EB_Garamond'] text-xs tracking-[0.25em] uppercase text-[#b8946a]/60 mb-2">
                      Semantic Analysis
                    </p>
                    <h2 className="font-['Cormorant_Garamond'] text-3xl font-semibold text-[#e8c98a]">
                      Terrain Distribution
                    </h2>
                    <p className="font-['EB_Garamond'] text-sm text-[#b8946a]/70 mt-3 leading-relaxed">
                      The Segformer model has analyzed {result.originalImage.length % 1000 * 1024} pixels to determine the exact composition of this landscape.
                    </p>
                  </div>

                  {/* Distribution List */}
                  <div className="flex flex-col gap-5">
                    {result.distribution.map((item) => (
                      <div key={item.name} className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            {/* Color Swatch */}
                            <span 
                              className="w-3 h-3 rounded-sm shadow-sm" 
                              style={{ backgroundColor: item.color, boxShadow: `0 0 10px ${item.color}40` }} 
                            />
                            <span className="font-['Cormorant_Garamond'] text-lg text-[#e8c98a]">
                              {item.name}
                            </span>
                          </div>
                          <span className="font-['EB_Garamond'] text-sm text-[#e8a94c]">
                            {item.percentage.toFixed(1)}%
                          </span>
                        </div>
                        <AnimatedBar target={item.percentage} color={item.color} />
                      </div>
                    ))}
                  </div>

                  {/* Export CTA */}
                  <div className="mt-10 pt-6 border-t border-[#c9913e]/20">
                    <button
                      onClick={() => {
                        const text = `DuneLens Segmentation\n\n` + result.distribution.map(d => `${d.name}: ${d.percentage.toFixed(1)}%`).join("\n");
                        navigator.clipboard.writeText(text);
                      }}
                      className="w-full py-3 font-['Cormorant_Garamond'] text-sm tracking-[0.18em] uppercase border border-[#c9913e]/40 text-[#e8a94c] hover:bg-[#c9913e]/10 transition-all duration-300"
                    >
                      Copy Statistics
                    </button>
                  </div>

                </div>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </main>
  );
}