"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "@/components/navbar";

const DESERT_TYPES = [
  { name: "Erg / Sand Sea", description: "Vast rolling seas of sand dunes sculpted by wind over millennia. Characterized by sinuous barchan and star dune formations.", region: "Sahara, Arabian Desert", color: "#e8a94c" },
  { name: "Rocky Plateau (Hammada)", description: "Bare rock surfaces swept clean of sand, exposing ancient geological strata. Resistant to erosion, these surfaces tell the story of deep time.", region: "North Africa, Middle East", color: "#c9913e" },
  { name: "Salt Flat (Playa)", description: "Hyper-arid basins where mineral-laden water evaporated, leaving blinding white salt crusts that mirror the sky.", region: "Atacama, Bonneville", color: "#d4c4a0" },
  { name: "Gravel Plain (Reg)", description: "Sheets of wind-polished pebbles and gravel, a desert pavement formed by deflation of finer particles.", region: "Gobi, Namib", color: "#b8946a" },
  { name: "Oasis Zone", description: "Rare pockets where groundwater reaches the surface, sustaining dense vegetation in stark contrast to surrounding aridity.", region: "Worldwide", color: "#7a9e6e" },
  { name: "Canyon Lands", description: "Deep gorges carved by ancient rivers through layered sedimentary rock, now dry but revealing millennia of geological history.", region: "American Southwest", color: "#a0522d" },
];

function simulatePrediction(imageData: string) {
  const seed = imageData.length % DESERT_TYPES.length;
  const primary = DESERT_TYPES[seed];
  const accuracy = 78 + ((imageData.length % 1000) / 1000) * 18;

  const others = DESERT_TYPES
    .filter((_, i) => i !== seed)
    .slice(0, 3)
    .map((t, i) => ({
      ...t,
      score: Math.max(5, (accuracy - 15 - i * 12) + ((imageData.length % 100) / 100) * 5),
    }));

  return { primary: { ...primary, score: accuracy }, others };
}

function AnimatedBar({ target, color }: { target: number; color: string }) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setWidth(target), 200);
    return () => clearTimeout(timer);
  }, [target]);

  return (
    <div className="h-1 bg-[#c9913e]/10 rounded-full overflow-hidden">
      <div
        style={{ width: `${width}%`, backgroundColor: color, transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)" }}
        className="h-full rounded-full"
      />
    </div>
  );
}

export default function PredictionPage() {
  const router = useRouter();
  const [image, setImage] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<ReturnType<typeof simulatePrediction> | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const stored = sessionStorage.getItem("desertImage");
    const name = sessionStorage.getItem("desertFileName");

    if (!stored) {
      router.push("/");
      return;
    }

    setImage(stored);
    setFileName(name || "Unknown Terrain");

    const analysisTimer = setTimeout(() => {
      setResult(simulatePrediction(stored));
      setLoading(false);
      const revealTimer = setTimeout(() => setRevealed(true), 100);
      return () => clearTimeout(revealTimer);
    }, 2200);

    return () => clearTimeout(analysisTimer);
  }, [router]);

  // Prevent hydration mismatch
  if (!mounted) return null;

  return (
    <main className="min-h-screen bg-[#0f0b06] text-[#e8c98a] overflow-x-hidden">
      <Navbar />

      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-1/4 right-1/4 w-[600px] h-[400px] bg-[#c9913e]/5 rounded-full blur-[100px]" />
        <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-[#0f0b06] to-transparent" />
      </div>

      <section className="relative z-10 pt-32 pb-20 px-6 max-w-6xl mx-auto">
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
          <div className="flex flex-col items-center justify-center min-h-[60vh] gap-8">
            <div className="relative w-20 h-20">
              <div className="absolute inset-0 rounded-full border border-[#c9913e]/20 animate-ping" />
              <div className="absolute inset-2 rounded-full border border-[#c9913e]/40" />
              <div className="absolute inset-4 rounded-full border-t border-[#e8a94c] animate-spin" />
              <div className="absolute inset-6 rounded-full bg-[#c9913e]/10" />
            </div>
            <div className="text-center">
              <p className="font-['Cormorant_Garamond'] text-2xl text-[#e8c98a] mb-2">Reading the Terrain</p>
              <p className="font-['EB_Garamond'] text-sm text-[#b8946a]/60">Analyzing geological features, texture, and topography…</p>
            </div>
            <div className="flex flex-col gap-2 w-64">
              {["Preprocessing image", "Extracting features", "Running classification"].map((step, i) => (
                <div key={step} className="flex items-center gap-3">
                  <div
                    className="w-1.5 h-1.5 rounded-full transition-all duration-500"
                    style={{
                      backgroundColor: "#c9913e",
                      opacity: 0.4 + i * 0.2,
                    }}
                  />
                  <span className="font-['EB_Garamond'] text-xs text-[#b8946a]/50 tracking-[0.1em]">{step}</span>
                </div>
              ))}
            </div>
          </div>
        ) : result ? (
          <div className={`transition-all duration-700 ${revealed ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
            <div className="mb-12 flex items-center gap-3">
              <span className="h-px w-8 bg-[#c9913e]/60" />
              <span className="font-['Cormorant_Garamond'] text-xs tracking-[0.3em] uppercase text-[#c9913e]/80">Analysis Complete</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <div className="relative border border-[#c9913e]/30 mb-6 overflow-hidden">
                  {["top-0 left-0", "top-0 right-0", "bottom-0 left-0", "bottom-0 right-0"].map((pos, i) => (
                    <div key={i} className={`absolute ${pos} w-4 h-4 z-10`}>
                      <div className={`absolute w-full h-px bg-[#e8a94c] ${pos.includes("bottom") ? "bottom-0" : "top-0"}`} />
                      <div className={`absolute h-full w-px bg-[#e8a94c] ${pos.includes("right") ? "right-0" : "left-0"}`} />
                    </div>
                  ))}
                  {image && (
                    <>
                      <img src={image} alt={fileName} className="w-full object-cover max-h-72" />
                      <div className="absolute inset-0 bg-gradient-to-t from-[#0f0b06]/70 to-transparent" />
                      <div className="absolute bottom-3 left-4">
                        <span className="font-['EB_Garamond'] text-xs text-[#e8c98a]/60">{fileName}</span>
                      </div>
                    </>
                  )}
                </div>

                <div className="border border-[#c9913e]/30 p-8 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-[#c9913e]/5 rounded-full -translate-y-1/2 translate-x-1/2 blur-2xl" />
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <p className="font-['EB_Garamond'] text-xs tracking-[0.25em] uppercase text-[#b8946a]/60 mb-1">Primary Classification</p>
                      <h2 className="font-['Cormorant_Garamond'] text-3xl font-semibold text-[#e8c98a]">{result.primary.name}</h2>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="relative w-16 h-16">
                        <svg className="w-16 h-16 -rotate-90" viewBox="0 0 64 64">
                          <circle cx="32" cy="32" r="26" fill="none" stroke="#c9913e20" strokeWidth="4" />
                          <circle
                            cx="32" cy="32" r="26"
                            fill="none" stroke="#e8a94c" strokeWidth="4" strokeLinecap="round"
                            strokeDasharray={`${2 * Math.PI * 26}`}
                            strokeDashoffset={`${2 * Math.PI * 26 * (1 - result.primary.score / 100)}`}
                            style={{ transition: "stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)" }}
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="font-['Cormorant_Garamond'] text-sm font-semibold text-[#e8a94c]">{result.primary.score.toFixed(1)}%</span>
                        </div>
                      </div>
                      <span className="font-['EB_Garamond'] text-xs text-[#b8946a]/50 mt-1">confidence</span>
                    </div>
                  </div>
                  <p className="font-['EB_Garamond'] text-base text-[#b8946a]/80 leading-relaxed mb-4">{result.primary.description}</p>
                  <div className="flex items-center gap-2">
                    <svg className="w-3.5 h-3.5 text-[#c9913e]/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <span className="font-['EB_Garamond'] text-xs text-[#b8946a]/50 tracking-[0.1em]">Commonly found: {result.primary.region}</span>
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-6">
                <div className="border border-[#c9913e]/20 p-6">
                  <p className="font-['EB_Garamond'] text-xs tracking-[0.25em] uppercase text-[#b8946a]/60 mb-5">Other Possibilities</p>
                  <div className="flex flex-col gap-5">
                    {result.others.map((item) => (
                      <div key={item.name} className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span className="font-['Cormorant_Garamond'] text-base text-[#e8c98a]">{item.name}</span>
                          <span className="font-['EB_Garamond'] text-xs text-[#b8946a]/60">{item.score.toFixed(1)}%</span>
                        </div>
                        <AnimatedBar target={item.score} color={item.color} />
                      </div>
                    ))}
                  </div>
                </div>

                <div className="border border-[#c9913e]/20 p-6">
                  <p className="font-['EB_Garamond'] text-xs tracking-[0.25em] uppercase text-[#b8946a]/60 mb-5">Detected Characteristics</p>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: "Texture", value: "Fine granular" },
                      { label: "Elevation", value: "Variable" },
                      { label: "Aridity", value: "Extreme" },
                      { label: "Wind Score", value: "High" },
                    ].map((c) => (
                      <div key={c.label} className="bg-[#c9913e]/5 border border-[#c9913e]/15 p-3">
                        <p className="font-['EB_Garamond'] text-xs text-[#b8946a]/50 tracking-[0.1em] uppercase mb-1">{c.label}</p>
                        <p className="font-['Cormorant_Garamond'] text-base text-[#e8c98a]">{c.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-3">
                  <Link href="/" className="flex-1 py-3 text-center font-['Cormorant_Garamond'] text-sm tracking-[0.18em] uppercase border border-[#c9913e]/40 text-[#e8a94c] hover:bg-[#c9913e]/10 transition-all duration-300">
                    New Analysis
                  </Link>
                  <button
                    onClick={() => {
                      const text = `DuneLens Classification\n\nTerrain: ${result.primary.name}\nConfidence: ${result.primary.score.toFixed(1)}%\nRegion: ${result.primary.region}`;
                      navigator.clipboard.writeText(text);
                    }}
                    className="flex-1 py-3 font-['Cormorant_Garamond'] text-sm tracking-[0.18em] uppercase bg-gradient-to-r from-[#8b5a1a] to-[#c9913e] text-[#0f0b06] font-semibold hover:from-[#a06b22] hover:to-[#e8a94c] transition-all duration-300"
                  >
                    Copy Results
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </main>
  );
}