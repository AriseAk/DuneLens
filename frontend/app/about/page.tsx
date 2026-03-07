import Navbar from "@/components/navbar";
import Link from "next/link";

export default function AboutPage() {
  const terrainTypes = [
    { name: "Erg / Sand Sea", icon: "〰", desc: "Undulating dune fields" },
    { name: "Hammada", icon: "◻", desc: "Bare rock plateaus" },
    { name: "Salt Flats", icon: "◈", desc: "Evaporite mineral plains" },
    { name: "Gravel Plains", icon: "◦", desc: "Deflated reg surfaces" },
    { name: "Canyon Lands", icon: "⋁", desc: "Eroded gorge systems" },
    { name: "Oasis Zones", icon: "✦", desc: "Groundwater-fed areas" },
  ];

  return (
    <main className="min-h-screen bg-[#0f0b06] text-[#e8c98a] overflow-x-hidden">
      <Navbar />

      {/* Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute bottom-1/4 left-1/3 w-[500px] h-[400px] bg-[#c9913e]/5 rounded-full blur-[100px]" />
        <svg className="absolute bottom-0 left-0 w-full opacity-10" viewBox="0 0 1440 200" preserveAspectRatio="none" fill="none">
          <path d="M0 200 Q360 50 720 130 Q1080 200 1440 80 L1440 200 Z" fill="#8b5a1a" fillOpacity="0.5" />
        </svg>
      </div>

      <section className="relative z-10 pt-32 pb-20 px-6 max-w-5xl mx-auto">
        {/* Eyebrow */}
        <div className="mb-6 flex items-center gap-3">
          <span className="h-px w-8 bg-[#c9913e]/60" />
          <span className="font-['Cormorant_Garamond'] text-xs tracking-[0.35em] uppercase text-[#c9913e]/80">
            About DuneLens
          </span>
        </div>

        <h1 className="font-['Cormorant_Garamond'] text-5xl md:text-6xl font-light text-[#e8c98a] mb-6 leading-[1.1]">
          The Science of<br />
          <span className="font-semibold text-[#e8a94c]">Desert Reading</span>
        </h1>

        <p className="font-['EB_Garamond'] text-lg text-[#b8946a]/70 max-w-2xl mb-16 leading-relaxed">
          DuneLens uses a convolutional neural network trained on thousands of
          geotagged desert photographs to classify terrain type from a single
          image — with sub-second inference and calibrated confidence scores.
        </p>

        {/* How it works */}
        <div className="mb-16">
          <h2 className="font-['Cormorant_Garamond'] text-2xl text-[#e8c98a] mb-8 tracking-wide">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-[#c9913e]/10">
            {[
              {
                step: "01",
                title: "Upload",
                body: "Your image is sent securely to our inference API. No images are stored beyond the session.",
              },
              {
                step: "02",
                title: "Prediction",
                body: "Our model processes the image and predicts the terrain type, along with a confidence score.",
              },
              {
                step: "03",
                title: "Display",
                body: "Results are displayed instantly, along with insights about the identified terrain and its typical characteristics.",
              },
            ].map((s) => (
              <div key={s.step} className="bg-[#0f0b06] p-8">
                <p className="font-['Cormorant_Garamond'] text-4xl text-[#c9913e]/30 mb-4">{s.step}</p>
                <h3 className="font-['Cormorant_Garamond'] text-xl text-[#e8c98a] mb-3">{s.title}</h3>
                <p className="font-['EB_Garamond'] text-sm text-[#b8946a]/60 leading-relaxed">{s.body}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Terrain types */}
        <div className="mb-16">
          <h2 className="font-['Cormorant_Garamond'] text-2xl text-[#e8c98a] mb-8 tracking-wide">
            Terrain Categories
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {terrainTypes.map((t) => (
              <div
                key={t.name}
                className="border border-[#c9913e]/20 p-5 hover:border-[#c9913e]/50 transition-all duration-300 group"
              >
                <span className="block text-2xl text-[#c9913e]/60 mb-3 group-hover:text-[#e8a94c] transition-colors">
                  {t.icon}
                </span>
                <p className="font-['Cormorant_Garamond'] text-base text-[#e8c98a] mb-1">{t.name}</p>
                <p className="font-['EB_Garamond'] text-xs text-[#b8946a]/50">{t.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Model stats */}
        <div className="border border-[#c9913e]/20 p-8 mb-12 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-48 h-48 bg-[#c9913e]/5 rounded-full -translate-y-1/2 translate-x-1/2 blur-3xl" />
          <h2 className="font-['Cormorant_Garamond'] text-2xl text-[#e8c98a] mb-6">Model Details</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { label: "Architecture", value: "ResNet-50" },
              { label: "Training Images", value: "48,000+" },
              { label: "Val Accuracy", value: "94.7%" },
              { label: "Inference Time", value: "~1.4s" },
            ].map((d) => (
              <div key={d.label}>
                <p className="font-['EB_Garamond'] text-xs tracking-[0.2em] uppercase text-[#b8946a]/50 mb-1">{d.label}</p>
                <p className="font-['Cormorant_Garamond'] text-2xl text-[#e8a94c]">{d.value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="flex justify-center">
          <Link
            href="/"
            className="font-['Cormorant_Garamond'] text-base tracking-[0.2em] uppercase px-10 py-4 bg-gradient-to-r from-[#8b5a1a] to-[#c9913e] text-[#0f0b06] font-semibold hover:from-[#a06b22] hover:to-[#e8a94c] transition-all duration-300"
          >
            Try DuneLens →
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 py-10 border-t border-[#c9913e]/10 text-center">
        <p className="font-['EB_Garamond'] text-xs tracking-[0.2em] uppercase text-[#b8946a]/40">
          © 2026 DuneLens · Desert Terrain Intelligence
        </p>
      </footer>
    </main>
  );
}