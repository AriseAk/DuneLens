"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const links = [
    { href: "/", label: "Home" },
    { href: "/about", label: "About" },
  ];

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? "bg-[#1a1208]/80 backdrop-blur-md border-b border-[#c9913e]/20 py-3"
          : "bg-transparent py-6"
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="group flex items-center gap-3">
          <div className="relative w-8 h-8">
            <Image src="/icon.svg" alt="DuneLens Logo" width={32} height={32} />
          </div>
          <span className="font-['Cormorant_Garamond'] text-xl font-semibold tracking-[0.12em] text-[#e8c98a] uppercase">
            Dune<span className="text-[#c9913e]">Lens</span>
          </span>
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-8">
          {links.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={`font-['Cormorant_Garamond'] text-sm tracking-[0.18em] uppercase transition-all duration-300 relative group ${
                pathname === href
                  ? "text-[#e8a94c]"
                  : "text-[#c4a96e]/70 hover:text-[#e8c98a]"
              }`}
            >
              {label}
              <span
                className={`absolute -bottom-1 left-0 h-px bg-[#c9913e] transition-all duration-300 ${
                  pathname === href ? "w-full" : "w-0 group-hover:w-full"
                }`}
              />
            </Link>
          ))}
          <Link
            href="/"
            className="font-['Cormorant_Garamond'] text-sm tracking-[0.18em] uppercase px-5 py-2 border border-[#c9913e]/60 text-[#e8a94c] hover:bg-[#c9913e]/10 hover:border-[#e8a94c] transition-all duration-300"
          >
            Analyze
          </Link>
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="md:hidden flex flex-col gap-1.5 p-2"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          <span className={`block w-6 h-px bg-[#e8a94c] transition-all duration-300 ${menuOpen ? "rotate-45 translate-y-2" : ""}`} />
          <span className={`block w-4 h-px bg-[#e8a94c] transition-all duration-300 ${menuOpen ? "opacity-0" : ""}`} />
          <span className={`block w-6 h-px bg-[#e8a94c] transition-all duration-300 ${menuOpen ? "-rotate-45 -translate-y-2" : ""}`} />
        </button>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="md:hidden bg-[#1a1208]/95 backdrop-blur-md border-t border-[#c9913e]/20 px-6 py-6 flex flex-col gap-4">
          {links.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              onClick={() => setMenuOpen(false)}
              className="font-['Cormorant_Garamond'] text-sm tracking-[0.18em] uppercase text-[#c4a96e]/70 hover:text-[#e8c98a] transition-colors"
            >
              {label}
            </Link>
          ))}
          <Link
            href="/"
            onClick={() => setMenuOpen(false)}
            className="font-['Cormorant_Garamond'] text-sm tracking-[0.18em] uppercase px-5 py-2 border border-[#c9913e]/60 text-[#e8a94c] text-center hover:bg-[#c9913e]/10 transition-all duration-300"
          >
            Analyze
          </Link>
        </div>
      )}
    </nav>
  );
}