"use client";

import { useState } from "react";
import { Hero } from "@/components/Hero";
import { QueryInterface } from "@/components/QueryInterface";
import { Features } from "@/components/Features";
import { Footer } from "@/components/Footer";
import { BackgroundBeams } from "@/components/ui/BackgroundBeams";

export default function Home() {
  const [showQuery, setShowQuery] = useState(false);

  return (
    <main className="min-h-screen bg-background relative overflow-hidden">
      <BackgroundBeams />
      
      <div className="relative z-10">
        {!showQuery ? (
          <>
            <Hero onGetStarted={() => setShowQuery(true)} />
            <Features />
            <Footer />
          </>
        ) : (
          <QueryInterface onBack={() => setShowQuery(false)} />
        )}
      </div>
    </main>
  );
}
