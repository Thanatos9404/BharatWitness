import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "BharatWitness - AI-Powered Legal Document Assistant",
  description:
    "Production-grade RAG system for Indian government policy and legal documents with temporal reasoning and multi-lingual support.",
  keywords: [
    "BharatWitness",
    "Legal AI",
    "RAG",
    "Indian Government",
    "Policy Documents",
    "NCIIPC",
  ],
  authors: [{ name: "BharatWitness Team" }],
  openGraph: {
    title: "BharatWitness - AI-Powered Legal Document Assistant",
    description:
      "Query Indian government policies with AI-powered temporal reasoning",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
