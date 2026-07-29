import type { Metadata } from "next";
import { Nunito, Quicksand } from "next/font/google";
import NavShell from "@/components/NavShell";
import Blob from "@/components/illustrations/Blob";
import "./globals.css";

const nunito = Nunito({
  variable: "--font-nunito",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const quicksand = Quicksand({
  variable: "--font-quicksand",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Mizizi — depression screening",
  description: "Multimodal depression-severity screening: intake, scales, scoring, and clinical notes.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${nunito.variable} ${quicksand.variable} h-full`}>
      <body className="relative min-h-full overflow-x-hidden bg-cream-50 text-ink-900 antialiased">
        <Blob
          className="pointer-events-none fixed -right-24 -top-24 h-96 w-96 opacity-60"
          color="var(--color-sage-100)"
        />
        <Blob
          className="pointer-events-none fixed -left-32 bottom-0 h-80 w-80 opacity-50"
          color="var(--color-clay-100)"
        />
        <div className="relative flex min-h-full flex-col">
          <NavShell />
          <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
            {children}
          </main>
          <footer className="px-4 py-6 text-center text-xs text-sage-600 sm:px-6">
            Local development build — see docs/system-roadmap.md for deployment status.
          </footer>
        </div>
      </body>
    </html>
  );
}
