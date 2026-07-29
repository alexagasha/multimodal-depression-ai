"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Leaf from "./illustrations/Leaf";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/intake", label: "New session" },
];

export default function NavShell() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20 border-b border-sage-200 bg-cream-50/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-display text-lg font-semibold text-sage-800">
          <Leaf className="h-7 w-7" />
          Mizizi
          <span className="hidden text-sm font-normal text-sage-600 sm:inline">
            depression screening
          </span>
        </Link>
        <nav className="flex items-center gap-1">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
                  active
                    ? "bg-sage-500 text-white"
                    : "text-sage-700 hover:bg-sage-100"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
