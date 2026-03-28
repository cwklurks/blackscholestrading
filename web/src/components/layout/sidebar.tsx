"use client";

import Link from "next/link";

export function Sidebar() {
  return (
    <aside
      aria-label="Brand"
      className="fixed inset-y-0 left-0 z-30 flex w-14 flex-col border-r border-border bg-surface"
    >
      <Link
        href="/"
        className="flex h-14 items-center justify-center border-b border-border"
      >
        <span className="font-mono text-sm font-bold text-primary">BST</span>
      </Link>
    </aside>
  );
}
