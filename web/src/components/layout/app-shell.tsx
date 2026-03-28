export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <main role="main" className="min-h-screen">
      <div className="mx-auto max-w-[1440px] px-6 py-4 lg:px-10 lg:py-6">
        {children}
      </div>
    </main>
  );
}
