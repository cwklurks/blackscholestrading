import { Sidebar } from "./sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main role="main" className="flex-1 pl-14">
        <div className="mx-auto max-w-7xl p-4 lg:p-6">{children}</div>
      </main>
    </div>
  );
}
