import { Workspace } from "@/components/pricing/workspace";

export default function Home() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Price options, view Greeks, and analyze payoff diagrams.
        </p>
      </div>

      <Workspace />
    </div>
  );
}
