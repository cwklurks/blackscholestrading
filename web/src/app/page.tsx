import { Workspace } from "@/components/pricing/workspace";

export default function Home() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Workspace</h1>
      </div>

      <Workspace />
    </div>
  );
}
