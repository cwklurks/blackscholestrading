"use client";

import { createContext, useContext } from "react";

interface WorkspaceContextValue {
  activeTool: string;
  setActiveTool: (id: string) => void;
}

const WorkspaceContext = createContext<WorkspaceContextValue>({
  activeTool: "pricing",
  setActiveTool: () => {},
});

export const WorkspaceProvider = WorkspaceContext.Provider;

export function useWorkspace(): WorkspaceContextValue {
  return useContext(WorkspaceContext);
}
