// Shape of data/explorer/graph.json, written by data/export_explorer_graph.py.

export const DIRECTIONS = ["positive", "negative", "ambiguous", "non-linear"] as const;
export type Direction = (typeof DIRECTIONS)[number];

export interface Concept {
  /** Canonical (English) concept name, shared by all sources. */
  id: string;
  /** German surface form shown in the UI. */
  label: string;
  /** Number of chunks that mention the concept. */
  chunks: number;
}

export interface Evidence {
  chunk: number;
  /** The sentence that states the mechanism, in the source's words. */
  mechanism: string;
  conditions: string;
  horizon: string;
  /** Timeless mechanism (true) or a time-bound statement. */
  timeless: boolean;
}

export interface Relation {
  source: string;
  target: string;
  direction: Direction;
  /** Number of chunks asserting this relation. */
  weight: number;
  evidence: Evidence[];
}

export interface ChunkSource {
  title: string;
  /** e.g. "Wikipedia: Aktie" or "Mjoelnir-Wirkungskette: …" */
  source: string;
  url?: string;
}

export interface GraphData {
  generated: string;
  concepts: Concept[];
  edges: Relation[];
  chunks: Record<string, ChunkSource>;
}

/** One step of a chain: every relation from one concept to the next. */
export interface Step {
  from: string;
  to: string;
  relations: Relation[];
}

/** Net effect of a step or chain: +1 raises, -1 lowers, 0 unclear. */
export type Sign = 1 | -1 | 0;

export interface Chain {
  concepts: string[];
  steps: Step[];
  /** Sum of the relation weights over all steps; more sources = better supported. */
  support: number;
  sign: Sign;
}

export interface Reached {
  id: string;
  hops: number;
}
