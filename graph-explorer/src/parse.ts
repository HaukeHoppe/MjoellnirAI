// Runtime validation of graph.json: the file is produced by a separate Python script, so its shape is checked
// once on load instead of trusting a cast. Errors name the offending path, e.g. "edges[12].direction".

import { DIRECTIONS, type ChunkSource, type Concept, type Direction, type Evidence, type GraphData, type Relation } from "./types";

export class GraphDataError extends Error {
  constructor(path: string, expected: string) {
    super(`graph.json: ${path} should be ${expected}`);
    this.name = "GraphDataError";
  }
}

type Json = Record<string, unknown>;

function object(value: unknown, path: string): Json {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new GraphDataError(path, "an object");
  return value as Json;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new GraphDataError(path, "an array");
  return value;
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string") throw new GraphDataError(path, "a string");
  return value;
}

function count(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new GraphDataError(path, "a non-negative integer");
  }
  return value;
}

function direction(value: unknown, path: string): Direction {
  if (!DIRECTIONS.includes(value as Direction)) throw new GraphDataError(path, `one of ${DIRECTIONS.join(", ")}`);
  return value as Direction;
}

function concept(value: unknown, path: string): Concept {
  const o = object(value, path);
  return { id: string(o.id, `${path}.id`), label: string(o.label, `${path}.label`), chunks: count(o.chunks, `${path}.chunks`) };
}

function evidence(value: unknown, path: string): Evidence {
  const o = object(value, path);
  if (typeof o.timeless !== "boolean") throw new GraphDataError(`${path}.timeless`, "a boolean");
  return {
    chunk: count(o.chunk, `${path}.chunk`),
    mechanism: string(o.mechanism, `${path}.mechanism`),
    conditions: string(o.conditions, `${path}.conditions`),
    horizon: string(o.horizon, `${path}.horizon`),
    timeless: o.timeless,
  };
}

function relation(value: unknown, path: string, known: Set<string>): Relation {
  const o = object(value, path);
  const source = string(o.source, `${path}.source`);
  const target = string(o.target, `${path}.target`);
  if (!known.has(source)) throw new GraphDataError(`${path}.source`, `a known concept (got "${source}")`);
  if (!known.has(target)) throw new GraphDataError(`${path}.target`, `a known concept (got "${target}")`);
  return {
    source,
    target,
    direction: direction(o.direction, `${path}.direction`),
    weight: count(o.weight, `${path}.weight`),
    evidence: array(o.evidence, `${path}.evidence`).map((e, i) => evidence(e, `${path}.evidence[${i}]`)),
  };
}

function chunkSource(value: unknown, path: string): ChunkSource {
  const o = object(value, path);
  const source: ChunkSource = { title: string(o.title, `${path}.title`), source: string(o.source, `${path}.source`) };
  if (o.url !== undefined) source.url = string(o.url, `${path}.url`);
  return source;
}

export function parseGraphData(value: unknown): GraphData {
  const root = object(value, "root");
  const concepts = array(root.concepts, "concepts").map((c, i) => concept(c, `concepts[${i}]`));
  const known = new Set(concepts.map((c) => c.id));
  const edges = array(root.edges, "edges").map((e, i) => relation(e, `edges[${i}]`, known));
  const chunks: Record<string, ChunkSource> = {};
  for (const [id, c] of Object.entries(object(root.chunks, "chunks"))) chunks[id] = chunkSource(c, `chunks.${id}`);
  return { generated: string(root.generated, "generated"), concepts, edges, chunks };
}
