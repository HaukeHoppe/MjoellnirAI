// Directed cause -> effect graph over the exported concepts, with the searches the explorer needs:
// chains between two concepts, everything reachable from (or leading to) a concept, a neighborhood for
// drawing, and a typo-tolerant-ish concept search.

import type { Chain, Concept, GraphData, Reached, Relation, Sign, Step } from "./types";

export type Way = "forward" | "backward";

export interface ChainOptions {
  /** Longest chain in steps; the pipeline's chain search uses 4 (CHAIN_MAX_HOPS). */
  maxHops?: number;
  /** Number of chains returned. */
  limit?: number;
}

/** Upper bound of paths collected per length, so a dense hub cannot blow up the search. */
const MAX_PATHS_PER_LENGTH = 500;

/** Lowercase, without diacritics, ß -> ss: "Ölpreis" and "olpreis" match. */
export function normalize(text: string): string {
  return text.toLowerCase().replace(/ß/g, "ss").normalize("NFD").replace(/[̀-ͯ]/g, "").trim();
}

export function stepSign(relations: Relation[]): Sign {
  if (relations.length > 0 && relations.every((r) => r.direction === "positive")) return 1;
  if (relations.length > 0 && relations.every((r) => r.direction === "negative")) return -1;
  return 0;
}

/** Signs multiply along a chain (more X -> less Y -> more Z); one unclear step makes the whole chain unclear. */
export function chainSign(steps: Step[]): Sign {
  let sign: Sign = 1;
  for (const step of steps) {
    const s = stepSign(step.relations);
    if (s === 0) return 0; // also avoids -0 from -1 * 0
    sign = (sign * s) as Sign;
  }
  return sign;
}

export class CausalGraph {
  readonly concepts = new Map<string, Concept>();
  /** source -> target -> relations (several directions are possible for one pair). */
  private readonly out = new Map<string, Map<string, Relation[]>>();
  private readonly in = new Map<string, Map<string, Relation[]>>();
  private readonly searchIndex: { concept: Concept; label: string; id: string }[];

  constructor(readonly data: GraphData) {
    for (const concept of data.concepts) this.concepts.set(concept.id, concept);
    for (const relation of data.edges) {
      add(this.out, relation.source, relation.target, relation);
      add(this.in, relation.target, relation.source, relation);
    }
    this.searchIndex = data.concepts.map((concept) => ({
      concept,
      label: normalize(concept.label),
      id: normalize(concept.id),
    }));
  }

  has(id: string): boolean {
    return this.concepts.has(id);
  }

  label(id: string): string {
    return this.concepts.get(id)?.label ?? id;
  }

  /** Number of relations touching the concept. */
  degree(id: string): number {
    return relationCount(this.out.get(id)) + relationCount(this.in.get(id));
  }

  /** Relations grouped by neighbor: effects (forward) or causes (backward) of a concept. */
  steps(id: string, way: Way): Step[] {
    const neighbors = (way === "forward" ? this.out : this.in).get(id) ?? new Map<string, Relation[]>();
    return [...neighbors].map(([other, relations]) =>
      way === "forward" ? { from: id, to: other, relations } : { from: other, to: id, relations },
    );
  }

  relationsBetween(from: string, to: string): Relation[] {
    return this.out.get(from)?.get(to) ?? [];
  }

  /** Hop distance from `start` to every concept within `maxHops`, following (or against) the arrows. */
  distances(start: string, way: Way, maxHops: number): Map<string, number> {
    const adjacency = way === "forward" ? this.out : this.in;
    const dist = new Map<string, number>([[start, 0]]);
    let frontier = [start];
    for (let hop = 1; hop <= maxHops && frontier.length > 0; hop++) {
      const next: string[] = [];
      for (const node of frontier) {
        for (const neighbor of adjacency.get(node)?.keys() ?? []) {
          if (dist.has(neighbor)) continue;
          dist.set(neighbor, hop);
          next.push(neighbor);
        }
      }
      frontier = next;
    }
    return dist;
  }

  /** Concepts reachable from `start` (forward: its effects; backward: its causes), nearest and best connected first. */
  reachable(start: string, way: Way, maxHops = 4): Reached[] {
    if (!this.has(start)) return [];
    return [...this.distances(start, way, maxHops)]
      .filter(([id]) => id !== start)
      .map(([id, hops]) => ({ id, hops }))
      .sort((a, b) => a.hops - b.hops || this.degree(b.id) - this.degree(a.id) || a.id.localeCompare(b.id));
  }

  /**
   * Simple directed paths from `from` to `to`, shortest first, then best supported.
   * A backward BFS from `to` gives each concept its distance to the goal, so the depth-first enumeration only
   * enters concepts that can still reach the goal within the remaining steps.
   */
  findChains(from: string, to: string, { maxHops = 4, limit = 3 }: ChainOptions = {}): Chain[] {
    if (!this.has(from) || !this.has(to) || from === to) return [];
    const toGoal = this.distances(to, "backward", maxHops);
    const shortest = toGoal.get(from);
    if (shortest === undefined) return [];

    const chains: Chain[] = [];
    for (let length = shortest; length <= maxHops && chains.length < limit; length++) {
      const found: string[][] = [];
      const path = [from];
      const onPath = new Set(path);
      const visit = (node: string): void => {
        if (found.length >= MAX_PATHS_PER_LENGTH) return;
        const left = length - (path.length - 1);
        if (left === 0) {
          if (node === to) found.push([...path]);
          return;
        }
        for (const next of this.out.get(node)?.keys() ?? []) {
          const dist = toGoal.get(next);
          // The goal only as the last step, and only concepts that can reach it in the steps left.
          if (onPath.has(next) || dist === undefined || dist > left - 1 || (next === to && left > 1)) continue;
          path.push(next);
          onPath.add(next);
          visit(next);
          path.pop();
          onPath.delete(next);
        }
      };
      visit(from);
      chains.push(...found.map((concepts) => this.chain(concepts)).sort((a, b) => b.support - a.support));
    }
    return chains.slice(0, limit);
  }

  chain(concepts: string[]): Chain {
    const steps = concepts.slice(1).map((to, i) => {
      const from = concepts[i] as string;
      return { from, to, relations: this.relationsBetween(from, to) };
    });
    const support = steps.reduce((sum, s) => sum + s.relations.reduce((w, r) => w + r.weight, 0), 0);
    return { concepts, steps, support, sign: chainSign(steps) };
  }

  /**
   * Concepts around `seeds` (ignoring arrow direction) for drawing, at most `maxNodes`; the seeds always stay,
   * further concepts are added by distance and then by how well connected they are.
   */
  neighborhood(seeds: string[], depth = 1, maxNodes = 40): { nodes: string[]; relations: Relation[] } {
    const kept = new Set(seeds.filter((id) => this.has(id)));
    let frontier = [...kept];
    for (let hop = 1; hop <= depth && kept.size < maxNodes; hop++) {
      const candidates = new Set<string>();
      for (const node of frontier) {
        for (const map of [this.out.get(node), this.in.get(node)]) {
          for (const neighbor of map?.keys() ?? []) if (!kept.has(neighbor)) candidates.add(neighbor);
        }
      }
      const ranked = [...candidates].sort((a, b) => this.degree(b) - this.degree(a) || a.localeCompare(b));
      frontier = ranked.slice(0, maxNodes - kept.size);
      for (const id of frontier) kept.add(id);
    }
    const relations: Relation[] = [];
    for (const source of kept) {
      for (const [target, rels] of this.out.get(source) ?? []) if (kept.has(target)) relations.push(...rels);
    }
    return { nodes: [...kept], relations };
  }

  /**
   * Concepts matching `query` in the German label or the English id: exact match first, then prefix,
   * word prefix, substring; ties go to the better connected concept.
   */
  search(query: string, limit = 8): Concept[] {
    const q = normalize(query);
    if (!q) return [];
    const scored: { concept: Concept; score: number }[] = [];
    for (const entry of this.searchIndex) {
      const score = Math.min(matchScore(entry.label, q), matchScore(entry.id, q) + 0.5);
      if (score < Infinity) scored.push({ concept: entry.concept, score });
    }
    return scored
      .sort(
        (a, b) =>
          a.score - b.score ||
          this.degree(b.concept.id) - this.degree(a.concept.id) ||
          a.concept.label.length - b.concept.label.length,
      )
      .slice(0, limit)
      .map((s) => s.concept);
  }
}

function matchScore(text: string, query: string): number {
  if (text === query) return 0;
  if (text.startsWith(query)) return 1;
  if (text.split(/[\s\-/]+/).some((word) => word.startsWith(query))) return 2;
  if (text.includes(query)) return 3;
  return Infinity;
}

function add(index: Map<string, Map<string, Relation[]>>, from: string, to: string, relation: Relation): void {
  let neighbors = index.get(from);
  if (!neighbors) index.set(from, (neighbors = new Map()));
  const list = neighbors.get(to);
  if (list) list.push(relation);
  else neighbors.set(to, [relation]);
}

function relationCount(neighbors: Map<string, Relation[]> | undefined): number {
  let n = 0;
  for (const list of neighbors?.values() ?? []) n += list.length;
  return n;
}
