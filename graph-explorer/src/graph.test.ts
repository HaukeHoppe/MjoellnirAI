import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { CausalGraph, chainSign, normalize } from "./graph";
import { GraphDataError, parseGraphData } from "./parse";
import type { Direction, GraphData, Relation } from "./types";

function rel(source: string, target: string, direction: Direction = "positive", weight = 1): Relation {
  return { source, target, direction, weight, evidence: [{ chunk: 0, mechanism: `${source} -> ${target}`, conditions: "", horizon: "unspecified", timeless: true }] };
}

// Leitzins -> Anleiherendite -> Anleihekurs, plus a longer detour and a cycle:
//   Leitzins -+-> Anleiherendite --(-)--> Anleihekurs
//             +-> Kreditkosten -> Investitionen -> Anleihekurs
//   Inflation <-> Leitzins (cycle)
function fixture(): GraphData {
  const ids = ["Leitzins", "Anleiherendite", "Anleihekurs", "Kreditkosten", "Investitionen", "Inflation", "Ölpreis"];
  return {
    generated: "2026-10-09T00:00:00Z",
    concepts: ids.map((id) => ({ id, label: id, chunks: 1 })),
    edges: [
      rel("Leitzins", "Anleiherendite", "positive", 2),
      rel("Anleiherendite", "Anleihekurs", "negative"),
      rel("Leitzins", "Kreditkosten"),
      rel("Kreditkosten", "Investitionen", "negative"),
      rel("Investitionen", "Anleihekurs", "ambiguous"),
      rel("Inflation", "Leitzins"),
      rel("Leitzins", "Inflation", "negative"),
      rel("Ölpreis", "Inflation"),
    ],
    chunks: { "0": { title: "Test", source: "Wikipedia: Test", url: "https://de.wikipedia.org/wiki/Test" } },
  };
}

describe("CausalGraph.findChains", () => {
  const graph = new CausalGraph(fixture());

  it("finds the shortest chain first and computes its net direction", () => {
    const [best] = graph.findChains("Leitzins", "Anleihekurs");
    expect(best?.concepts).toEqual(["Leitzins", "Anleiherendite", "Anleihekurs"]);
    expect(best?.support).toBe(3);
    expect(best?.sign).toBe(-1); // higher rates -> higher yields -> lower bond prices
  });

  it("returns longer alternatives after the shortest", () => {
    const chains = graph.findChains("Leitzins", "Anleihekurs");
    expect(chains.map((c) => c.concepts.length)).toEqual([3, 4]);
    expect(chains[1]?.sign).toBe(0); // the "ambiguous" step makes the detour unclear
  });

  it("follows chains across several steps and through the cycle without looping", () => {
    const chains = graph.findChains("Ölpreis", "Anleihekurs");
    expect(chains[0]?.concepts).toEqual(["Ölpreis", "Inflation", "Leitzins", "Anleiherendite", "Anleihekurs"]);
    for (const chain of chains) expect(new Set(chain.concepts).size).toBe(chain.concepts.length);
  });

  it("respects maxHops and limit", () => {
    expect(graph.findChains("Ölpreis", "Anleihekurs", { maxHops: 3 })).toEqual([]);
    expect(graph.findChains("Leitzins", "Anleihekurs", { limit: 1 })).toHaveLength(1);
  });

  it("only follows arrows forward", () => {
    expect(graph.findChains("Anleihekurs", "Leitzins")).toEqual([]);
  });

  it("returns nothing for unknown or identical concepts", () => {
    expect(graph.findChains("Leitzins", "Unbekannt")).toEqual([]);
    expect(graph.findChains("Leitzins", "Leitzins")).toEqual([]);
  });
});

describe("CausalGraph.reachable", () => {
  const graph = new CausalGraph(fixture());

  it("lists effects by distance", () => {
    const reached = graph.reachable("Ölpreis", "forward");
    expect(reached[0]).toEqual({ id: "Inflation", hops: 1 });
    expect(reached.find((r) => r.id === "Anleihekurs")?.hops).toBe(4);
  });

  it("lists causes when going backward", () => {
    expect(graph.reachable("Inflation", "backward", 1).map((r) => r.id).sort()).toEqual(["Leitzins", "Ölpreis"]);
  });
});

describe("CausalGraph.neighborhood", () => {
  it("keeps the seeds and caps the number of concepts", () => {
    const graph = new CausalGraph(fixture());
    const { nodes, relations } = graph.neighborhood(["Anleihekurs"], 2, 4);
    expect(nodes).toContain("Anleihekurs");
    expect(nodes).toHaveLength(4);
    for (const r of relations) expect(nodes.includes(r.source) && nodes.includes(r.target)).toBe(true);
  });
});

describe("CausalGraph.search", () => {
  const graph = new CausalGraph(fixture());

  it("ignores case and umlauts and ranks prefixes before substrings", () => {
    expect(graph.search("olpreis")[0]?.id).toBe("Ölpreis");
    expect(graph.search("anleihe").map((c) => c.id)).toEqual(["Anleihekurs", "Anleiherendite"]);
    expect(graph.search("kurs")[0]?.id).toBe("Anleihekurs");
  });

  it("returns nothing for an empty query", () => {
    expect(graph.search("  ")).toEqual([]);
  });
});

describe("helpers", () => {
  it("normalize folds ß and diacritics", () => {
    expect(normalize("  Straße Öl ")).toBe("strasse ol");
  });

  it("chainSign multiplies the step signs", () => {
    const step = (direction: Direction) => ({ from: "a", to: "b", relations: [rel("a", "b", direction)] });
    expect(chainSign([step("negative"), step("negative")])).toBe(1);
    expect(chainSign([step("positive"), step("negative")])).toBe(-1);
    expect(chainSign([step("positive"), step("non-linear")])).toBe(0);
  });
});

describe("parseGraphData", () => {
  it("accepts a valid export", () => {
    expect(parseGraphData(JSON.parse(JSON.stringify(fixture()))).edges).toHaveLength(8);
  });

  it("names the offending field", () => {
    const broken = fixture() as unknown as { edges: Record<string, unknown>[] };
    (broken.edges[1] as Record<string, unknown>).direction = "up";
    expect(() => parseGraphData(broken)).toThrow(GraphDataError);
    expect(() => parseGraphData(broken)).toThrow("edges[1].direction");
  });

  it("rejects relations to unknown concepts", () => {
    const broken = fixture();
    broken.edges.push(rel("Leitzins", "Gibt es nicht"));
    expect(() => parseGraphData(broken)).toThrow("edges[8].target");
  });
});

// Against the real export, if it exists (not in the Docker build, where data/ is not part of the context).
const exported = resolve(__dirname, "../../data/explorer/graph.json");
describe.skipIf(!existsSync(exported))("exported graph", () => {
  it("parses and finds chains between well-connected concepts", () => {
    const graph = new CausalGraph(parseGraphData(JSON.parse(readFileSync(exported, "utf-8"))));
    expect(graph.concepts.size).toBeGreaterThan(1000);
    const chains = graph.findChains("Liquidity", "Interest Rates");
    expect(chains.length).toBeGreaterThan(0);
    for (const chain of chains) {
      expect(chain.steps.length).toBeLessThanOrEqual(4);
      for (const step of chain.steps) expect(step.relations.length).toBeGreaterThan(0);
    }
  });
});
