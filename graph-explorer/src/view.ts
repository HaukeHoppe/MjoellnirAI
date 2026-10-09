// Force-directed drawing of a small part of the graph (a chain with its surroundings, or one concept's
// neighborhood). Chain concepts are pulled onto a left-to-right line in chain order, so the chain reads as a
// sequence while its neighbors arrange themselves around it.

import { drag } from "d3-drag";
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { select, type Selection } from "d3-selection";
import { zoom, zoomIdentity, type ZoomBehavior } from "d3-zoom";
import { DIRECTIONS, type Direction, type Relation } from "./types";

interface Node extends SimulationNodeDatum {
  id: string;
  label: string;
  radius: number;
  /** Position in the highlighted chain, if the node is on it. */
  chainIndex?: number;
}

interface Link extends SimulationLinkDatum<Node> {
  source: Node;
  target: Node;
  direction: Direction;
  onChain: boolean;
  /** Offset for the second relation of a pair (opposite direction or a second direction), drawn as a curve. */
  bend: number;
}

export interface Scene {
  nodes: string[];
  relations: Relation[];
  /** Concepts of the highlighted chain, in order. */
  chain?: string[];
  focus?: string;
}

export interface ViewOptions {
  label: (id: string) => string;
  degree: (id: string) => number;
  onSelect: (id: string) => void;
}

const MAX_LABEL = 26;

export class GraphView {
  private readonly svg: Selection<SVGSVGElement, unknown, null, undefined>;
  private readonly root: Selection<SVGGElement, unknown, null, undefined>;
  private readonly zoomer: ZoomBehavior<SVGSVGElement, unknown>;
  private simulation?: Simulation<Node, Link>;

  constructor(
    element: SVGSVGElement,
    private readonly options: ViewOptions,
  ) {
    this.svg = select(element);
    const defs = this.svg.append("defs");
    for (const direction of DIRECTIONS) {
      for (const size of ["normal", "chain"] as const) {
        defs
          .append("marker")
          .attr("id", `arrow-${direction}-${size}`)
          .attr("class", `marker ${direction}`)
          .attr("viewBox", "0 -5 10 10")
          .attr("refX", 10)
          .attr("markerWidth", size === "chain" ? 4 : 6)
          .attr("markerHeight", size === "chain" ? 4 : 6)
          .attr("orient", "auto")
          .append("path")
          .attr("d", "M0,-5L10,0L0,5");
      }
    }
    this.root = this.svg.append("g");
    this.zoomer = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => this.root.attr("transform", event.transform));
    this.svg.call(this.zoomer);
  }

  render(scene: Scene): void {
    this.simulation?.stop();
    this.root.selectAll("*").remove();
    const { width, height } = this.size();
    this.svg.call(this.zoomer.transform, zoomIdentity);

    // Chain concepts spread evenly from left to right; everything else floats around the middle.
    const chainLength = Math.max(1, (scene.chain?.length ?? 1) - 1);
    const chainX = (i: number) => width * 0.12 + (width * 0.76 * i) / chainLength;
    const chainIndex = new Map((scene.chain ?? []).map((id, i) => [id, i]));
    const chainSteps = new Set((scene.chain ?? []).slice(1).map((to, i) => `${scene.chain?.[i]}\u0000${to}`));
    const nodes: Node[] = scene.nodes.map((id) => {
      const node: Node = { id, label: this.options.label(id), radius: 5 + Math.min(10, Math.sqrt(this.options.degree(id)) * 1.6) };
      const index = chainIndex.get(id);
      if (index !== undefined) node.chainIndex = index;
      // Start near the final place instead of d3's default spiral around (0, 0).
      node.x = (index !== undefined ? chainX(index) : width / 2) + (Math.random() - 0.5) * 120;
      node.y = height / 2 + (Math.random() - 0.5) * 120;
      return node;
    });
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const pairCount = new Map<string, number>();
    const links: Link[] = [];
    for (const relation of scene.relations) {
      const source = byId.get(relation.source);
      const target = byId.get(relation.target);
      if (!source || !target) continue;
      const pair = [relation.source, relation.target].sort().join("\u0000");
      const nth = pairCount.get(pair) ?? 0;
      pairCount.set(pair, nth + 1);
      links.push({
        source,
        target,
        direction: relation.direction,
        onChain: chainSteps.has(`${relation.source}\u0000${relation.target}`),
        bend: nth === 0 ? 0 : (nth % 2 === 1 ? 1 : -1) * Math.ceil(nth / 2) * 28,
      });
    }

    const link = this.root
      .append("g")
      .selectAll<SVGPathElement, Link>("path")
      .data(links)
      .join("path")
      .attr("class", (d) => `link ${d.direction}${d.onChain ? " on-chain" : ""}`)
      .attr("marker-end", (d) => `url(#arrow-${d.direction}-${d.onChain ? "chain" : "normal"})`);

    const node = this.root
      .append("g")
      .selectAll<SVGGElement, Node>("g")
      .data(nodes)
      .join("g")
      .attr("class", (d) => `node${d.chainIndex !== undefined ? " on-chain" : ""}${d.id === scene.focus ? " focus" : ""}`)
      .on("click", (_event, d) => this.options.onSelect(d.id));
    node.append("circle").attr("r", (d) => d.radius);
    node
      .append("text")
      .attr("x", (d) => d.radius + 4)
      .attr("y", 4)
      .text((d) => (d.label.length > MAX_LABEL ? `${d.label.slice(0, MAX_LABEL - 1)}…` : d.label));
    node.append("title").text((d) => (d.label === d.id ? d.id : `${d.label} (${d.id})`));

    const simulation = forceSimulation<Node, Link>(nodes)
      .force("link", forceLink<Node, Link>(links).distance((d) => (d.onChain ? 140 : 80)).strength(0.4))
      .force("charge", forceManyBody<Node>().strength(-260))
      .force("collide", forceCollide<Node>().radius((d) => d.radius + 14))
      .force("x", forceX<Node>((d) => (d.chainIndex !== undefined ? chainX(d.chainIndex) : width / 2)).strength((d) => (d.chainIndex !== undefined ? 0.6 : 0.04)))
      .force("y", forceY<Node>(height / 2).strength((d) => (d.chainIndex !== undefined ? 0.6 : 0.06)));
    this.simulation = simulation;

    node.call(
      drag<SVGGElement, Node>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }),
    );

    const draw = () => {
      link.attr("d", (d) => linkPath(d));
      node.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    };
    // Settle most of the layout before the first paint, then fit it into view; the rest animates.
    simulation.tick(150);
    draw();
    this.fit(nodes, width, height);
    simulation.alpha(0.1).on("tick", draw);
  }

  /** Zoom so that all nodes (with room for their labels) fit into the canvas. */
  private fit(nodes: Node[], width: number, height: number): void {
    if (nodes.length === 0) return;
    const xs = nodes.map((n) => n.x ?? 0);
    const ys = nodes.map((n) => n.y ?? 0);
    const [minX, maxX] = [Math.min(...xs) - 30, Math.max(...xs) + 150]; // labels extend to the right
    const [minY, maxY] = [Math.min(...ys) - 30, Math.max(...ys) + 30];
    const scale = Math.min(1.15, 0.92 * Math.min(width / (maxX - minX), height / (maxY - minY)));
    const transform = zoomIdentity
      .translate(width / 2, height / 2)
      .scale(scale)
      .translate(-(minX + maxX) / 2, -(minY + maxY) / 2);
    this.svg.call(this.zoomer.transform, transform);
  }

  private size(): { width: number; height: number } {
    const rect = (this.svg.node() as SVGSVGElement).getBoundingClientRect();
    return { width: rect.width || 800, height: rect.height || 600 };
  }
}

/** Line (or curve, for a second relation of the same pair) that ends at the target's border, not its center. */
function linkPath(d: Link): string {
  const sx = d.source.x ?? 0;
  const sy = d.source.y ?? 0;
  const tx = d.target.x ?? 0;
  const ty = d.target.y ?? 0;
  const dx = tx - sx;
  const dy = ty - sy;
  const length = Math.hypot(dx, dy) || 1;
  const ex = tx - (dx / length) * (d.target.radius + 2);
  const ey = ty - (dy / length) * (d.target.radius + 2);
  if (d.bend === 0) return `M${sx},${sy}L${ex},${ey}`;
  const cx = (sx + tx) / 2 - (dy / length) * d.bend;
  const cy = (sy + ty) / 2 + (dx / length) * d.bend;
  return `M${sx},${sy}Q${cx},${cy} ${ex},${ey}`;
}
