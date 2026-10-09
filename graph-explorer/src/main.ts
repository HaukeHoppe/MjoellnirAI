import "./style.css";
import { CausalGraph } from "./graph";
import { parseGraphData } from "./parse";
import type { Chain, Concept, Direction, Evidence, Sign, Step } from "./types";
import { GraphView, type Scene } from "./view";

/** Model id of the RAG pipeline in Open WebUI (see data/apply_start_suggestions.py). */
const CHAT_MODEL = "capital_rag_pipeline";
const MAX_HOPS = 4;

/** Start-screen examples; only those with a chain in the current export are shown. */
const EXAMPLES: [string, string][] = [
  ["Leitzins", "Bond Price"],
  ["Oil Price", "Inflation"],
  ["Interest Rates", "Exchange Rate"],
  ["Inflation", "Credit Spread"],
  ["Stock Price", "Recession"],
  ["Liquidity", "Interest Rates"],
  ["Economic Growth", "Credit Risk"],
];

const HORIZON: Record<string, string> = {
  immediate: "sofort",
  "short-term": "kurzfristig",
  "medium-term": "mittelfristig",
  "long-term": "langfristig",
};

const ARROW: Record<Direction, { symbol: string; text: string }> = {
  positive: { symbol: "→ +", text: "fördert" },
  negative: { symbol: "→ −", text: "dämpft" },
  ambiguous: { symbol: "→ ?", text: "uneindeutig" },
  "non-linear": { symbol: "→ ~", text: "nichtlinear" },
};

// ---- tiny DOM helper: builds elements with textContent only, so text from the data is never parsed as HTML ----

type Child = Node | string | null | undefined | false;

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | boolean | ((event: Event) => void)> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (typeof value === "function") element.addEventListener(key.replace(/^on/, ""), value);
    else if (value === true) element.setAttribute(key, "");
    else if (value !== false) element.setAttribute(key, value);
  }
  for (const child of children) if (child) element.append(child);
  return element;
}

function byId<T extends HTMLElement | SVGSVGElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`#${id} missing in index.html`);
  return element as T;
}

// ---- state, mirrored in the URL hash (#von=…&nach=…&konzept=…) so a view can be shared ----

interface State {
  from?: string;
  to?: string;
  focus?: string;
  chainIndex: number;
}

function readHash(graph: CausalGraph): State {
  const params = new URLSearchParams(location.hash.slice(1));
  const known = (key: string) => {
    const id = params.get(key);
    return id && graph.has(id) ? id : undefined;
  };
  const state: State = { chainIndex: 0 };
  const from = known("von");
  const to = known("nach");
  const focus = known("konzept");
  if (from) state.from = from;
  if (to) state.to = to;
  if (focus) state.focus = focus;
  return state;
}

function writeHash(state: State): void {
  const params = new URLSearchParams();
  if (state.from) params.set("von", state.from);
  if (state.to) params.set("nach", state.to);
  if (state.focus) params.set("konzept", state.focus);
  const hash = params.toString();
  history.replaceState(null, "", hash ? `#${hash}` : location.pathname + location.search);
}

// ---- concept picker with keyboard navigation ----

class Picker {
  private results: Concept[] = [];
  private active = -1;
  selected?: string;

  constructor(
    private readonly input: HTMLInputElement,
    private readonly list: HTMLUListElement,
    private readonly graph: CausalGraph,
    private readonly onPick: () => void,
  ) {
    input.addEventListener("input", () => {
      delete this.selected;
      this.update();
    });
    input.addEventListener("focus", () => this.update());
    input.addEventListener("blur", () => setTimeout(() => this.close(), 150));
    input.addEventListener("keydown", (event) => this.key(event));
  }

  set(id: string | undefined): void {
    if (id) this.selected = id;
    else delete this.selected;
    this.input.value = id ? this.graph.label(id) : "";
    this.close();
  }

  /** The picked concept, or the best match for what was typed. */
  resolve(): string | undefined {
    if (this.selected) return this.selected;
    const best = this.graph.search(this.input.value, 1)[0];
    if (best) this.set(best.id);
    return best?.id;
  }

  private update(): void {
    this.results = this.input.value.trim() ? this.graph.search(this.input.value, 8) : [];
    this.active = this.results.length > 0 ? 0 : -1;
    this.draw();
  }

  private draw(): void {
    this.list.replaceChildren(
      ...this.results.map((concept, i) =>
        h(
          "li",
          { role: "option", "aria-selected": String(i === this.active), onmousedown: (e) => { e.preventDefault(); this.pick(i); } },
          h("span", {}, concept.label),
          h("small", {}, `${concept.label !== concept.id ? `${concept.id} · ` : ""}${this.graph.degree(concept.id)} Rel.`),
        ),
      ),
    );
    const open = this.results.length > 0;
    this.list.hidden = !open;
    this.input.setAttribute("aria-expanded", String(open));
  }

  private close(): void {
    this.list.hidden = true;
    this.input.setAttribute("aria-expanded", "false");
  }

  private pick(i: number): void {
    const concept = this.results[i];
    if (!concept) return;
    this.set(concept.id);
    this.onPick();
  }

  private key(event: KeyboardEvent): void {
    if (this.list.hidden || this.results.length === 0) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      this.active = (this.active + step + this.results.length) % this.results.length;
      this.draw();
    } else if (event.key === "Enter" && this.active >= 0) {
      event.preventDefault();
      this.pick(this.active);
    } else if (event.key === "Escape") {
      this.close();
    }
  }
}

// ---- app ----

class App {
  private state: State;
  private readonly view: GraphView;
  private readonly fromPicker: Picker;
  private readonly toPicker: Picker;
  private readonly results = byId<HTMLElement>("results");
  private chains: Chain[] = [];

  constructor(
    private readonly graph: CausalGraph,
    private readonly chatUrl: URL,
  ) {
    this.view = new GraphView(byId<SVGSVGElement>("graph"), {
      label: (id) => graph.label(id),
      degree: (id) => graph.degree(id),
      onSelect: (id) => this.update({ focus: id }),
    });
    this.fromPicker = new Picker(byId("from"), byId("from-list"), graph, () => this.pickersChanged());
    this.toPicker = new Picker(byId("to"), byId("to-list"), graph, () => this.pickersChanged());

    byId<HTMLFormElement>("chain-form").addEventListener("submit", (event) => {
      event.preventDefault();
      this.pickersChanged();
    });
    byId("swap").addEventListener("click", () => this.update({ from: this.state.to, to: this.state.from }));
    window.addEventListener("hashchange", () => this.apply(readHash(graph)));
    let resizeTimer = 0;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => this.draw(), 200);
    });

    this.state = readHash(graph);
    this.apply(this.state);
  }

  private pickersChanged(): void {
    this.update({ from: this.fromPicker.resolve(), to: this.toPicker.resolve(), focus: undefined });
  }

  private update(change: Partial<Record<"from" | "to" | "focus", string | undefined>> & { chainIndex?: number }): void {
    const next: State = { ...this.state };
    for (const key of ["from", "to", "focus"] as const) {
      if (!(key in change)) continue;
      const value = change[key];
      if (value) next[key] = value;
      else delete next[key];
    }
    // A new cause or effect searches again; focusing a concept keeps the chains and the selected one.
    const pairChanged = next.from !== this.state.from || next.to !== this.state.to;
    if (pairChanged) {
      this.chains = next.from && next.to ? this.graph.findChains(next.from, next.to, { maxHops: MAX_HOPS }) : [];
    }
    next.chainIndex = change.chainIndex ?? (pairChanged ? 0 : this.state.chainIndex);
    this.state = next;
    writeHash(next);
    this.render();
  }

  private apply(state: State): void {
    this.state = state;
    this.chains = state.from && state.to ? this.graph.findChains(state.from, state.to, { maxHops: MAX_HOPS }) : [];
    this.render();
  }

  private render(): void {
    this.fromPicker.set(this.state.from);
    this.toPicker.set(this.state.to);
    const { from, to, focus } = this.state;
    const sections: Child[] = [];
    if (focus) sections.push(this.conceptCard(focus));
    if (from && to) sections.push(...this.chainResults(from, to));
    else if (from) sections.push(...this.onward(from));
    else if (to) sections.push(...this.upstream(to));
    else if (!focus) sections.push(...this.intro());
    this.results.replaceChildren(...sections.filter((s): s is Node => s instanceof Node));
    this.draw();
  }

  private draw(): void {
    const chain = this.chains[this.state.chainIndex];
    const { from, to, focus } = this.state;
    let scene: Scene;
    if (chain) {
      scene = { ...this.graph.neighborhood(chain.concepts, 1, 32), chain: chain.concepts };
    } else {
      const seeds = [focus, from, to].filter((id): id is string => Boolean(id));
      const start = seeds.length > 0 ? seeds : [this.topConcepts(1)[0]?.id ?? ""];
      scene = this.graph.neighborhood(start, seeds.length > 1 ? 1 : 2, 40);
    }
    if (focus) scene.focus = focus;
    this.view.render(scene);
  }

  // ---- result sections ----

  private intro(): Child[] {
    const examples = EXAMPLES.filter(([a, b]) => this.graph.findChains(a, b, { maxHops: MAX_HOPS, limit: 1 }).length > 0);
    return [
      h(
        "p",
        {},
        "Wähle eine Ursache und eine Wirkung. Der Explorer sucht Ketten mit bis zu vier Schritten durch das Netz der Ursache-Wirkungs-Beziehungen aus den Quellen und zeigt zu jedem Schritt den belegenden Satz.",
      ),
      examples.length > 0 && h("h2", {}, "Beispiele"),
      examples.length > 0 &&
        h(
          "div",
          { class: "chips" },
          ...examples.map(([a, b]) =>
            h("button", { class: "chip", type: "button", onclick: () => this.update({ from: a, to: b, focus: undefined }) }, `${this.graph.label(a)} → ${this.graph.label(b)}`),
          ),
        ),
      h("h2", {}, "Gut vernetzte Konzepte"),
      this.conceptChips(this.topConcepts(14).map((c) => c.id), (id) => this.update({ focus: id })),
    ];
  }

  private chainResults(from: string, to: string): Child[] {
    const a = this.graph.label(from);
    const b = this.graph.label(to);
    if (this.chains.length === 0) {
      const onward = this.graph.reachable(from, "forward", MAX_HOPS).slice(0, 12);
      const upstream = this.graph.reachable(to, "backward", MAX_HOPS).slice(0, 12);
      return [
        h("p", {}, `Keine Kette von „${a}“ nach „${b}“ mit höchstens ${MAX_HOPS} Schritten in den Quellen.`),
        onward.length > 0 && h("h2", {}, `Wirkungen von „${a}“`),
        onward.length > 0 && this.reachedChips(onward, (id) => this.update({ to: id, focus: undefined })),
        upstream.length > 0 && h("h2", {}, `Ursachen von „${b}“`),
        upstream.length > 0 && this.reachedChips(upstream, (id) => this.update({ from: id, focus: undefined })),
        this.askButton(`Wie wirkt sich eine Veränderung von „${a}“ auf „${b}“ aus?`),
      ];
    }
    return [
      h("h2", {}, this.chains.length === 1 ? "1 Kette gefunden" : `${this.chains.length} Ketten gefunden`),
      ...this.chains.map((chain, i) => this.chainCard(chain, i)),
    ];
  }

  private chainCard(chain: Chain, index: number): HTMLElement {
    const current = index === this.state.chainIndex;
    const sources = new Set(chain.steps.flatMap((s) => s.relations.flatMap((r) => r.evidence.map((e) => e.chunk)))).size;
    const via = chain.concepts.slice(1, -1).map((id) => `„${this.graph.label(id)}“`);
    const first = this.graph.label(chain.concepts[0] ?? "");
    const last = this.graph.label(chain.concepts[chain.concepts.length - 1] ?? "");
    return h(
      "article",
      { class: "chain", "aria-current": String(current) },
      h(
        "button",
        { class: "chain-head", type: "button", onclick: () => this.update({ chainIndex: index }), "aria-expanded": String(current) },
        h("strong", {}, `${chain.steps.length} ${chain.steps.length === 1 ? "Schritt" : "Schritte"} · ${sources} ${sources === 1 ? "Beleg" : "Belege"}`),
        this.netBadge(chain.sign, first, last),
      ),
      current &&
        h(
          "ol",
          { class: "steps" },
          ...chain.steps.map((step) => this.stepItem(step)),
          h(
            "li",
            {},
            this.askButton(
              `Wie wirkt sich eine Veränderung von „${first}“ auf „${last}“ aus${via.length > 0 ? `, etwa über ${via.join(" und ")}` : ""}? Erkläre die Wirkungskette Schritt für Schritt.`,
            ),
          ),
        ),
    );
  }

  private netBadge(sign: Sign, from: string, to: string): HTMLElement {
    if (sign === 1) return h("span", { class: "net up", title: `Mehr ${from} führt zu mehr ${to}` }, "gleichläufig ↑↑");
    if (sign === -1) return h("span", { class: "net down", title: `Mehr ${from} führt zu weniger ${to}` }, "gegenläufig ↑↓");
    return h("span", { class: "net unclear", title: "Mindestens ein Schritt hat keine eindeutige Richtung" }, "Richtung offen");
  }

  private stepItem(step: Step): HTMLElement {
    const directions = [...new Set(step.relations.map((r) => r.direction))];
    const evidence = step.relations.flatMap((r) => r.evidence.map((e) => ({ e, direction: r.direction })));
    return h(
      "li",
      { class: "step" },
      h(
        "div",
        { class: "step-line" },
        this.conceptLink(step.from),
        ...directions.map((d) => h("span", { class: `arrow ${d}`, title: ARROW[d].text }, ARROW[d].symbol)),
        this.conceptLink(step.to),
      ),
      h(
        "details",
        { class: "evidence" },
        h("summary", {}, evidence.length === 1 ? "Beleg anzeigen" : `${evidence.length} Belege anzeigen`),
        ...evidence.map(({ e, direction }) => this.evidenceItem(e, direction)),
      ),
    );
  }

  private evidenceItem(evidence: Evidence, direction: Direction): HTMLElement {
    const chunk = this.graph.data.chunks[String(evidence.chunk)];
    const horizon = HORIZON[evidence.horizon];
    return h(
      "div",
      {},
      h("blockquote", {}, evidence.mechanism),
      evidence.conditions && h("div", { class: "meta" }, `Bedingung: ${evidence.conditions}`),
      h(
        "div",
        { class: "meta" },
        h("span", { class: "badge" }, evidence.timeless ? "zeitlos" : "zeitgebunden"),
        " ",
        h("span", { class: "badge" }, ARROW[direction].text),
        horizon && " ",
        horizon && h("span", { class: "badge" }, horizon),
        " ",
        chunk?.url ? h("a", { href: chunk.url, target: "_blank", rel: "noopener" }, chunk.source) : (chunk?.source ?? "Quelle unbekannt"),
        chunk && ` – ${chunk.title}`,
      ),
    );
  }

  private conceptCard(id: string): HTMLElement {
    const concept = this.graph.concepts.get(id);
    const effects = this.graph.steps(id, "forward");
    const causes = this.graph.steps(id, "backward");
    return h(
      "section",
      { class: "concept-card" },
      h("h3", {}, this.graph.label(id)),
      h("p", { class: "muted" }, `${concept?.label !== id ? `${id} · ` : ""}in ${concept?.chunks ?? 0} Textabschnitten · ${effects.length} Wirkungen · ${causes.length} Ursachen`),
      h(
        "div",
        { class: "actions" },
        h("button", { class: "button", type: "button", onclick: () => this.update({ from: id, focus: undefined }) }, "Als Ursache"),
        h("button", { class: "button", type: "button", onclick: () => this.update({ to: id, focus: undefined }) }, "Als Wirkung"),
        h("button", { class: "button ghost", type: "button", onclick: () => this.update({ focus: undefined }) }, "Schließen"),
      ),
      effects.length > 0 && h("h2", {}, "Wirkt auf"),
      effects.length > 0 && this.stepChips(effects, "forward"),
      causes.length > 0 && h("h2", {}, "Wird beeinflusst von"),
      causes.length > 0 && this.stepChips(causes, "backward"),
    );
  }

  private onward(from: string): Child[] {
    const reached = this.graph.reachable(from, "forward", MAX_HOPS).slice(0, 16);
    return [
      h("p", {}, `Wähle eine Wirkung, oder nimm eine der Folgen von „${this.graph.label(from)}“:`),
      this.reachedChips(reached, (id) => this.update({ to: id })),
      this.askButton(`Welche Folgen hat eine Veränderung von „${this.graph.label(from)}“ am Kapitalmarkt?`),
    ];
  }

  private upstream(to: string): Child[] {
    const reached = this.graph.reachable(to, "backward", MAX_HOPS).slice(0, 16);
    return [
      h("p", {}, `Wähle eine Ursache, oder nimm einen der Auslöser von „${this.graph.label(to)}“:`),
      this.reachedChips(reached, (id) => this.update({ from: id })),
    ];
  }

  // ---- small building blocks ----

  private conceptLink(id: string): HTMLElement {
    return h("button", { class: "concept-link", type: "button", onclick: () => this.update({ focus: id }) }, this.graph.label(id));
  }

  private conceptChips(ids: string[], onClick: (id: string) => void): HTMLElement {
    return h("div", { class: "chips" }, ...ids.map((id) => h("button", { class: "chip", type: "button", onclick: () => onClick(id) }, this.graph.label(id))));
  }

  private reachedChips(reached: { id: string; hops: number }[], onClick: (id: string) => void): HTMLElement {
    return h(
      "div",
      { class: "chips" },
      ...reached.map((r) =>
        h("button", { class: "chip", type: "button", onclick: () => onClick(r.id) }, this.graph.label(r.id), " ", h("small", {}, r.hops === 1 ? "direkt" : `${r.hops} Schritte`)),
      ),
    );
  }

  private stepChips(steps: Step[], way: "forward" | "backward"): HTMLElement {
    const sorted = [...steps].sort((a, b) => weight(b) - weight(a)).slice(0, 20);
    return h(
      "div",
      { class: "chips" },
      ...sorted.map((step) => {
        const other = way === "forward" ? step.to : step.from;
        const direction = step.relations[0]?.direction ?? "ambiguous";
        return h(
          "button",
          { class: "chip", type: "button", title: ARROW[direction].text, onclick: () => this.update({ focus: other }) },
          h("span", { class: `arrow ${direction}` }, direction === "positive" ? "+ " : direction === "negative" ? "− " : "? "),
          this.graph.label(other),
        );
      }),
    );
  }

  private askButton(question: string): HTMLElement {
    const url = new URL(this.chatUrl);
    url.searchParams.set("models", CHAT_MODEL);
    url.searchParams.set("q", question);
    return h("a", { class: "button ask", href: url.toString(), target: "_blank", rel: "noopener", title: question }, "Im Chat fragen ↗");
  }

  private topConcepts(n: number): Concept[] {
    return [...this.graph.concepts.values()].sort((a, b) => this.graph.degree(b.id) - this.graph.degree(a.id)).slice(0, n);
  }
}

function weight(step: Step): number {
  return step.relations.reduce((sum, r) => sum + r.weight, 0);
}

async function loadChatUrl(): Promise<URL> {
  try {
    const response = await fetch("config.json", { cache: "no-cache" });
    const config = (await response.json()) as { chatUrl?: unknown };
    if (typeof config.chatUrl === "string" && config.chatUrl) return new URL(config.chatUrl, location.href);
  } catch {
    // No config (e.g. a plain static server): the chat is assumed at the site root.
  }
  return new URL("/", location.href);
}

async function main(): Promise<void> {
  const results = byId<HTMLElement>("results");
  results.replaceChildren(h("p", { class: "muted" }, "Lade Konzeptgraph …"));
  try {
    const [chatUrl, response] = await Promise.all([loadChatUrl(), fetch("data/graph.json", { cache: "no-cache" })]);
    if (!response.ok) throw new Error(`data/graph.json: HTTP ${response.status}. Wurde data/export_explorer_graph.py ausgeführt?`);
    const graph = new CausalGraph(parseGraphData(await response.json()));

    byId<HTMLAnchorElement>("chat-link").href = chatUrl.toString();
    for (const link of document.querySelectorAll<HTMLAnchorElement>("a[data-legal]")) {
      link.href = new URL(link.getAttribute("href") ?? "", chatUrl).toString();
    }
    const generated = new Date(graph.data.generated).toLocaleDateString("de-DE");
    byId("stats").textContent =
      `${graph.concepts.size.toLocaleString("de-DE")} Konzepte · ${graph.data.edges.length.toLocaleString("de-DE")} Beziehungen · Stand ${generated}`;

    new App(graph, chatUrl);
  } catch (error) {
    console.error(error);
    results.replaceChildren(h("div", { class: "error" }, `Der Konzeptgraph konnte nicht geladen werden. ${error instanceof Error ? error.message : String(error)}`));
  }
}

void main();
