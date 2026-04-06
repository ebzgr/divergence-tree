/**
 * Animated tree-growth diagram for the "How the tree grows" slide.
 * Advances one step per click (like reveal.js fragments).
 */
(function () {
  "use strict";

  var CONTAINER_ID = "tree-anim-container";
  var W = 440, H = 260;
  var NODE_RX = 10, NODE_H = 34;
  var ACCENT = "#2a9d8f";
  var DANGER = "#e76f51";
  var MUTED = "rgba(26,26,46,0.12)";
  var FONT = "Inter, system-ui, sans-serif";

  var nodes = [
    { id: "root", label: "All data",  x: 220, y: 24,  w: 100, fill: "#e8f8f5", stroke: ACCENT },
    { id: "A",    label: "Leaf A",    x: 100, y: 115, w: 80,  fill: "#fff",    stroke: MUTED },
    { id: "B",    label: "Leaf B",    x: 340, y: 115, w: 80,  fill: "#fff",    stroke: MUTED },
    { id: "BL",   label: "B\u2097",   x: 275, y: 210, w: 56,  fill: "#f0fff4", stroke: ACCENT, dash: true },
    { id: "BR",   label: "B\u1D63",   x: 405, y: 210, w: 56,  fill: "#f0fff4", stroke: ACCENT, dash: true },
  ];

  var edges = [
    { from: "root", to: "A" },
    { from: "root", to: "B" },
    { from: "B",    to: "BL" },
    { from: "B",    to: "BR" },
  ];

  var steps = [
    { show: ["root"],                         edges: [],                              highlight: null },
    { show: ["root", "A", "B"],               edges: ["root-A", "root-B"],            highlight: "B"  },
    { show: ["root", "A", "B", "BL", "BR"],   edges: ["root-A", "root-B", "B-BL", "B-BR"], highlight: "BL" },
  ];

  var stepLabels = [
    "All data in one node",
    "Best g at root \u2192 split into A, B",
    "Best g at B \u2192 split into B\u2097, B\u1D63"
  ];

  function nById(id) { return nodes.find(function (n) { return n.id === id; }); }
  function edgeId(e) { return e.from + "-" + e.to; }

  var currentStep = 0;
  var svg = null;
  var slideEl = null;

  function build(container) {
    container.innerHTML = "";
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.style.display = "block";
    svg.style.cursor = "pointer";

    var edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    svg.appendChild(edgeGroup);
    var nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    svg.appendChild(nodeGroup);

    edges.forEach(function (e) {
      var a = nById(e.from), b = nById(e.to);
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", a.x);
      line.setAttribute("y1", a.y + NODE_H);
      line.setAttribute("x2", b.x);
      line.setAttribute("y2", b.y);
      line.setAttribute("stroke", "#aab");
      line.setAttribute("stroke-width", "2");
      line.setAttribute("data-edge", edgeId(e));
      line.style.opacity = "0";
      line.style.transition = "opacity 0.45s ease";
      edgeGroup.appendChild(line);
    });

    nodes.forEach(function (n) {
      var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("data-node", n.id);
      g.style.opacity = "0";
      g.style.transition = "opacity 0.45s ease";

      var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", n.x - n.w / 2);
      rect.setAttribute("y", n.y);
      rect.setAttribute("width", n.w);
      rect.setAttribute("height", NODE_H);
      rect.setAttribute("rx", NODE_RX);
      rect.setAttribute("fill", n.fill);
      rect.setAttribute("stroke", n.stroke);
      rect.setAttribute("stroke-width", "2");
      if (n.dash) rect.setAttribute("stroke-dasharray", "6 3");
      g.appendChild(rect);

      var text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", n.x);
      text.setAttribute("y", n.y + NODE_H / 2 + 5);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("font-family", FONT);
      text.setAttribute("font-size", "12");
      text.setAttribute("font-weight", "600");
      text.setAttribute("fill", "#1a1a2e");
      text.textContent = n.label;
      g.appendChild(text);

      nodeGroup.appendChild(g);
    });

    var noteEl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    noteEl.setAttribute("x", W / 2);
    noteEl.setAttribute("y", H - 4);
    noteEl.setAttribute("text-anchor", "middle");
    noteEl.setAttribute("font-family", FONT);
    noteEl.setAttribute("font-size", "11");
    noteEl.setAttribute("font-style", "italic");
    noteEl.setAttribute("fill", "#4a4a68");
    noteEl.setAttribute("data-role", "note");
    svg.appendChild(noteEl);

    var hint = document.createElementNS("http://www.w3.org/2000/svg", "text");
    hint.setAttribute("x", W / 2);
    hint.setAttribute("y", H - 18);
    hint.setAttribute("text-anchor", "middle");
    hint.setAttribute("font-family", FONT);
    hint.setAttribute("font-size", "9");
    hint.setAttribute("fill", "#999");
    hint.setAttribute("data-role", "hint");
    hint.textContent = "click to advance";
    svg.appendChild(hint);

    container.appendChild(svg);
    return svg;
  }

  function renderStep() {
    if (!svg) return;
    var step = steps[currentStep];

    nodes.forEach(function (n) {
      var el = svg.querySelector('[data-node="' + n.id + '"]');
      el.style.opacity = step.show.indexOf(n.id) >= 0 ? "1" : "0";
      var rect = el.querySelector("rect");
      if (step.highlight === n.id) {
        rect.setAttribute("stroke", DANGER);
        rect.setAttribute("stroke-width", "3");
        rect.setAttribute("fill", "#fff5f5");
      } else {
        rect.setAttribute("stroke", n.stroke);
        rect.setAttribute("stroke-width", "2");
        rect.setAttribute("fill", n.fill);
      }
    });

    edges.forEach(function (e) {
      var el = svg.querySelector('[data-edge="' + edgeId(e) + '"]');
      el.style.opacity = step.edges.indexOf(edgeId(e)) >= 0 ? "1" : "0";
    });

    var noteEl = svg.querySelector('[data-role="note"]');
    noteEl.textContent = stepLabels[currentStep];

    var hint = svg.querySelector('[data-role="hint"]');
    if (hint) hint.style.opacity = currentStep < steps.length - 1 ? "1" : "0";
  }

  function advance() {
    if (currentStep < steps.length - 1) {
      currentStep++;
      renderStep();
    }
  }

  function resetToStart() {
    currentStep = 0;
    renderStep();
  }

  function init() {
    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    build(container);
    renderStep();

    svg.addEventListener("click", function (e) {
      e.stopPropagation();
      advance();
    });

    slideEl = container.closest("section");

    if (window.Reveal) {
      Reveal.on("slidechanged", function (ev) {
        if (ev.currentSlide === slideEl) {
          resetToStart();
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
