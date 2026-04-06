/**
 * 2D split visualization for the "split objective" slide.
 * Draws an SVG scatter of treated/control points with a candidate split line.
 */
(function () {
  "use strict";

  var CONTAINER_ID = "split-visual-container";
  var W = 300, H = 300;
  var PAD = 36;
  var ACCENT = "#2a9d8f";
  var DANGER = "#e76f51";
  var ORANGE = "#e67e22";
  var DARK = "#1a1a2e";
  var FONT = "Inter, system-ui, sans-serif";

  var SPLIT_X = 0.48;

  var points = [
    {x:0.10,y:0.22,t:1},{x:0.14,y:0.68,t:0},{x:0.08,y:0.85,t:1},
    {x:0.22,y:0.40,t:0},{x:0.18,y:0.55,t:1},{x:0.30,y:0.15,t:0},
    {x:0.25,y:0.78,t:1},{x:0.35,y:0.50,t:0},{x:0.12,y:0.35,t:1},
    {x:0.38,y:0.28,t:1},{x:0.32,y:0.90,t:0},{x:0.42,y:0.62,t:1},
    {x:0.28,y:0.10,t:0},{x:0.40,y:0.45,t:0},{x:0.20,y:0.92,t:1},
    {x:0.55,y:0.30,t:1},{x:0.60,y:0.72,t:0},{x:0.65,y:0.18,t:1},
    {x:0.58,y:0.55,t:0},{x:0.72,y:0.42,t:1},{x:0.68,y:0.82,t:0},
    {x:0.75,y:0.60,t:1},{x:0.80,y:0.25,t:0},{x:0.62,y:0.90,t:1},
    {x:0.85,y:0.48,t:1},{x:0.78,y:0.70,t:0},{x:0.90,y:0.35,t:1},
    {x:0.88,y:0.85,t:0},{x:0.52,y:0.15,t:0},{x:0.70,y:0.50,t:1},
  ];

  function sx(v) { return PAD + v * (W - 2 * PAD); }
  function sy(v) { return (H - PAD) - v * (H - 2 * PAD); }

  function build(container) {
    container.innerHTML = "";
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.style.display = "block";

    var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", PAD);
    rect.setAttribute("y", PAD - 10);
    rect.setAttribute("width", W - 2 * PAD);
    rect.setAttribute("height", H - 2 * PAD + 10);
    rect.setAttribute("fill", "#fafbfc");
    rect.setAttribute("stroke", "rgba(26,26,46,0.1)");
    rect.setAttribute("rx", "6");
    svg.appendChild(rect);

    var splitLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    splitLine.setAttribute("x1", sx(SPLIT_X));
    splitLine.setAttribute("y1", PAD - 10);
    splitLine.setAttribute("x2", sx(SPLIT_X));
    splitLine.setAttribute("y2", H - PAD);
    splitLine.setAttribute("stroke", DANGER);
    splitLine.setAttribute("stroke-width", "2.5");
    splitLine.setAttribute("stroke-dasharray", "7 4");
    svg.appendChild(splitLine);

    var leftBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    leftBg.setAttribute("x", PAD);
    leftBg.setAttribute("y", PAD - 10);
    leftBg.setAttribute("width", sx(SPLIT_X) - PAD);
    leftBg.setAttribute("height", H - 2 * PAD + 10);
    leftBg.setAttribute("fill", "rgba(42,157,143,0.05)");
    leftBg.setAttribute("rx", "6");
    svg.appendChild(leftBg);

    var rightBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rightBg.setAttribute("x", sx(SPLIT_X));
    rightBg.setAttribute("y", PAD - 10);
    rightBg.setAttribute("width", (W - PAD) - sx(SPLIT_X));
    rightBg.setAttribute("height", H - 2 * PAD + 10);
    rightBg.setAttribute("fill", "rgba(231,111,81,0.04)");
    rightBg.setAttribute("rx", "0");
    svg.appendChild(rightBg);

    points.forEach(function (p) {
      var c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", sx(p.x));
      c.setAttribute("cy", sy(p.y));
      c.setAttribute("r", "5");
      c.setAttribute("fill", p.t ? ORANGE : DARK);
      c.setAttribute("opacity", "0.82");
      svg.appendChild(c);
    });

    function addLabel(text, x, y, size, anchor, weight, color) {
      var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", x);
      t.setAttribute("y", y);
      t.setAttribute("text-anchor", anchor || "middle");
      t.setAttribute("font-family", FONT);
      t.setAttribute("font-size", size || "11");
      t.setAttribute("font-weight", weight || "400");
      t.setAttribute("fill", color || "#4a4a68");
      t.textContent = text;
      svg.appendChild(t);
    }

    addLabel("X\u2081", W / 2, H - 4, "12", "middle", "600", DARK);
    addLabel("X\u2082", 8, (H - PAD + PAD - 10) / 2 + 4, "12", "middle", "600", DARK);

    addLabel("S\u2097", (PAD + sx(SPLIT_X)) / 2, PAD - 16, "14", "middle", "700", ACCENT);
    addLabel("S\u1D63", (sx(SPLIT_X) + W - PAD) / 2, PAD - 16, "14", "middle", "700", DANGER);

    addLabel("X\u2081 \u2264 t", sx(SPLIT_X), H - PAD + 14, "10", "middle", "600", DANGER);

    container.appendChild(svg);
  }

  function init() {
    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;
    build(container);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
