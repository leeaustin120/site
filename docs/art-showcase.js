/* ================================================================
   ART SHOWCASE — interactive edge detection demo
   Placed below the "Classifying Art Styles" project entry.

   Click-cycle per card:
     State 0  Raw image               hint: "click to explore"
     State 1  Canny-style edge map    hint: "click for genre"
     State 2  Image + genre label     hint: "click for next"
     click →  New image, new genre,   back to state 0

   Edge detection: grayscale → 5×5 Gaussian blur → Sobel → threshold.
   Processed at low resolution (PROCESS_WIDTH) for blocky pixel look.
   ================================================================ */

(function () {
  'use strict';

  // ── Tuneable constants ──────────────────────────────────────
  var PROCESS_SIZE = 256;  // matches Python notebook: pad-to-square then resize 256×256
  var CANNY_LO     = 50;   // cv2.Canny low threshold
  var CANNY_HI     = 150;  // cv2.Canny high threshold

  // ── Genre label ────────────────────────────────────────────
  function formatGenre(folder) {
    if (folder === 'Ukiyo_e') return 'Ukiyo-e';
    return folder.replace(/_/g, ' ');
  }

  function genreFromPath(path) {
    return path.split('/')[1] || '';   // wikiart/{genre}/file.jpg
  }

  // ── Random helpers ─────────────────────────────────────────
  function pickRandom(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  // Build genre → paths map once
  var genreMap = {};

  function buildGenreMap() {
    if (typeof WIKIART_IMAGES === 'undefined' || !WIKIART_IMAGES.length) return;
    WIKIART_IMAGES.forEach(function (p) {
      var g = genreFromPath(p);
      if (!genreMap[g]) genreMap[g] = [];
      genreMap[g].push(p);
    });
  }

  // Active paths across all three cards (avoid repeats)
  var activePaths = [];

  function pickNextImage(excludeGenre) {
    var genres = Object.keys(genreMap);
    // Prefer a different genre
    var altGenres = genres.filter(function (g) { return g !== excludeGenre; });
    var pool = [];
    (altGenres.length ? altGenres : genres).forEach(function (g) {
      genreMap[g].forEach(function (p) {
        if (activePaths.indexOf(p) === -1) pool.push(p);
      });
    });
    if (!pool.length) {
      // All paths exhausted — allow repeats
      pool = WIKIART_IMAGES.filter(function (p) {
        return activePaths.indexOf(p) === -1;
      });
    }
    if (!pool.length) pool = WIKIART_IMAGES;
    return pickRandom(pool);
  }

  // ── Edge detection ─────────────────────────────────────────

  function toGrayscale(data) {
    var gray = new Uint8ClampedArray(data.length / 4);
    for (var i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = (0.299 * data[i]) + (0.587 * data[i + 1]) + (0.114 * data[i + 2]);
    }
    return gray;
  }

  var GAUSS5 = [
     1,  4,  6,  4, 1,
     4, 16, 24, 16, 4,
     6, 24, 36, 24, 6,
     4, 16, 24, 16, 4,
     1,  4,  6,  4, 1,
  ];
  var GAUSS5_SUM = 256;

  function gaussianBlur(gray, w, h) {
    var out = new Uint8ClampedArray(gray.length);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var acc = 0;
        for (var ky = -2; ky <= 2; ky++) {
          for (var kx = -2; kx <= 2; kx++) {
            var ny = Math.min(Math.max(y + ky, 0), h - 1);
            var nx = Math.min(Math.max(x + kx, 0), w - 1);
            acc += gray[ny * w + nx] * GAUSS5[(ky + 2) * 5 + (kx + 2)];
          }
        }
        out[y * w + x] = acc / GAUSS5_SUM;
      }
    }
    return out;
  }

  // Sobel gradients → magnitude (Float32) + quantised direction (4 bins).
  function sobelGradients(b, W, H) {
    var mag = new Float32Array(W * H);
    var dir = new Uint8Array(W * H);
    for (var y = 1; y < H - 1; y++) {
      for (var x = 1; x < W - 1; x++) {
        var gx =
          -b[(y-1)*W+(x-1)] + b[(y-1)*W+(x+1)] +
          -2*b[y*W+(x-1)]   + 2*b[y*W+(x+1)] +
          -b[(y+1)*W+(x-1)] + b[(y+1)*W+(x+1)];
        var gy =
          -b[(y-1)*W+(x-1)] - 2*b[(y-1)*W+x] - b[(y-1)*W+(x+1)] +
           b[(y+1)*W+(x-1)] + 2*b[(y+1)*W+x] + b[(y+1)*W+(x+1)];
        mag[y*W+x] = Math.sqrt(gx*gx + gy*gy);
        var ang = Math.atan2(gy, gx) * 180 / Math.PI;
        if (ang < 0) ang += 180;
        dir[y*W+x] = ang < 22.5 ? 0 : ang < 67.5 ? 1 : ang < 112.5 ? 2 : ang < 157.5 ? 3 : 0;
      }
    }
    return { mag: mag, dir: dir };
  }

  // Non-maximum suppression: keep only local maxima along gradient direction.
  var NMS_NEIGHBORS = [
    [ 0, -1,  0,  1],   // dir 0 — horizontal: compare W, E
    [-1, -1,  1,  1],   // dir 1 — NE-SW:      compare NW, SE
    [-1,  0,  1,  0],   // dir 2 — vertical:   compare N, S
    [-1,  1,  1, -1],   // dir 3 — NW-SE:      compare NE, SW
  ];
  function nonMaxSuppression(mag, dir, W, H) {
    var out = new Float32Array(W * H);
    for (var y = 1; y < H - 1; y++) {
      for (var x = 1; x < W - 1; x++) {
        var i  = y*W+x;
        var m  = mag[i];
        var nb = NMS_NEIGHBORS[dir[i]];
        if (m >= mag[(y+nb[0])*W+(x+nb[1])] && m >= mag[(y+nb[2])*W+(x+nb[3])]) {
          out[i] = m;
        }
      }
    }
    return out;
  }

  // Double threshold + BFS hysteresis linking.
  function hysteresisThreshold(nms, W, H) {
    var edges = new Uint8Array(W * H);
    var queue = [];
    for (var i = 0; i < W * H; i++) {
      if      (nms[i] >= CANNY_HI) { edges[i] = 255; queue.push(i); }
      else if (nms[i] >= CANNY_LO)   edges[i] = 128;   // weak — may be promoted
    }
    // BFS: promote weak edges connected to strong ones
    var qi = 0;
    while (qi < queue.length) {
      var idx = queue[qi++];
      var ey  = Math.floor(idx / W), ex = idx % W;
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          if (!dy && !dx) continue;
          var ny = ey+dy, nx = ex+dx;
          if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
            var ni = ny*W+nx;
            if (edges[ni] === 128) { edges[ni] = 255; queue.push(ni); }
          }
        }
      }
    }
    // Remove unconnected weak edges
    for (var i = 0; i < W * H; i++) {
      if (edges[i] === 128) edges[i] = 0;
    }
    return edges;
  }

  // Full Canny pipeline: pad to square → 256×256 → grayscale → blur → Sobel →
  // NMS → double threshold + hysteresis.  Matches cv2.Canny(img, 50, 150).
  function computeEdgeCanvas(imgEl) {
    // Pad to square with white background, then resize to PROCESS_SIZE
    var side = Math.max(imgEl.naturalWidth, imgEl.naturalHeight, 1);
    var tmp  = document.createElement('canvas');
    tmp.width  = PROCESS_SIZE;
    tmp.height = PROCESS_SIZE;
    var tc = tmp.getContext('2d');
    tc.fillStyle = '#ffffff';
    tc.fillRect(0, 0, PROCESS_SIZE, PROCESS_SIZE);
    var scale = PROCESS_SIZE / side;
    var dw = Math.round(imgEl.naturalWidth  * scale);
    var dh = Math.round(imgEl.naturalHeight * scale);
    var dx = Math.round((PROCESS_SIZE - dw) / 2);
    var dy = Math.round((PROCESS_SIZE - dh) / 2);
    tc.imageSmoothingEnabled  = true;
    tc.imageSmoothingQuality  = 'high';
    tc.drawImage(imgEl, dx, dy, dw, dh);

    var W = PROCESS_SIZE, H = PROCESS_SIZE;
    var imgData = tc.getImageData(0, 0, W, H);
    var gray    = toGrayscale(imgData.data);
    var blurred = gaussianBlur(gray, W, H);
    var sg      = sobelGradients(blurred, W, H);
    var nms     = nonMaxSuppression(sg.mag, sg.dir, W, H);
    var edges   = hysteresisThreshold(nms, W, H);

    // Render: white edges on black
    var out  = document.createElement('canvas');
    out.width  = W;
    out.height = H;
    var oc   = out.getContext('2d');
    var rgba = new Uint8ClampedArray(W * H * 4);
    for (var i = 0; i < W * H; i++) {
      var v = edges[i];
      rgba[i*4] = rgba[i*4+1] = rgba[i*4+2] = v;
      rgba[i*4+3] = 255;
    }
    oc.putImageData(new ImageData(rgba, W, H), 0, 0);
    return out;
  }

  // ── Card controller ────────────────────────────────────────

  function makeCard(cardEl, hintEl, initialPath) {
    var state     = 0;
    var curPath   = initialPath;
    var rawImg    = null;
    var edgeCvs   = null;

    var cvs   = cardEl.querySelector('canvas');
    var label = cardEl.querySelector('.art-label');
    var ctx   = cvs.getContext('2d');

    function setDims() {
      var cw = cardEl.offsetWidth  || 200;
      var ch = cardEl.offsetHeight || 200;
      if (cvs.width !== cw || cvs.height !== ch) {
        cvs.width  = cw;
        cvs.height = ch;
      }
      return { cw: cw, ch: ch };
    }

    function renderState() {
      var d  = setDims();
      var cw = d.cw, ch = d.ch;

      label.classList.remove('visible');
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, cw, ch);

      if (state === 0) {
        if (rawImg) {
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(rawImg, 0, 0, cw, ch);
        }
        hintEl.textContent = 'click to explore';

      } else if (state === 1) {
        if (edgeCvs) {
          ctx.imageSmoothingEnabled = true;    // smooth scaling — edges are already thin
          ctx.drawImage(edgeCvs, 0, 0, cw, ch);
        }
        hintEl.textContent = 'click for genre';

      } else if (state === 2) {
        if (rawImg) {
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(rawImg, 0, 0, cw, ch);
        }
        label.textContent = formatGenre(genreFromPath(curPath));
        label.classList.add('visible');
        hintEl.textContent = 'click for next';
      }
    }

    function loadImage(path) {
      state   = 0;
      curPath = path;
      rawImg  = null;
      edgeCvs = null;
      label.classList.remove('visible');
      hintEl.textContent = '·  ·  ·';

      var img = new Image();
      img.onload = function () {
        rawImg = img;
        try { edgeCvs = computeEdgeCanvas(img); } catch (_) {}
        renderState();
      };
      img.onerror = function () {
        hintEl.textContent = 'click to explore';
        renderState();
      };
      img.src = path;
    }

    cardEl.addEventListener('click', function () {
      if (state < 2) {
        state++;
        renderState();
      } else {
        // Load new image from a different genre, update activePaths
        var oldGenre = genreFromPath(curPath);
        var newPath  = pickNextImage(oldGenre);
        var idx      = activePaths.indexOf(curPath);
        if (idx !== -1) activePaths[idx] = newPath;
        else            activePaths.push(newPath);
        loadImage(newPath);
      }
    });

    // Keyboard a11y (Enter / Space trigger click)
    cardEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        cardEl.click();
      }
    });

    window.addEventListener('resize', function () { renderState(); });

    return { loadImage: loadImage, getPath: function () { return curPath; } };
  }

  // ── Init ───────────────────────────────────────────────────

  function init() {
    // typeof guard: const at top-level script scope is NOT a window property,
    // so !window.WIKIART_IMAGES would always be true. typeof is safe for both.
    if (typeof WIKIART_IMAGES === 'undefined' || !WIKIART_IMAGES.length) return;
    buildGenreMap();

    var items = document.querySelectorAll('.art-showcase-item');
    if (!items.length) return;

    // Pick one image from each of 3 different genres for the initial display
    var genreKeys = Object.keys(genreMap);
    var shuffled  = genreKeys.slice().sort(function () { return Math.random() - 0.5; });
    var selected  = [];
    for (var i = 0; i < Math.min(3, shuffled.length); i++) {
      selected.push(pickRandom(genreMap[shuffled[i]]));
    }
    while (selected.length < 3) {
      selected.push(pickRandom(WIKIART_IMAGES));
    }
    activePaths = selected.slice();

    items.forEach(function (item, i) {
      var cardEl = item.querySelector('.art-card');
      var hintEl = item.querySelector('.art-hint');
      if (!cardEl || !hintEl) return;
      var ctrl = makeCard(cardEl, hintEl, selected[i]);
      ctrl.loadImage(selected[i]);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
