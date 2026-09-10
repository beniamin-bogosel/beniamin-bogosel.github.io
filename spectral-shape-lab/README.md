# Spectral Shape Lab — browser-only app

The app performs PDE-free shape optimization, shows the current/best boundary,
objective history and all ten predicted eigenvalues, and exports JSON, CSV, SVG
and PNG. It uses the saved three-member 192×3 ReLU ensemble. Users can replace
the model and enter new polynomial/elementary-function objectives.

All calculations run locally in a Web Worker. There are **no runtime npm
dependencies, CDN requests, backend API calls or PDE solves**. A web server is
needed only to deliver static files, because browsers restrict module workers
and fetch when a page is opened directly with `file://`.

## Run locally

From the project root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open **http://localhost:8000/**. This Python process serves files only; it never
loads the model or performs optimization. Any static HTTP server works.
No Node/npm installation is needed to use the existing app.

Choose a preset or edit F, choose area/direction/start count/step budget, and
click **Start optimization**. Pause/resume or stop at any time. The best candidate
is retained and can be saved during a run as a snapshot or afterward as a full
run record. The default is four starts and 150 steps; a browser's actual speed
depends on hardware and the line-search path.

## New functionals

The expression editor accepts:

- Variables `x1` through `x10`: the physical eigenvalues at the chosen area.
- Arithmetic `+ - * / ^` (`**` also works), parentheses and real constants,
  including scientific notation, `pi` and `e`.
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`,
  `exp`, `log`/`ln`, `sqrt`.

Examples:

```text
x5 + 0.5*x6
(x6-x5)^2
x6/x1 + 0.01*log(x10)
x5^2/x1 + exp(x2/100)
```

Multiplication is explicit: `2*x5`, not `2x5`. Unary minus has mathematical
precedence: `-x1^2` means `-(x1^2)`. Powers associate to the right.
The parser supports up to 1,024 characters, 256 tokens, and 32 levels of explicit
parenthesis/function/unary nesting. It builds an expression tree and ten symbolic
derivative trees. It never uses `eval` or `new Function`; property access,
assignments, indexing and arbitrary JavaScript are rejected. No piecewise,
`abs`, `min`, or `max` is accepted in this C¹-oriented interface.

F and its partial derivatives must be real and finite at evaluated spectra.
The disk must be in their domain. Invalid trial points trigger backtracking;
invalid training spectra are excluded from start ranking. C¹ regularity is the
user's responsibility: the app does not prove it or remove the ReLU/PDE
eigenvalue-crossing nonsmoothness.

**Save formula** keeps up to 30 custom formulas in this browser's local storage.
For presets shared with every visitor, edit `objectives.json` (name/expression
pairs). The browser's formula definitions are separate from Python's
`objective_functions.py`; the built-in cases reproduce the corresponding
Python expressions, and arbitrary new cases are differentiated in the browser.

## Replace a model

Use **Replace model** to import a `.spectral.json` file. The file is read locally
and is not uploaded. Model dimensions, output mapping, normalization, ensemble
members, radial constraints, training starts and support metadata all come from
the bundle. No interface edit is needed to switch compatible models.

Export another trained model from the project root:

```bash
MPLCONFIGDIR=/tmp/spectral-mpl python -m spectral_shape.export_browser \
  --models results/my_ten_output_model \
  --out web/models/my_model.spectral.json \
  --name "My ten-output model"
```

The exporter requires the model's checkpoints, `metrics.json`, `split.npz` and,
when present, `training_data.npz`. It calibrates support using training and
validation geometry, with no new PDE computation and no test labels used for
starts. Existing output files are protected; use `--force` only when intentionally
replacing an export. To change the default deployed model, replace
`models/vector10.spectral.json` with the new compatible bundle or change its
relative URL in `src/app.js`.

Version 1 supports exactly all ten Dirichlet outputs, 1–32 harmonics,
1–8 hidden layers, hidden widths up to 2,048, 1–8 ensemble members, and
ReLU/tanh/SiLU activations. The optional Faber–Krahn head is rejected. Support
normalization must have equal sine/cosine scales in each harmonic pair (as in
the current training pipeline). Imported bundles are limited to 64 MB.

The provided bundle is **7.29 MB**: it includes the unchanged network weights
and all 14,000 training coefficient/label rows, not only the roughly 1 MB
ensemble. Weights use little-endian float32; training/support data use float64;
arrays are base64-encoded inside one JSON file. Normalization, layer dimensions,
output ordering, constraints, support radius and source hashes are explicit.
Download JSON records include the bundle's SHA-256 on HTTPS/localhost browsers
with Web Crypto, so a run can be associated with its exact model artifact.

## Numerical implementation

The saved network computes J=Aλ. For fixed requested area A*, the browser uses
λ=J/A* and the chain rule

```text
∇q F = Σk (∂F/∂xk)(J/A*) · ∇q Jk / A*
```

`src/model.js` implements dense layers, activations, input/output normalization,
the 24 rotation/reflection prediction views per member, and ensemble averaging.
It records a forward tape and reverses those operations to form a vector-Jacobian
product. The only optimized variables are the Fourier inputs; network weights
stay fixed. This compact reverse-mode implementation avoids a large browser ML
runtime and has been checked against PyTorch. It uses float64 accumulation with
the exported float32 weights, so small floating-point differences and different
ReLU/line-search decisions are possible.

The optimizer minimizes F/s, or -F/s for maximization, with fixed
`s=max(1,abs(F(predicted disk)))`. It does not take log(F): negative/zero objectives
are valid. Coefficients are normalized analytically to the requested area, and
the same sum-of-harmonic-amplitudes budget and per-harmonic caps are projected at
every accepted step. The provided model therefore retains
**r(θ) ≥ 0.25*a0 > 0 for every angle**, preventing self-intersection.

The support check retains all training points and the original 48-view support
orbit. Equal pair scales let us transform a query against a base-point KD tree
instead of storing 48 trees. This is equivalent in exact arithmetic. The
exporter measures a global bound on the old float32 orbit's rounding difference
and the browser conservatively requires

```text
ideal nearest distance + roundoff bound <= original support radius
```

For the bundled model the radius is 10.73994742 and the margin is about
4.65e-5 in standardized coordinates. Thus this numerical adaptation does not
silently enlarge the permitted support region. It saves the large replicated
tree and does not replace it with a small sampled support set.

Starts are the disk, the best training shapes ranked by F of their stored
spectra, and remaining random training shapes. The browser uses a documented
Mulberry32 RNG, so its random starts differ from NumPy's PCG64. Exact start
coordinates are saved. Historical single-eigenvalue reference shapes are not
automatically imported. Each initial candidate and gradient endpoint is retained;
selection uses only the surrogate. Step budgets and support/line-search stops
are not convergence certificates.

Single-eigenvalue presets also use this fixed-scaled F objective, matching the
general Python functional optimizer. The historical single-eigenvalue study
used log(λk); this scaling difference, the browser RNG and floating-point
differences mean full trajectories need not reproduce that older experiment.

Predicted outputs are displayed without sorting. An ordering warning appears
if heads invert. Ensemble spread is not a confidence interval. Optimization can
find regions with larger surrogate errors than the held-out test average;
the app cannot independently certify PDE values.

## Code map

| File | Responsibility |
|---|---|
| `index.html`, `style.css` | Responsive interface and layout |
| `src/app.js` | Controls, worker messages, local presets, file import/export |
| `src/worker.js` | Model loading, background computation, pause/resume/stop |
| `src/model.js` | Model schema, prediction tape, neural reverse differentiation |
| `src/expression.js` | Restricted grammar, symbolic derivatives, domain checks |
| `src/geometry.js` | Projection, area normalization, Fourier boundary |
| `src/support.js` | Full training-support orbit search |
| `src/optimizer.js` | Objective chain rule, starts, projected line search, result records |
| `src/plots.js` | Live/final SVG plots and export rendering |
| `objectives.json` | Shared saved functionals |
| `../spectral_shape/export_browser.py` | Offline Python-to-browser model export |
| `scripts/build.mjs` | Assemble only runtime files into `dist/` |
| `tests/` | Numerical reference checks and browser integration tests |

## Tests and reproduction

From the project root, the recorded export command was:

```bash
MPLCONFIGDIR=/tmp/spectral-mpl python -m spectral_shape.export_browser \
  --models results/vector10_relu/model \
  --out web/models/vector10.spectral.json \
  --fixtures web/tests/python-reference.json
```

The output already exists; use fresh paths or an intentional `--force` for a
repeat. The reference fixture contains PyTorch predictions, all ten input
Jacobians, original support distances, and Python metric/Euclidean projections.

From `web/`:

```bash
npm test
npm ci
npx playwright install chromium
# In another terminal: python3 -m http.server 8000 --bind 127.0.0.1
npm run test:browser
npm run build
# Test the production folder under a project-style subpath:
SPECTRAL_TEST_URL=http://127.0.0.1:8000/dist/ npm run test:browser
node scripts/validate.mjs
python scripts/package.py
```

Only browser tests need Playwright; the app and numerical tests have no npm
runtime dependencies. This workspace downloaded Chromium to
`/tmp/spectral-playwright`; its test commands additionally set
`PLAYWRIGHT_BROWSERS_PATH=/tmp/spectral-playwright`.

Numerical checks cover formula derivatives/domain rejection, all ten prediction
heads and Jacobians, non-unit area and maximization, output mapping, both
projections, orbit distances, feasibility and cancellation. Browser checks
exercise live/final optimization, all four downloads, pause/resume/stop, custom
formula persistence, invalid formulas, local model replacement, recovery after
a malformed bundle, and mobile layout. Browser run logs and screenshots are
under `test-results/`; the durable summary is `VALIDATION.json`.
The current suite has 16 numerical checks and five end-to-end browser tests.
The replacement-model fixture is synthetic and exists only for tests: it
changes the harmonic count, layer dimensions, activations and output order.
To regenerate it from the project root, run
`MPLCONFIGDIR=/tmp/spectral-mpl python web/tests/make_replacement_fixture.py`.

## Publish on username.github.io

**Yes: GitHub Pages is sufficient.** No Python/Node/FreeFem process runs on the
host. Models and assets use relative paths, including when the app lives under
`https://username.github.io/repository/`.

Build with `node web/scripts/build.mjs` from the project root, then publish the
**contents of `web/dist/`**. It includes a `.nojekyll` file. Do not publish
`node_modules`, test results, or the whole research dataset directory.

Two options:

1. Copy the contents of `web/dist/` to the root of a separate Pages repository
   or publishing branch and select that branch in Settings → Pages.
2. For this repository layout, copy `web/github-pages.yml.example` to
   `.github/workflows/spectral-pages.yml`, select **GitHub Actions** as the Pages
   source, and run **Publish spectral shape app** manually. The example builds
   the static folder and deploys only that artifact.

The workflow is supplied as an example and has not been enabled or published.
See [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

All model replacement and result saving are local browser operations. Shipping
a model with a static site makes its weights and exported training support
downloadable to visitors. No telemetry or upload service is included.
