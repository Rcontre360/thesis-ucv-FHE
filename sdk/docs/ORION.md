# Running the Orion baseline

`examples/orion_shallow_cnn.py` runs a single encrypted MNIST inference under
[Orion](https://github.com/baahl-nyu/orion) (Lattigo backend) on the same CNN
topology our SDK example uses (`Conv(1→4, k=7, s=5) → ReLU → Linear(100, 10)`).
This document lists the prerequisites and walks through the setup.

The Orion source tree must be present at `sdk/temp/orion/` (it's used as the
working copy *and* as the source of the config file the example loads).

## System prerequisites

### 1. Go ≥ 1.22

Orion's Lattigo backend is Go-native — it compiles a CGO bridge that loads as
a Python shared library. Without Go on `PATH`, the `pip install` step fails at
the `cgo` compile stage.

```bash
# Check first:
go version             # should print >= go1.22

# If missing or older:
cd /tmp
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. C/C++ build deps (GMP, OpenSSL, build-essential)

Lattigo links against system GMP and OpenSSL.

```bash
sudo apt update && sudo apt install -y \
    build-essential pkg-config \
    libgmp-dev libssl-dev \
    python3 python3-pip python3-venv
```

To verify:

```bash
ldconfig -p | grep -E 'libgmp|libssl'
```

If both `libgmp.so` and `libssl.so.3` show up, you're set.

### 3. Python in `[3.9, 3.13)`

Orion's `pyproject.toml` pins `requires-python = ">=3.9,<3.13"`. Newer Python
(3.13+) is not supported.

```bash
python3 --version
```

### 4. CUDA — **not required**

Orion's `lattigo` backend runs on **CPU only**. You don't need a GPU at all
for this example (training happens on whatever device PyTorch picks; FHE
inference is pure CPU). The script will use CUDA for the training pass if
available, but the bottleneck is the FHE forward and that lives in Lattigo.

### 5. RAM

The default Orion ResNet config (`temp/orion/configs/resnet.yml`) targets
`LogN=16` with bootstrapping enabled. Expect **8-16 GB of RAM** during
`orion.compile` and the FHE forward. The 4 GB GPU on a local laptop is
irrelevant; what matters is host RAM.

## Install

From the SDK root, with your `.env` virtualenv active:

```bash
cd sdk
source .env/bin/activate
pip install -e temp/orion
```

The first install takes several minutes — it compiles the Lattigo CGO bridge.
Verify the install:

```bash
python -c "import orion; print(orion.__file__)"
```

You should see a path inside `temp/orion/orion/`.

### ⚠ Torch version conflict

Orion's `pyproject.toml` requires `torch>=2.2.0` with **no upper bound**, so
`pip install -e temp/orion` will silently upgrade torch to the latest
available wheel (e.g. 2.12.0). That **breaks** other libraries pinned to a
specific torch version, in particular:

- `concrete-ml 1.9.0` — pins `torch==2.3.1`
- Our `fhe-sdk` (built against the CUDA build of torch 2.3.1+cu121)

If you intend to run both Orion and the SDK / Concrete-ML in the **same**
Python env, install Orion with `--no-deps` to skip the torch upgrade:

```bash
pip install --no-deps -e temp/orion
# then install Orion's other (non-torch) deps manually if missing:
pip install "PyYAML>=6.0" "tqdm>=4.30" "scipy>=1.7,<=1.14.1" h5py certifi matplotlib
```

If you only need Orion (this baseline script), the unrestricted install is
fine — the script doesn't depend on the SDK or Concrete-ML.

The cleanest separation is **two virtualenvs**: one for our SDK +
Concrete-ML, one for Orion. We don't need them coexisting at runtime; the
comparison is across separate measurement runs.

## Run the example

```bash
cd sdk
python examples/orion_shallow_cnn.py
```

Stages and expected timing on a modern laptop CPU:

| Stage | What it does | Time |
|---|---|---|
| `load_mnist` | Hugging Face download + 56k/14k split | first time ~30s, cached after |
| `train` | 8-epoch PyTorch training of the CNN | ~30s on CPU, much less with CUDA |
| `orion.init_scheme` | Loads YAML, generates the Lattigo CKKS scheme + keys | ~60-180s |
| `orion.fit` | Calibrates the polynomial-ReLU input range from training data | ~10-30s |
| `orion.compile` | Runs auto-bootstrap placement (shortest-path over the level DAG) | seconds |
| `orion.encode` + `encrypt` | One-time encryption of the single input sample | seconds |
| `net(vec_ctxt)` (the FHE forward) | One encrypted inference through the CNN | **2-10 minutes** |
| `decrypt` + `decode` | Recover the 10-class logits | seconds |

The FHE forward is the headline number — Orion's `on.ReLU` defaults to
`degrees=[15, 15, 27]` for the Chebyshev sign approximation, which is quite
deep. Expect the forward to dominate total runtime.

## Output

The script prints a summary like:

```
=== Orion FHE inference summary ===
  true label         : 7
  plaintext pred     : 7
  encrypted pred     : 7
  plaintext == FHE   : OK
  MAE (clear vs FHE) : 0.000123
  precision (-log2)  : 13.0 bits
  FHE forward time   : 217.5s
```

A working setup gets `plaintext == FHE   : OK` and `precision` in the 10-20
bit range.

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install` fails at `go build` step | Go missing / wrong version | Install Go ≥ 1.22 (see step 1) |
| `ImportError: libgmp.so.10: cannot open` | GMP not installed | `apt install libgmp-dev` |
| `RuntimeError: scheme is None` at `orion.encode` | Forgot `orion.init_scheme(...)` | Make sure the scheme init runs before encode |
| `MemoryError` in `orion.compile` or FHE forward | Host RAM too small | Need ≥ 8 GB free RAM, ideally 16 GB |
| FHE forward extremely slow (>30 min) | Running on a slow CPU or under load | Move to a dedicated machine; Lattigo is single-threaded per ciphertext op |
| `ModuleNotFoundError: orion` after `pip install -e` | Editable install picked the wrong site-packages | Activate the SDK's `.env` first, then reinstall |

## Why this baseline matters for the thesis

Orion is the closest published reference for our use case:
- Same FHE scheme (CKKS)
- Same workflow (PyTorch network → polynomial ReLU → encrypted inference)
- Same target precision (~15-25 bits after bootstrap, Lattigo-grade)
- Open-source, reproducible

Comparing our SDK against Orion on the same network gives a direct
apples-to-apples measurement of how much our GPU acceleration buys (latency)
and where Orion's CPU + Chebyshev bootstrap still has the edge (precision).
