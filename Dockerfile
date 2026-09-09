# syntax=docker/dockerfile:1.7

# One image, three roles. `docker compose` picks the role with a command:
#
#   icu-monitor serve       -> the FastAPI service          (the image's default)
#   streamlit run app.py    -> the dashboard
#   icu-monitor etl/train   -> the one-shot data + model bootstrap
#
# Building the same layers for all three is what keeps "the API and the dashboard are the
# same tick" true in deployment as well as in the source: there is one install, so there is
# one NEWS2 implementation, one fusion, one model registry.
#
# The vision extra is NOT installed by default. `ultralytics` pulls in Torch - roughly
# 800 MB - and the detector degrades to a pure-NumPy backend that runs anywhere, so the
# default image stays small and still shows a live camera panel. To build with YOLO:
#
#   docker build --build-arg EXTRAS='[vision]' -t icu-monitor:vision .

ARG PYTHON_VERSION=3.12

# ======================================================================== builder
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG EXTRAS=""

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# A self-contained venv is the whole build output: the runtime stage copies this one
# directory and inherits nothing else, so no compiler or header ever reaches the final image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Dependency metadata first. Editing a source file must not invalidate the layer that
# resolved and downloaded ~200 MB of scientific Python wheels.
COPY pyproject.toml README.md ./
COPY src/icu_monitor/__init__.py src/icu_monitor/__init__.py
RUN pip install --upgrade pip setuptools wheel \
 && pip install ".${EXTRAS}" \
 && pip uninstall -y icu-monitor

# Now the real source, and install the project itself on top of the cached dependencies.
COPY src/ src/
RUN pip install --no-deps ".${EXTRAS}"

# ======================================================================== runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="ICU Sentinel" \
      org.opencontainers.image.description="ICU deterioration monitoring: NEWS2, ML risk stratification, bedside vision." \
      org.opencontainers.image.source="https://github.com/gunarithvick/icu-patient-risk-monitoring-system" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # The codebase logs SpO₂, ≥ and °C. Without this the first alert line can take the
    # process down on a container whose locale resolves to ASCII.
    PYTHONIOENCODING=utf-8 \
    # Resolve data/ and artifacts/ inside the image rather than beside the source tree,
    # which no longer exists here - the package is installed, not mounted.
    ICU_PROJECT_ROOT=/app \
    ICU_ENVIRONMENT=docker \
    ICU_FRAME_SOURCE=synthetic \
    ICU_API_HOST=0.0.0.0 \
    ICU_API_PORT=8000

# libgomp is scikit-learn's OpenMP runtime; the histogram gradient booster needs it at
# import time. Nothing else from the toolchain is required.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Unprivileged, with a fixed uid so a bind-mounted host volume has predictable ownership.
RUN groupadd --gid 1000 icu \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash icu

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# The dashboard entry point and its pinned dark theme. `app.py` is the same shim hosted
# Streamlit runs; with the package installed its sys.path insert is a no-op.
COPY --chown=icu:icu app.py ./
COPY --chown=icu:icu .streamlit/ .streamlit/

# Declared as volumes in compose. Created here so the first write does not fail on a
# read-only parent, and owned by icu so a non-root process can fill them.
RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts \
 && chown -R icu:icu /app

USER icu

EXPOSE 8000 8501

# Matches the default CMD. urllib rather than curl: no extra package, and it fails on a
# non-2xx status the way a probe should. The dashboard service overrides this in compose.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["icu-monitor", "serve"]

# =========================================================================== test
# `docker compose run --rm test` - the suite on a clean Linux Python, which is the check
# that matters when the development machine is Windows.
FROM builder AS test

ENV PYTHONIOENCODING=utf-8 \
    ICU_PROJECT_ROOT=/build

RUN pip install ".[dev]"
COPY tests/ tests/
CMD ["sh", "-c", "python -m ruff check src tests && python -m ruff format --check src tests && python -m pytest tests/"]
