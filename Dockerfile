# Reproduction image for FlowCon-X.
#
#   docker build -t flowconx .
#   docker run --rm -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" flowconx make repro-small
#
# The raw archives are mounted rather than copied: they are 24 GB together and
# the pipeline streams from them without expanding, so the image stays small.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# `git` is required at runtime, not for the build: every run records the
# commit it came from, and a run that cannot determine it is recorded as
# "unknown" rather than silently attributed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt -r requirements-dev.txt

COPY . .
RUN pip install -e .

# Verify the install rather than trusting it: the leakage suite is the gate
# that this image must not ship without.
RUN python -m pytest tests/ -q -m "not slow"

CMD ["make", "help"]
