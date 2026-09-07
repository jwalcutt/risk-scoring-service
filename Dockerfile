# The scoring service. Built to be a Compose component: the registry, the
# service config, and the working directory all arrive by mount, so the
# image carries code and dependencies and nothing else.
#
# Dependencies resolve from the same uv.lock CI installs, so the container
# runs the versions the tests ran against.
#
# Both base images are pinned by digest as well as tag. The digest is the
# multi-arch index, so one line covers every platform; bump the tag and
# the digest together, and record the new digest in docs/service-notes.md.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# LightGBM's wheel links against libgomp, which the slim base omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /usr/local/bin/uv

WORKDIR /opt/service

# Dependencies first, so a source edit does not reinstall mlflow.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

ENV PATH="/opt/service/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# The image has no .git, so the build stamps the commit in and /version
# stays honest in a container. Unset leaves it empty, which the service
# reads as "no stamp" rather than as a SHA.
ARG GIT_SHA=""
ENV RISK_SCORING_GIT_SHA=$GIT_SHA

RUN useradd --create-home --uid 10001 service
USER service

EXPOSE 8000

# 0.0.0.0 because the port is published; the host default stays loopback.
CMD ["python", "-m", "risk_scoring.service", "run", "--host", "0.0.0.0"]
