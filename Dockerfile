# The scoring service. Built to be a Compose component: the registry, the
# service config, and the working directory all arrive by mount, so the
# image carries code and dependencies and nothing else.
#
# Dependencies resolve from the same uv.lock CI installs, so the container
# runs the versions the tests ran against.
FROM python:3.12-slim

# LightGBM's wheel links against libgomp, which the slim base omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

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
