FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY . ./

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "main.py"]
