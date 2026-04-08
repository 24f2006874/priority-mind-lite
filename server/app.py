from __future__ import annotations

from environment import PriorityMindEnv
from models import Action, Observation

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core with server dependencies is required to run the HTTP server."
    ) from exc


app = create_app(
    PriorityMindEnv,
    Action,
    Observation,
    env_name="priority-mind-lite",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
