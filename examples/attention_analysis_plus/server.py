"""One seeded VLAct policy server for one complete Plus suite."""

import argparse
import logging
from pathlib import Path

from examples.attention_analysis_plus.model import RecordedPolicy, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10103)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    seed_all(args.seed)
    policy = RecordedPolicy(args.checkpoint, args.traces)
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    WebsocketPolicyServer(
        policy, host="127.0.0.1", port=args.port, metadata={"action_chunk_size": 8, "normalized_actions": True}
    ).serve_forever()


if __name__ == "__main__":
    main()
