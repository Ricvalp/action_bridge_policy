"""Evaluate one whole-plan Push-T DDIM checkpoint in closed loop."""
from action_bridge.eval.revision_pusht_cli import main as evaluate_main


def main(argv=None):
    return evaluate_main("ddim", argv)


if __name__ == "__main__":
    raise SystemExit(main())
