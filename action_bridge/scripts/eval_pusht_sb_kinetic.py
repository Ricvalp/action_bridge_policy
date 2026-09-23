"""Evaluate one whole-plan Push-T kinetic Schrodinger-bridge checkpoint in closed loop."""
from action_bridge.eval.revision_pusht_cli import main as evaluate_main


def main(argv=None):
    return evaluate_main("sb_kinetic", argv)


if __name__ == "__main__":
    raise SystemExit(main())
