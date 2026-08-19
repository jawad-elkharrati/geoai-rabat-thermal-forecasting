from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, load_config
from .dataset import build_and_save_demo_dataset
from .modeling import train_week5_models
from .sources import fetch_samples
from .verify import verify_week5


def _print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoai-rabat",
        description="Pipeline GeoAI de cartographie thermique de Rabat - semaines 1 à 5",
    )
    parser.add_argument(
        "command",
        choices=[
            "fetch-samples",
            "build-demo",
            "train",
            "run-week5",
            "verify",
            "run-real",
            "verify-real",
            "report-real",
            "evaluate-final",
            "forecast-two-days",
            "run-complete",
            "verify-final",
            "verify-delivery",
            "report-final",
            "report-defense",
        ],
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Chemin de la configuration JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    real_commands = {
        "run-real",
        "verify-real",
        "report-real",
        "evaluate-final",
        "forecast-two-days",
        "run-complete",
        "verify-final",
        "verify-delivery",
        "report-final",
        "report-defense",
    }
    config = None if args.command in real_commands else load_config(Path(args.config))

    try:
        if args.command == "fetch-samples":
            assert config is not None
            _print_result(fetch_samples(config))
        elif args.command == "build-demo":
            assert config is not None
            result = build_and_save_demo_dataset(config)
            _print_result(
                {
                    "status": result["audit"]["status"],
                    "rows": result["audit"]["rows"],
                    "dates": result["audit"]["dates"],
                }
            )
        elif args.command == "train":
            assert config is not None
            result = train_week5_models(config)
            _print_result(result["metadata"])
        elif args.command == "run-week5":
            assert config is not None
            dataset = build_and_save_demo_dataset(config)
            training = train_week5_models(config)
            verification = verify_week5(config)
            _print_result(
                {
                    "status": "ok",
                    "dataset_rows": dataset["audit"]["rows"],
                    "selected_model": training["metadata"]["selected_model"],
                    "verification": verification,
                }
            )
        elif args.command == "verify":
            assert config is not None
            _print_result(verify_week5(config))
        elif args.command == "run-real":
            from .real_pipeline import run_real_pipeline

            _print_result(run_real_pipeline(args.config))
        elif args.command == "verify-real":
            from .real_pipeline import verify_real_pipeline

            _print_result(verify_real_pipeline(args.config))
        elif args.command == "report-real":
            from .real_report import generate_real_report

            _print_result(generate_real_report(args.config))
        elif args.command == "evaluate-final":
            from .final_pipeline import evaluate_final_test

            _print_result(evaluate_final_test(args.config))
        elif args.command == "forecast-two-days":
            from .final_pipeline import generate_two_day_forecast

            _print_result(generate_two_day_forecast(args.config))
        elif args.command == "run-complete":
            from .final_pipeline import run_complete_pipeline

            _print_result(run_complete_pipeline(args.config))
        elif args.command == "verify-final":
            from .final_pipeline import verify_final_pipeline

            _print_result(verify_final_pipeline(args.config))
        elif args.command == "verify-delivery":
            from .final_pipeline import verify_complete_delivery

            _print_result(verify_complete_delivery(args.config))
        elif args.command == "report-final":
            from .final_report import generate_final_report

            _print_result(generate_final_report(args.config))
        elif args.command == "report-defense":
            from .defense_report import generate_defense_guide

            _print_result(generate_defense_guide(args.config))
        return 0
    except Exception as exc:
        print(f"ERREUR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
