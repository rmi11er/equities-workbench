from src.metals.cli import build_parser


def test_databento_cli_defaults_to_bounded_silver_collection():
    args = build_parser().parse_args(["collect-databento"])

    assert args.command == "collect-databento"
    assert args.metal == "silver"
    assert args.duration_seconds == 60.0
    assert args.max_records == 0
