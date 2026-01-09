"""Run context management for versioned logging and debugging.

Each execution creates a timestamped run directory with:
- All log files (ops.log, decisions.jsonl, tape.csv)
- manifest.json with version info, config snapshot, git state
- Symlink 'latest' pointing to most recent run
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class GitInfo:
    """Git repository state at runtime."""
    commit_hash: str = "unknown"
    commit_short: str = "unknown"
    branch: str = "unknown"
    dirty: bool = False

    @classmethod
    def detect(cls) -> "GitInfo":
        """Detect git info from current directory."""
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
            commit_short = commit_hash[:8] if commit_hash != "unknown" else "unknown"

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"

            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            dirty = bool(result.stdout.strip()) if result.returncode == 0 else False

            return cls(
                commit_hash=commit_hash,
                commit_short=commit_short,
                branch=branch,
                dirty=dirty,
            )
        except Exception:
            return cls()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunContext:
    """
    Context for a single execution run.

    Provides:
    - Unique run ID (timestamp-based)
    - Git version info
    - Run-specific directory paths
    - Manifest generation
    """
    run_id: str = ""
    start_time: str = ""
    git_info: GitInfo = field(default_factory=GitInfo)
    python_version: str = ""
    base_log_dir: str = "logs"

    # Paths (populated by setup())
    run_dir: str = ""
    ops_log_path: str = ""
    decisions_path: str = ""
    tape_path: str = ""
    manifest_path: str = ""

    # Config snapshot (set externally)
    config_snapshot: dict = field(default_factory=dict)
    ticker: str = ""
    environment: str = ""

    def __post_init__(self):
        if not self.run_id:
            now = datetime.now()
            self.run_id = now.strftime("%Y%m%d_%H%M%S")
            self.start_time = now.isoformat()

        if not self.python_version:
            self.python_version = sys.version

        if not self.git_info.commit_hash or self.git_info.commit_hash == "unknown":
            self.git_info = GitInfo.detect()

    def setup(self) -> "RunContext":
        """
        Create run directory and setup paths.

        Returns self for chaining.
        """
        # Create run directory
        self.run_dir = str(Path(self.base_log_dir) / f"run_{self.run_id}")
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)

        # Set paths
        self.ops_log_path = str(Path(self.run_dir) / "ops.log")
        self.decisions_path = str(Path(self.run_dir) / "decisions.jsonl")
        self.tape_path = str(Path(self.run_dir) / "tape.csv")
        self.manifest_path = str(Path(self.run_dir) / "manifest.json")

        # Create/update 'latest' symlink
        latest_link = Path(self.base_log_dir) / "latest"
        try:
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.exists():
                # Not a symlink, don't overwrite
                pass
            else:
                # Create relative symlink
                latest_link.symlink_to(f"run_{self.run_id}")
        except OSError:
            pass  # Symlinks may not work on all systems

        return self

    def write_manifest(self) -> None:
        """Write manifest.json with run metadata."""
        manifest = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "git": self.git_info.to_dict(),
            "python_version": self.python_version,
            "ticker": self.ticker,
            "environment": self.environment,
            "config": self.config_snapshot,
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def get_startup_banner(self) -> str:
        """Generate startup banner with version info."""
        dirty_marker = " (dirty)" if self.git_info.dirty else ""
        lines = [
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Git:    {self.git_info.commit_short}{dirty_marker} ({self.git_info.branch})",
            f"Ticker: {self.ticker}",
            f"Env:    {self.environment}",
            f"Logs:   {self.run_dir}/",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "git_commit": self.git_info.commit_short,
            "git_branch": self.git_info.branch,
            "git_dirty": self.git_info.dirty,
            "ticker": self.ticker,
            "environment": self.environment,
            "run_dir": self.run_dir,
        }


def create_run_context(
    base_log_dir: str = "logs",
    ticker: str = "",
    environment: str = "",
    config: Optional[Any] = None,
) -> RunContext:
    """
    Factory function to create and setup a RunContext.

    Args:
        base_log_dir: Base directory for all logs
        ticker: Market ticker being traded
        environment: Trading environment (demo/production)
        config: Config object to snapshot (optional)

    Returns:
        Fully initialized RunContext with directories created
    """
    ctx = RunContext(
        base_log_dir=base_log_dir,
        ticker=ticker,
        environment=environment,
    )

    # Snapshot config if provided
    if config is not None:
        try:
            from dataclasses import asdict
            ctx.config_snapshot = asdict(config)
        except Exception:
            ctx.config_snapshot = {"error": "Could not serialize config"}

    # Setup directories and paths
    ctx.setup()

    # Write manifest
    ctx.write_manifest()

    return ctx
