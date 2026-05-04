"""
live.py -- Main live trading entry point (multi-pair, multi-timeframe).

Loads saved models, starts live candle fetcher and signal generator
for all pairs x timeframes, and launches the Streamlit dashboard.

Usage:
    python live.py
    python live.py --asset EURUSD GBPUSD --timeframe 1M 5M
    python live.py --no-dashboard
"""

import argparse
import asyncio
import signal
import subprocess
import sys
import os
from pathlib import Path

from config import (
    ASSETS,
    TIMEFRAME,
    TIMEFRAMES,
    MIN_CONFIDENCE,
    MODEL_DIR,
    BASE_DIR,
    DASHBOARD_PORT,
)
from utils.logger import setup_logger

logger = setup_logger(
    "binary_predictor",
    log_file=BASE_DIR / "logs" / "app.log",
    level="INFO",
)


class LiveRunner:
    """
    Orchestrates multi-pair, multi-timeframe live signal generation + dashboard.
    """

    def __init__(
        self,
        pairs: list[str],
        timeframes: list[str],
        min_confidence: float,
        launch_dashboard: bool = True,
    ):
        self.pairs = pairs
        self.timeframes = timeframes
        self.min_confidence = min_confidence
        self.launch_dashboard = launch_dashboard
        self._dashboard_proc = None
        self._shutdown = False

    async def run(self):
        """Start all components."""
        n_channels = len(self.pairs) * len(self.timeframes)

        logger.info("=" * 60)
        logger.info("  BINARY PREDICTOR -- LIVE MODE")
        logger.info(f"  Assets:      {self.pairs}")
        logger.info(f"  Timeframes:  {self.timeframes}")
        logger.info(f"  Channels:    {n_channels} ({len(self.pairs)} pairs x {len(self.timeframes)} TFs)")
        logger.info(f"  Confidence:  >={self.min_confidence}")
        logger.info("=" * 60)

        # Verify models exist
        if not self._check_models():
            logger.error(
                "No trained models found. Run `python train.py` first."
            )
            return

        # Start dashboard in subprocess
        if self.launch_dashboard:
            self._start_dashboard()

        # Start signal generator with multi-TF support
        from signals.generator import SignalGenerator

        generator = SignalGenerator(
            pairs=self.pairs,
            timeframes=self.timeframes,
            min_confidence=self.min_confidence,
        )

        # Set up signal handler for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: self._request_shutdown(generator))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Initialize and run
        ok = await generator.initialize()
        if not ok:
            logger.error("Failed to initialize signal generator")
            self._stop_dashboard()
            return

        logger.info(f"[OK] Live system running with {n_channels} channels. Press Ctrl+C to stop.")

        try:
            await generator.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            generator.stop()
            self._stop_dashboard()
            logger.info("Live system shut down gracefully")

    def _check_models(self) -> bool:
        """Check if trained models exist."""
        required = [
            MODEL_DIR / "xgb_model.pkl",
            MODEL_DIR / "lgbm_model.pkl",
            MODEL_DIR / "ensemble_meta.pkl",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            for p in missing:
                logger.warning(f"Missing model file: {p}")
            return False
        return True

    def _start_dashboard(self):
        """Launch Streamlit dashboard in a subprocess."""
        dashboard_script = BASE_DIR / "dashboard" / "app.py"
        if not dashboard_script.exists():
            logger.warning("Dashboard script not found, skipping")
            return

        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(dashboard_script),
                "--server.port", str(DASHBOARD_PORT),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--theme.base", "dark",
            ]
            self._dashboard_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(BASE_DIR),
            )
            logger.info(
                f"[OK] Dashboard started at http://localhost:{DASHBOARD_PORT}"
            )
        except Exception as e:
            logger.warning(f"Could not start dashboard: {e}")

    def _stop_dashboard(self):
        """Stop the Streamlit dashboard subprocess."""
        if self._dashboard_proc:
            try:
                self._dashboard_proc.terminate()
                self._dashboard_proc.wait(timeout=5)
                logger.info("Dashboard stopped")
            except Exception:
                self._dashboard_proc.kill()

    def _request_shutdown(self, generator):
        """Request graceful shutdown."""
        if not self._shutdown:
            self._shutdown = True
            logger.info("Shutdown requested...")
            generator.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Binary Options Candle Direction Predictor -- Live Mode"
    )
    parser.add_argument(
        "--asset", type=str, nargs="+", default=ASSETS,
        help="Currency pair(s) to trade",
    )
    parser.add_argument(
        "--timeframe", type=str, nargs="+", default=TIMEFRAMES,
        help="Candle timeframe(s): 1M 5M (multiple allowed)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=MIN_CONFIDENCE,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Don't launch the Streamlit dashboard",
    )
    args = parser.parse_args()

    runner = LiveRunner(
        pairs=[a.upper() for a in args.asset],
        timeframes=[t.upper() for t in args.timeframe],
        min_confidence=args.min_confidence,
        launch_dashboard=not args.no_dashboard,
    )

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Exiting...")


if __name__ == "__main__":
    main()
