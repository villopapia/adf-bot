"""
================================================================================
 ALPACA BROKER  --  Live Order Execution & Position Management
================================================================================
 Wraps the Alpaca Trading API (alpaca-py) to bridge the paper trading system
 with real (or Alpaca paper) brokerage execution.

 Architecture
 ------------
 The trading bot has three independent signal modules:

   1. Pairs   (trade_tracker.py)     -- statistical arbitrage, two-leg trades
   2. Momentum (momentum_tracker.py) -- single-stock momentum, long only
   3. Bear    (bear_tracker.py)      -- mean-reversion bounces & inverse ETFs

 Each module generates "diamond" signals that are logged as paper trades.
 When LIVE_TRADING_ENABLED is True, this broker layer submits corresponding
 orders to Alpaca and tracks their state in broker_state.json.

 Safety guarantees
 -----------------
 - Every public method is wrapped in try/except and returns a safe default
   on failure.  The broker NEVER crashes the main system.
 - When LIVE_TRADING_ENABLED is False, all methods are no-ops.
 - Idempotency: duplicate submit calls for the same trade_id are ignored.
 - broker_state.json provides an audit trail of every order submitted.

 Usage
 -----
   from alpaca_broker import AlpacaBroker
   broker = AlpacaBroker()

   if broker.is_active:
       broker.submit_entry_single(trade_id, "AAPL", 10, "buy", "momentum")
       broker.submit_entry_pairs(trade_id, info_dict)
       broker.sync_positions()

 Dependencies
 ------------
   pip install alpaca-py
================================================================================
"""

import os
import sys
import json
import math
import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import config


# ---------------------------------------------------------------------------
#  AlpacaBroker
# ---------------------------------------------------------------------------

class AlpacaBroker:
    """Alpaca order execution layer for the ADF trading bot."""

    def __init__(self):
        self._client = None
        self._enabled = getattr(config, "LIVE_TRADING_ENABLED", False)
        self._paper = getattr(config, "ALPACA_PAPER", True)
        self._max_retries = getattr(config, "ALPACA_MAX_ORDER_RETRIES", 2)
        self._state_path = os.path.join(
            _SCRIPT_DIR,
            getattr(config, "BROKER_STATE_JSON", "broker_state.json"),
        )
        self._state = self._load_broker_state()

        if not self._enabled:
            self._log("LIVE_TRADING_ENABLED is False -- broker inactive (no-op mode)")
            return

        # Resolve API credentials: config first, then env vars
        api_key = getattr(config, "ALPACA_API_KEY", "") or os.environ.get("ALPACA_API_KEY", "")
        secret_key = getattr(config, "ALPACA_SECRET_KEY", "") or os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            self._log("WARNING: API keys missing -- broker cannot initialise")
            self._enabled = False
            return

        try:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(api_key, secret_key, paper=self._paper)
            self._log(
                f"Initialised ({'PAPER' if self._paper else 'LIVE'} mode)"
            )
        except Exception as exc:
            self._log(f"ERROR: Failed to create TradingClient -- {exc}")
            self._enabled = False

    # ------------------------------------------------------------------
    #  Properties / helpers
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True only if live trading is enabled and the client was created."""
        return self._enabled and self._client is not None

    def _log(self, msg: str):
        """Print with [BROKER] prefix."""
        print(f"[BROKER] {msg}")

    def _load_broker_state(self) -> dict:
        """Load broker_state.json from disk, or return empty skeleton."""
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            self._log(f"WARNING: Could not load broker state -- {exc}")
        return {"orders": {}, "last_sync": None}

    def _save_broker_state(self):
        """Persist broker_state.json to disk."""
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as exc:
            self._log(f"WARNING: Could not save broker state -- {exc}")

    def _now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ------------------------------------------------------------------
    #  Market hours
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Check if the US equity market is currently open."""
        if not self.is_active:
            return False
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception as exc:
            self._log(f"ERROR: get_clock failed -- {exc}")
            return False

    # ------------------------------------------------------------------
    #  Single-leg entry (momentum, bear)
    # ------------------------------------------------------------------

    def submit_entry_single(
        self,
        trade_id: str,
        ticker: str,
        shares: float,
        side: str,
        module: str,
    ) -> dict:
        """
        Submit a market order for a single stock.

        Parameters
        ----------
        trade_id : str   -- unique trade id from the tracker
        ticker   : str   -- symbol (e.g. "AAPL")
        shares   : float -- will be floored to int
        side     : str   -- "buy" or "sell"
        module   : str   -- "momentum" or "bear"

        Returns
        -------
        dict with keys: status, order_id, error
        """
        safe = {"status": "failed", "order_id": None, "error": None}

        if not self.is_active:
            safe["error"] = "broker inactive"
            return safe

        # Idempotency check
        if trade_id in self._state.get("orders", {}):
            existing = self._state["orders"][trade_id]
            if existing.get("status") not in ("failed",):
                self._log(f"Skipping duplicate submit for trade_id={trade_id}")
                return {
                    "status": existing.get("status", "submitted"),
                    "order_id": (existing.get("alpaca_order_ids") or [None])[0],
                    "error": None,
                }

        int_shares = int(math.floor(abs(shares)))
        if int_shares <= 0:
            safe["error"] = f"invalid share count after floor: {shares}"
            return safe

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            req = MarketOrderRequest(
                symbol=ticker,
                qty=int_shares,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            order = None
            last_err = None
            for attempt in range(1, self._max_retries + 1):
                try:
                    order = self._client.submit_order(req)
                    break
                except Exception as retry_exc:
                    last_err = str(retry_exc)
                    self._log(
                        f"Order attempt {attempt}/{self._max_retries} failed "
                        f"for {ticker}: {retry_exc}"
                    )

            if order is None:
                safe["error"] = last_err or "all retries exhausted"
                self._state.setdefault("orders", {})[trade_id] = {
                    "module": module,
                    "alpaca_order_ids": [],
                    "status": "failed",
                    "submitted_at": self._now_iso(),
                    "error": safe["error"],
                    "legs": [],
                }
                self._save_broker_state()
                return safe

            order_id = str(order.id)
            self._state.setdefault("orders", {})[trade_id] = {
                "module": module,
                "alpaca_order_ids": [order_id],
                "status": "submitted",
                "submitted_at": self._now_iso(),
                "error": None,
                "legs": [
                    {
                        "symbol": ticker,
                        "side": side.lower(),
                        "qty": int_shares,
                        "order_id": order_id,
                        "status": str(order.status),
                    }
                ],
            }
            self._save_broker_state()
            self._log(f"Submitted {side.upper()} {int_shares} {ticker} -- order {order_id}")
            return {"status": "submitted", "order_id": order_id, "error": None}

        except Exception as exc:
            safe["error"] = str(exc)
            self._log(f"ERROR: submit_entry_single failed -- {exc}")
            return safe

    # ------------------------------------------------------------------
    #  Pairs entry (two legs)
    # ------------------------------------------------------------------

    def submit_entry_pairs(self, trade_id: str, info: dict) -> dict:
        """
        Submit a pairs trade (two market orders).

        Parameters
        ----------
        trade_id : str
        info     : dict with keys:
            a         : str   -- stock A symbol
            b         : str   -- stock B symbol
            direction : str   -- "LONG" or "SHORT"
            shares_a  : float -- shares of stock A
            shares_b  : float -- shares of stock B

        Direction logic:
            LONG  = BUY a  + SELL b (short sell)
            SHORT = SELL a (short sell) + BUY b

        Returns
        -------
        dict with keys: status, order_ids, error
        """
        safe = {"status": "failed", "order_ids": [], "error": None}

        if not self.is_active:
            safe["error"] = "broker inactive"
            return safe

        # Idempotency
        if trade_id in self._state.get("orders", {}):
            existing = self._state["orders"][trade_id]
            if existing.get("status") not in ("failed",):
                self._log(f"Skipping duplicate pairs submit for trade_id={trade_id}")
                return {
                    "status": existing.get("status", "submitted"),
                    "order_ids": existing.get("alpaca_order_ids", []),
                    "error": None,
                }

        stock_a = info.get("a", "")
        stock_b = info.get("b", "")
        direction = info.get("direction", "").upper()
        shares_a = int(math.floor(abs(info.get("shares_a", 0))))
        shares_b = int(math.floor(abs(info.get("shares_b", 0))))

        if shares_a <= 0 or shares_b <= 0:
            safe["error"] = f"invalid shares after floor: a={shares_a}, b={shares_b}"
            return safe

        if direction == "LONG":
            side_a, side_b = "buy", "sell"
        elif direction == "SHORT":
            side_a, side_b = "sell", "buy"
        else:
            safe["error"] = f"unknown direction: {direction}"
            return safe

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            legs_plan = [
                (stock_a, shares_a, side_a),
                (stock_b, shares_b, side_b),
            ]

            order_ids = []
            legs_detail = []
            any_failed = False

            for symbol, qty, side in legs_plan:
                order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
                leg_order = None
                last_err = None
                for attempt in range(1, self._max_retries + 1):
                    try:
                        leg_order = self._client.submit_order(req)
                        break
                    except Exception as retry_exc:
                        last_err = str(retry_exc)
                        self._log(
                            f"Pairs leg attempt {attempt}/{self._max_retries} "
                            f"failed for {symbol}: {retry_exc}"
                        )

                if leg_order is not None:
                    oid = str(leg_order.id)
                    order_ids.append(oid)
                    legs_detail.append({
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "order_id": oid,
                        "status": str(leg_order.status),
                    })
                    self._log(f"Pairs leg: {side.upper()} {qty} {symbol} -- order {oid}")
                else:
                    any_failed = True
                    legs_detail.append({
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "order_id": None,
                        "status": "failed",
                    })

            if len(order_ids) == 0:
                status = "failed"
            elif any_failed:
                status = "partial"
            else:
                status = "submitted"

            self._state.setdefault("orders", {})[trade_id] = {
                "module": "pairs",
                "alpaca_order_ids": order_ids,
                "status": status,
                "submitted_at": self._now_iso(),
                "error": last_err if any_failed else None,
                "legs": legs_detail,
            }
            self._save_broker_state()

            return {
                "status": status,
                "order_ids": order_ids,
                "error": last_err if any_failed else None,
            }

        except Exception as exc:
            safe["error"] = str(exc)
            self._log(f"ERROR: submit_entry_pairs failed -- {exc}")
            return safe

    # ------------------------------------------------------------------
    #  Close single position
    # ------------------------------------------------------------------

    def close_position_single(self, trade_id: str, ticker: str) -> dict:
        """
        Close a position by symbol.

        Returns dict with keys: status, error
        """
        safe = {"status": "failed", "error": None}

        if not self.is_active:
            safe["error"] = "broker inactive"
            return safe

        try:
            self._client.close_position(symbol_or_asset_id=ticker)
            self._log(f"Closed position: {ticker} (trade_id={trade_id})")

            # Update broker state if trade exists
            if trade_id in self._state.get("orders", {}):
                self._state["orders"][trade_id]["status"] = "closed"
                self._save_broker_state()

            return {"status": "closed", "error": None}

        except Exception as exc:
            safe["error"] = str(exc)
            self._log(f"ERROR: close_position_single({ticker}) failed -- {exc}")
            return safe

    # ------------------------------------------------------------------
    #  Close pairs position (both legs)
    # ------------------------------------------------------------------

    def close_position_pairs(
        self, trade_id: str, stock_a: str, stock_b: str
    ) -> dict:
        """
        Close both legs of a pairs position.

        Returns dict with keys: status, details
        """
        safe = {"status": "failed", "details": []}

        if not self.is_active:
            safe["details"].append({"error": "broker inactive"})
            return safe

        details = []
        any_failed = False

        for symbol in (stock_a, stock_b):
            try:
                self._client.close_position(symbol_or_asset_id=symbol)
                details.append({"symbol": symbol, "status": "closed", "error": None})
                self._log(f"Closed pairs leg: {symbol}")
            except Exception as exc:
                any_failed = True
                details.append({"symbol": symbol, "status": "failed", "error": str(exc)})
                self._log(f"ERROR: close pairs leg {symbol} failed -- {exc}")

        if not any_failed:
            status = "closed"
        elif len([d for d in details if d["status"] == "closed"]) > 0:
            status = "partial"
        else:
            status = "failed"

        # Update broker state
        if trade_id in self._state.get("orders", {}):
            self._state["orders"][trade_id]["status"] = status
            self._save_broker_state()

        return {"status": status, "details": details}

    # ------------------------------------------------------------------
    #  Emergency liquidation
    # ------------------------------------------------------------------

    def liquidate_all(self) -> dict:
        """
        Emergency: close all positions and cancel all open orders.

        Returns dict with keys: status, positions_closed
        """
        safe = {"status": "failed", "positions_closed": 0}

        if not self.is_active:
            safe["status"] = "failed"
            return safe

        try:
            # Get current position count before liquidating
            positions = self._client.get_all_positions()
            count = len(positions)

            self._client.close_all_positions(cancel_orders=True)
            self._log(f"LIQUIDATED all positions ({count} positions closed)")

            return {"status": "liquidated", "positions_closed": count}

        except Exception as exc:
            self._log(f"ERROR: liquidate_all failed -- {exc}")
            safe["positions_closed"] = 0
            return safe

    # ------------------------------------------------------------------
    #  Position sync
    # ------------------------------------------------------------------

    def sync_positions(self) -> dict:
        """
        Compare Alpaca positions vs paper trades from all 3 trackers.

        Returns dict with keys: synced, orphan_alpaca, missing_alpaca, qty_mismatch
        """
        safe = {
            "synced": False,
            "orphan_alpaca": [],
            "missing_alpaca": [],
            "qty_mismatch": [],
        }

        if not self.is_active:
            return safe

        try:
            # Import trackers lazily to avoid circular imports
            import trade_tracker
            import momentum_tracker
            import bear_tracker

            # --- Alpaca side: symbol -> total abs qty ---
            alpaca_positions = self._client.get_all_positions()
            alpaca_map = {}
            for pos in alpaca_positions:
                sym = pos.symbol
                qty = abs(int(pos.qty))
                alpaca_map[sym] = alpaca_map.get(sym, 0) + qty

            # --- Paper side: symbol -> total abs qty ---
            paper_map = {}

            # Pairs trades: each has stock_a and stock_b with shares_a, shares_b
            for t in trade_tracker.get_open_trades():
                sym_a = str(t.get("stock_a", "")).strip()
                sym_b = str(t.get("stock_b", "")).strip()
                try:
                    qty_a = abs(int(float(t.get("shares_a", 0))))
                except (ValueError, TypeError):
                    qty_a = 0
                try:
                    qty_b = abs(int(float(t.get("shares_b", 0))))
                except (ValueError, TypeError):
                    qty_b = 0
                if sym_a:
                    paper_map[sym_a] = paper_map.get(sym_a, 0) + qty_a
                if sym_b:
                    paper_map[sym_b] = paper_map.get(sym_b, 0) + qty_b

            # Momentum trades: single ticker + shares
            for t in momentum_tracker.get_open_mom_trades():
                sym = str(t.get("ticker", "")).strip()
                try:
                    qty = abs(int(float(t.get("shares", 0))))
                except (ValueError, TypeError):
                    qty = 0
                if sym:
                    paper_map[sym] = paper_map.get(sym, 0) + qty

            # Bear trades: single ticker + shares
            for t in bear_tracker.get_open_bear_trades():
                sym = str(t.get("ticker", "")).strip()
                try:
                    qty = abs(int(float(t.get("shares", 0))))
                except (ValueError, TypeError):
                    qty = 0
                if sym:
                    paper_map[sym] = paper_map.get(sym, 0) + qty

            # --- Compute differences ---
            all_symbols = set(alpaca_map.keys()) | set(paper_map.keys())
            orphan_alpaca = []
            missing_alpaca = []
            qty_mismatch = []

            for sym in sorted(all_symbols):
                a_qty = alpaca_map.get(sym, 0)
                p_qty = paper_map.get(sym, 0)

                if a_qty > 0 and p_qty == 0:
                    orphan_alpaca.append(sym)
                elif a_qty == 0 and p_qty > 0:
                    missing_alpaca.append(sym)
                elif a_qty != p_qty:
                    qty_mismatch.append(
                        {"symbol": sym, "alpaca_qty": a_qty, "paper_qty": p_qty}
                    )

            synced = (
                len(orphan_alpaca) == 0
                and len(missing_alpaca) == 0
                and len(qty_mismatch) == 0
            )

            self._state["last_sync"] = self._now_iso()
            self._save_broker_state()

            if synced:
                self._log("Positions synced -- Alpaca matches paper trades")
            else:
                if orphan_alpaca:
                    self._log(f"Orphan Alpaca positions (no paper trade): {orphan_alpaca}")
                if missing_alpaca:
                    self._log(f"Missing Alpaca positions (paper exists): {missing_alpaca}")
                if qty_mismatch:
                    self._log(f"Quantity mismatches: {qty_mismatch}")

            return {
                "synced": synced,
                "orphan_alpaca": orphan_alpaca,
                "missing_alpaca": missing_alpaca,
                "qty_mismatch": qty_mismatch,
            }

        except Exception as exc:
            self._log(f"ERROR: sync_positions failed -- {exc}")
            return safe

    # ------------------------------------------------------------------
    #  Account summary
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict | None:
        """
        Fetch account equity, buying power, and cash from Alpaca.

        Returns dict with keys: equity, buying_power, cash, paper  (or None on failure)
        """
        if not self.is_active:
            return None

        try:
            acct = self._client.get_account()
            summary = {
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
                "paper": self._paper,
            }
            return summary

        except Exception as exc:
            self._log(f"ERROR: get_account_summary failed -- {exc}")
            return None


# ---------------------------------------------------------------------------
#  CLI test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("-" * 60)
    print("  Alpaca Broker -- Connectivity Test")
    print("-" * 60)

    broker = AlpacaBroker()

    if not broker.is_active:
        print("\nBroker is INACTIVE (LIVE_TRADING_ENABLED=False or missing keys)")
        print("Set LIVE_TRADING_ENABLED=True and provide API keys to test.")
    else:
        summary = broker.get_account_summary()
        if summary:
            print(f"\n  Account Type : {'Paper' if summary['paper'] else 'LIVE'}")
            print(f"  Equity       : ${summary['equity']:,.2f}")
            print(f"  Buying Power : ${summary['buying_power']:,.2f}")
            print(f"  Cash         : ${summary['cash']:,.2f}")
        else:
            print("\n  Could not retrieve account summary.")

        mkt = broker.is_market_open()
        print(f"\n  Market Open  : {'Yes' if mkt else 'No'}")

        sync = broker.sync_positions()
        print(f"  Positions Synced: {sync['synced']}")
        if sync["orphan_alpaca"]:
            print(f"  Orphan Alpaca   : {sync['orphan_alpaca']}")
        if sync["missing_alpaca"]:
            print(f"  Missing Alpaca  : {sync['missing_alpaca']}")
        if sync["qty_mismatch"]:
            print(f"  Qty Mismatch    : {sync['qty_mismatch']}")

    print("\n" + "-" * 60)
