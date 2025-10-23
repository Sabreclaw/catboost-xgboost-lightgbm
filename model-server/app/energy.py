import os
import re
import csv
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class EnergyProfiler:
    """
    Wrapper around the energibridge profiler.
    - start(): spawns a long-running process writing metrics to <name>.csv
    - stop(): terminates it, parses stdout for energy/duration, and summarizes CSV metrics
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._cmd: Optional[str] = None
        self._lock = threading.Lock()
        self._outfile: Optional[str] = None

    def _safe_name(self, model_name: Optional[str]) -> str:
        name = (model_name or os.getenv("LOAD_MODEL") or "model").strip()
        return (
            "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")).strip()
            or "model"
        )

    def _format_cmd_and_out(self, model_name: Optional[str]) -> Tuple[str, str]:
        safe = self._safe_name(model_name)
        profiler_bin = os.getenv("PROFILER_PATH") or "energibridge"
        profiler_bin_quoted = shlex.quote(profiler_bin)
        out_file = str((Path.cwd() / f"{safe}.csv").resolve())
        cmd = f"{profiler_bin_quoted} --summary --interval 200 --output {safe}.csv sleep 1e9"
        return cmd, out_file

    def start(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                raise RuntimeError("Energy profiler already running")
            cmd, out_file = self._format_cmd_and_out(model_name)
            if os.path.isfile(out_file):
                os.remove(out_file)
            try:
                proc = subprocess.Popen(
                    shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except FileNotFoundError as e:
                # bubble up to caller
                raise e
            self._proc = proc
            self._cmd = cmd
            self._outfile = out_file
            return {"pid": proc.pid, "cmd": cmd, "output_file": out_file}

    @staticmethod
    def _parse_energy_stdout(
        stdout_text: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        # Looks like: "Energy consumption in joules: 3719.35595 for 65.10 sec of execution."
        energy = None
        duration = None
        if stdout_text:
            m_e = re.search(
                r"Energy consumption in joules:\s*([0-9]+\.?[0-9]*)", stdout_text
            )
            if m_e:
                try:
                    energy = float(m_e.group(1))
                except ValueError:
                    energy = None
            m_t = re.search(r"for\s*([0-9]+\.?[0-9]*)\s*sec", stdout_text)
            if m_t:
                try:
                    duration = float(m_t.group(1))
                except ValueError:
                    duration = None
        return energy, duration

    @staticmethod
    def _summarize_csv(path: str) -> Dict[str, Optional[float]]:
        if not path or not os.path.isfile(path):
            return {
                "mean_cpu_usage": None,
                "mean_memory_gb": None,
                "mean_power_watts": None,
            }
        cpu_sum = 0.0
        cpu_count = 0
        mem_sum_gb = 0.0
        mem_count = 0
        pwr_sum = 0.0
        pwr_count = 0
        try:
            with open(path, "r", newline="") as f:
                r = csv.DictReader(f)
                # Identify columns once from header
                fieldnames = r.fieldnames or []
                cpu_cols = [c for c in fieldnames if c.startswith("CPU_USAGE_")]
                # Power column can be named "SYSTEM_POWER (Watts)" or similar
                pwr_cols = [c for c in fieldnames if "SYSTEM_POWER" in c.upper()]
                mem_col = "USED_MEMORY" if "USED_MEMORY" in fieldnames else None
                for row in r:
                    # CPU usage: accumulate all CPU_USAGE_* values
                    if cpu_cols:
                        vals = []
                        for c in cpu_cols:
                            v = row.get(c)
                            if v is None or v == "":
                                continue
                            try:
                                vals.append(float(v))
                            except ValueError:
                                continue
                        if vals:
                            cpu_sum += sum(vals)
                            cpu_count += len(vals)
                    # Memory usage: bytes → GB
                    if mem_col:
                        v = row.get(mem_col)
                        if v not in (None, ""):
                            try:
                                mem_sum_gb += float(v) / (1000**3)
                                mem_count += 1
                            except ValueError:
                                pass
                    # System power (Watts)
                    for pc in pwr_cols:
                        v = row.get(pc)
                        if v not in (None, ""):
                            try:
                                pwr_sum += float(v)
                                pwr_count += 1
                            except ValueError:
                                pass
        except Exception:
            # On any parse error, return Nones to avoid failing the API
            return {
                "mean_cpu_usage": None,
                "mean_memory_gb": None,
                "mean_power_watts": None,
            }
        mean_cpu = (cpu_sum / cpu_count) if cpu_count > 0 else None
        mean_mem_gb = (mem_sum_gb / mem_count) if mem_count > 0 else None
        mean_pwr = (pwr_sum / pwr_count) if pwr_count > 0 else None
        return {
            "mean_cpu_usage": mean_cpu,
            "mean_memory_gb": mean_mem_gb,
            "mean_power_watts": mean_pwr,
        }

    def stop(self, timeout: float = 5.0) -> Dict[str, Any]:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                raise RuntimeError("Energy profiler is not running")
            proc = self._proc
            cmd = self._cmd
            outfile = self._outfile
            try:
                proc.terminate()
                try:
                    out, err = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = proc.communicate(timeout=timeout)
            finally:
                self._proc = None
                self._cmd = None
                self._outfile = None

            def _clip(b: Optional[bytes]) -> str:
                if not b:
                    return ""
                s = b.decode(errors="replace")
                if len(s) > 4000:
                    return s[:4000] + "... [truncated]"
                return s

            stdout_text = _clip(out)
            stderr_text = _clip(err)
            energy_j, duration_s = self._parse_energy_stdout(stdout_text)
            csv_metrics = self._summarize_csv(outfile or "")
            # Fallback: if mean power missing but energy and duration are available, compute it
            if (
                (csv_metrics.get("mean_power_watts") is None)
                and (energy_j is not None)
                and (duration_s is not None)
                and (duration_s > 0)
            ):
                try:
                    csv_metrics["mean_power_watts"] = float(energy_j) / float(
                        duration_s
                    )
                except Exception:
                    pass

            # Convenience top-level metrics
            result: Dict[str, Any] = {
                "stopped": True,
                "cmd": cmd,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "output_file": outfile,
                "metrics": {
                    **csv_metrics,
                    "energy_joules": energy_j,
                    "duration_seconds": duration_s,
                },
                # Flattened keys for convenience
                "energy_joules": energy_j,
                "duration_seconds": duration_s,
                "mean_cpu_usage": csv_metrics.get("mean_cpu_usage"),
                "mean_memory_gb": csv_metrics.get("mean_memory_gb"),
                "mean_power_watts": csv_metrics.get("mean_power_watts"),
            }
            return result
