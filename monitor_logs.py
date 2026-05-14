"""
monitor_logs.py — monitora o crescimento da pasta logs durante execução Docker.

Modos de uso:
  python monitor_logs.py collect            # coleta a cada 10 s (padrão)
  python monitor_logs.py collect --interval 30 --output meu_run.csv
  python monitor_logs.py plot               # plota o último arquivo de coleta
  python monitor_logs.py plot --input meu_run.csv --output grafico.png
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path


LOGS_DIR = Path(__file__).parent / "logs"
DEFAULT_INTERVAL = 10  # segundos entre amostras
DEFAULT_DATA_FILE = Path(__file__).parent / "log_growth_data.csv"
DATA_HEADER = ["elapsed_s", "timestamp", "total_kb", "n_jpg", "n_avi", "n_csv", "n_other"]


# ─── COLETA ──────────────────────────────────────────────────────────────────

def _snapshot(logs_dir: Path) -> dict:
    total_bytes = 0
    counts = {"jpg": 0, "avi": 0, "csv": 0, "other": 0}
    if logs_dir.exists():
        for f in logs_dir.iterdir():
            if f.is_file():
                total_bytes += f.stat().st_size
                ext = f.suffix.lower().lstrip(".")
                if ext in counts:
                    counts[ext] += 1
                else:
                    counts["other"] += 1
    return {
        "total_kb": total_bytes / 1024,
        "n_jpg": counts["jpg"],
        "n_avi": counts["avi"],
        "n_csv": counts["csv"],
        "n_other": counts["other"],
    }


def collect(interval: int, output: Path, logs_dir: Path) -> None:
    print(f"Monitorando: {logs_dir}")
    print(f"Intervalo:   {interval} s")
    print(f"Saída:       {output}")
    print("Pressione Ctrl+C para encerrar.\n")

    start = time.time()
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DATA_HEADER)
        writer.writeheader()

        while True:
            elapsed = round(time.time() - start, 1)
            snap = _snapshot(logs_dir)
            row = {
                "elapsed_s":  elapsed,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
                "total_kb":   round(snap["total_kb"], 1),
                "n_jpg":      snap["n_jpg"],
                "n_avi":      snap["n_avi"],
                "n_csv":      snap["n_csv"],
                "n_other":    snap["n_other"],
            }
            writer.writerow(row)
            f.flush()
            print(f"[{row['timestamp']}] {row['total_kb']:>10.1f} KB  "
                  f"jpg={row['n_jpg']}  avi={row['n_avi']}  csv={row['n_csv']}")
            time.sleep(interval)


# ─── PLOT ────────────────────────────────────────────────────────────────────

def plot(input_file: Path, output_file: Path | None) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise SystemExit("matplotlib não encontrado. Instale com: pip install matplotlib")

    rows = []
    with open(input_file, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k not in ("timestamp",) else v for k, v in row.items()})

    if not rows:
        raise SystemExit("Arquivo de dados vazio.")

    elapsed  = [r["elapsed_s"] / 60 for r in rows]   # minutos
    total_mb = [r["total_kb"] / 1024 for r in rows]
    n_jpg    = [r["n_jpg"]  for r in rows]
    n_avi    = [r["n_avi"]  for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Crescimento da pasta logs durante execução Docker", fontsize=13, fontweight="bold")

    # — painel superior: tamanho total —
    ax1.fill_between(elapsed, total_mb, alpha=0.25, color="steelblue")
    ax1.plot(elapsed, total_mb, color="steelblue", linewidth=1.8, label="Tamanho total")
    ax1.set_ylabel("Tamanho (MB)")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f MB"))
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # adiciona rótulo do valor final
    if total_mb:
        ax1.annotate(
            f"{total_mb[-1]:.2f} MB",
            xy=(elapsed[-1], total_mb[-1]),
            xytext=(-40, 8), textcoords="offset points",
            fontsize=8, color="steelblue",
        )

    # — painel inferior: contagem de arquivos —
    ax2.step(elapsed, n_jpg, where="post", color="darkorange", linewidth=1.5, label="Snapshots (.jpg)")
    ax2.step(elapsed, n_avi, where="post", color="crimson",    linewidth=1.5, label="Vídeos (.avi)",
             linestyle="--")
    ax2.set_ylabel("Quantidade de arquivos")
    ax2.set_xlabel("Tempo de execução (min)")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # eixo X em minutos com marca a cada 5 min
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g min"))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Gráfico salvo em: {output_file}")
    else:
        plt.show()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitora crescimento da pasta logs e plota gráfico.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # subcomando collect
    p_collect = sub.add_parser("collect", help="Coleta dados de crescimento da pasta logs")
    p_collect.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                           metavar="SEG", help=f"Intervalo entre amostras em segundos (padrão: {DEFAULT_INTERVAL})")
    p_collect.add_argument("--output", type=Path, default=DEFAULT_DATA_FILE,
                           metavar="ARQUIVO", help="CSV de saída (padrão: log_growth_data.csv)")
    p_collect.add_argument("--logs-dir", type=Path, default=LOGS_DIR,
                           metavar="DIR", help=f"Pasta a monitorar (padrão: {LOGS_DIR})")

    # subcomando plot
    p_plot = sub.add_parser("plot", help="Plota gráfico a partir dos dados coletados")
    p_plot.add_argument("--input", type=Path, default=DEFAULT_DATA_FILE,
                        metavar="ARQUIVO", help="CSV gerado pelo modo collect")
    p_plot.add_argument("--output", type=Path, default=None,
                        metavar="IMAGEM", help="Salva gráfico em arquivo PNG (opcional; padrão: exibe na tela)")

    args = parser.parse_args()

    if args.mode == "collect":
        try:
            collect(args.interval, args.output, args.logs_dir)
        except KeyboardInterrupt:
            print("\nColeta encerrada.")
    elif args.mode == "plot":
        if not args.input.exists():
            raise SystemExit(f"Arquivo não encontrado: {args.input}\nRode primeiro: python monitor_logs.py collect")
        plot(args.input, args.output)


if __name__ == "__main__":
    main()
