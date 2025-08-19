
# summarize_cases.py
# Summarize JSON cards in ./cases into a Markdown table (prints + writes cases/CASE_SUMMARY.md)
import json
from pathlib import Path

def main():
    cases_dir = Path("cases")
    out_md = cases_dir / "CASE_SUMMARY.md"
    rows = []
    for p in sorted(cases_dir.glob("*.json")):
        d = json.loads(p.read_text())
        m = d["metrics"]
        rows.append((p.stem, d["assumed_prevalence"], d["threshold"],
                     m["FN"], m["fnr"], m["wlr"], m["npv"], m.get("auc", None)))

    header = "| Case | Assumed prevalence | Threshold | FN | FNR | WLR | NPV | AUROC |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = []
    for name, prev, thr, fn, fnr, wlr, npv, auc in rows:
        lines.append(f"| {name} | {prev*100:.2f}% | {thr:.4f} | {fn} | {fnr*100:.2f}% | {wlr*100:.2f}% | {npv*100:.2f}% | {'' if auc is None else f'{auc:.4f}'} |")
    md = header + "\n".join(lines) + "\n"

    print(md)
    out_md.write_text(md)
    print(f"Saved {out_md.resolve()}")

if __name__ == "__main__":
    main()
