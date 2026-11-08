"""HTML reports: one per month, plus a summary of the whole simulation."""

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402

BLUE, ORANGE, GREY, RED = "#2563eb", "#ea580c", "#9ca3af", "#dc2626"

STYLE = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;margin:32px auto;
     padding:0 20px;color:#111827;line-height:1.5}
h1{font-size:26px;margin-bottom:4px} h2{font-size:19px;margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}
.sub{color:#6b7280;margin-top:0}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.card{border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;min-width:140px}
.card .v{font-size:22px;font-weight:600} .card .l{color:#6b7280;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}
th,td{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:left} th{background:#f9fafb}
.bad{color:#dc2626;font-weight:600} .ok{color:#16a34a}
.action{background:#eff6ff;border-left:4px solid #2563eb;padding:10px 14px;border-radius:4px}
img{max-width:100%}
"""


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _drift_plots(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[str]:
    images = []
    for f in features[:4]:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(reference[f], bins=20, alpha=0.6, density=True, color=GREY, label="training data")
        ax.hist(current[f], bins=20, alpha=0.6, density=True, color=ORANGE, label="this month")
        ax.set_title(f)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        images.append(_img(fig))
    return images


MONTH_TEMPLATE = Template("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Churn monitoring: month {{ month }}</title><style>{{ style }}</style></head><body>
<h1>Churn monitoring: month {{ month }}</h1>
<p class="sub">Scored with model v{{ version }} · {{ n }} customers</p>
<div class="cards">
 <div class="card"><div class="v">{{ perf.pr_auc }}</div><div class="l">PR-AUC (expected {{ expected_pr_auc }})</div></div>
 <div class="card"><div class="v">{{ perf.recall }}</div><div class="l">Recall</div></div>
 <div class="card"><div class="v">{{ perf.precision }}</div><div class="l">Precision</div></div>
 <div class="card"><div class="v {{ 'bad' if drift.data_drift else 'ok' }}">{{ (drift.drift_share*100)|round|int }}%</div><div class="l">Features drifted</div></div>
 <div class="card"><div class="v">{{ "{:,.0f}".format(perf.campaign.campaign_profit) }}</div><div class="l">Campaign profit</div></div>
 <div class="card"><div class="v">{{ "{:,.0f}".format(perf.campaign.random_profit) }}</div><div class="l">Random targeting</div></div>
</div>
<div class="action"><b>Action:</b> {{ action }}</div>
<h2>Retention campaign</h2>
<p>{{ perf.campaign.n_targeted }} customers targeted (threshold {{ perf.threshold }});
{{ perf.campaign.churners_reached }} of {{ perf.confusion_matrix.tp + perf.confusion_matrix.fn }} churners reached.</p>
<h2>Data drift by feature</h2>
<table><tr><th>Feature</th><th>PSI</th><th>KS p-value</th><th>Training mean</th><th>This month</th><th>Drifted</th></tr>
{% for r in drift_rows %}<tr><td>{{ r.feature }}</td><td>{{ r.psi }}</td><td>{{ r.ks_pvalue if r.ks_pvalue is not none else "–" }}</td>
<td>{{ r.reference_mean }}</td><td>{{ r.current_mean }}</td><td class="{{ 'bad' if r.drifted else '' }}">{{ "yes" if r.drifted else "no" }}</td></tr>{% endfor %}
</table>
{% if plots %}<h2>Biggest shifts</h2>{% for p in plots %}<img src="data:image/png;base64,{{ p }}">{% endfor %}{% endif %}
</body></html>""")


def month_report(path: Path, month: int, version: int, n: int, perf: dict, expected_pr_auc,
                 drift: dict, table: pd.DataFrame, reference: pd.DataFrame, current: pd.DataFrame,
                 action: str) -> Path:
    shifted = table.loc[table["psi"] > 0.1, "feature"].head(4).tolist()
    rows = table.astype(object).where(table.notna(), None).to_dict("records")
    html = MONTH_TEMPLATE.render(style=STYLE, month=month, version=version, n=n, perf=perf,
                                 expected_pr_auc=expected_pr_auc, drift=drift, action=action,
                                 drift_rows=rows,
                                 plots=_drift_plots(reference, current, shifted))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


SUMMARY_TEMPLATE = Template("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Churn MLOps simulation summary</title><style>{{ style }}</style></head><body>
<h1>Churn MLOps: simulation summary</h1>
<p class="sub">{{ n_months }} monthly cycles · {{ n_versions }} model versions · champion: v{{ champion }}</p>
<div class="cards">
 <div class="card"><div class="v">{{ "{:,.0f}".format(total_profit) }}</div><div class="l">Total campaign profit</div></div>
 <div class="card"><div class="v">{{ "{:,.0f}".format(total_random) }}</div><div class="l">Random targeting</div></div>
 <div class="card"><div class="v">{{ "{:,.0f}".format(total_profit - total_random) }}</div><div class="l">Gained by the model</div></div>
 <div class="card"><div class="v">{{ n_retrains }}</div><div class="l">Retrains triggered</div></div>
</div>
<h2>Model performance and drift</h2><img src="data:image/png;base64,{{ perf_plot }}">
<h2>Money saved</h2><img src="data:image/png;base64,{{ profit_plot }}">
<h2>Month by month</h2>
<table><tr><th>Month</th><th>Model</th><th>PR-AUC</th><th>Recall</th><th>Drift share</th><th>Profit</th><th>Random</th><th>Action</th><th>Report</th></tr>
{% for r in rows %}<tr><td>{{ r.month }}</td><td>v{{ r.model_version }}</td><td>{{ r.pr_auc }}</td><td>{{ r.recall }}</td>
<td class="{{ 'bad' if r.data_drift else '' }}">{{ r.drift_share }}</td><td>{{ "{:,.0f}".format(r.campaign_profit) }}</td>
<td>{{ "{:,.0f}".format(r.random_profit) }}</td><td>{{ r.action }}</td><td><a href="{{ r.report_name }}">open</a></td></tr>{% endfor %}
</table>
<h2>Model versions</h2>
<table><tr><th>Version</th><th>Registered</th><th>Test PR-AUC</th><th>Reason</th><th>Champion</th></tr>
{% for v in versions %}<tr><td>v{{ v.version }}</td><td>{{ v.registered_at }}</td><td>{{ v.test_pr_auc }}</td><td>{{ v.reason }}</td><td>{{ "★" if v.champion else "" }}</td></tr>{% endfor %}
</table></body></html>""")


def summary_report(path: Path, monitoring: pd.DataFrame, versions: list[dict], champion: int,
                   drift_threshold: float) -> Path:
    m = monitoring.sort_values("month")

    fig, ax1 = plt.subplots(figsize=(9, 3.6))
    ax1.plot(m["month"], m["pr_auc"], marker="o", color=BLUE, label="PR-AUC")
    for _, r in m.iterrows():
        ax1.annotate(f"v{int(r.model_version)}", (r.month, r.pr_auc), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=BLUE)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("month")
    ax1.set_ylabel("PR-AUC", color=BLUE)
    ax2 = ax1.twinx()
    ax2.bar(m["month"], m["drift_share"], alpha=0.25, color=ORANGE, label="share of features drifted")
    ax2.axhline(drift_threshold, ls="--", color=RED, lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("drift share", color=ORANGE)
    ax1.set_xticks(m["month"])
    for a in (ax1, ax2):
        a.spines[["top"]].set_visible(False)
    perf_plot = _img(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    w = 0.38
    ax.bar(m["month"] - w / 2, m["campaign_profit"], w, color=BLUE, label="model-targeted campaign")
    ax.bar(m["month"] + w / 2, m["random_profit"], w, color=GREY, label="random targeting")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(m["month"])
    ax.set_xlabel("month")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    profit_plot = _img(fig)

    rows = m.assign(report_name=m["report_path"].map(lambda p: Path(p).name)).to_dict("records")
    html = SUMMARY_TEMPLATE.render(
        style=STYLE, n_months=len(m), n_versions=len(versions), champion=champion,
        total_profit=m["campaign_profit"].sum(), total_random=m["random_profit"].sum(),
        n_retrains=int(m["action"].str.startswith("Retrained").sum()),
        perf_plot=perf_plot, profit_plot=profit_plot, rows=rows, versions=versions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path
