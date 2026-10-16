# -*- coding: utf-8 -*-
"""Stage_1 性能图 v4：保留 4 张（fig2 修口径 / fig3 / fig5 / fig6 补档）+ 新增 3 张（CDF / 尾延迟瀑布 / rerank 重排幅度）。

配色：用户品牌五色（浅粉/紫/青绿/中灰/浅蓝）+ 2 衍生 tint。
口径：检索=四路并行净耗时 max(检索_*)；planner_multi 路径检索耗时记入「多文档规划+检索」。
"""
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent
DATA = BASE.parent

# 品牌五色 + 衍生 tint
BLUE = "#89B2F4"          # 串行族·面 / 规划改写 / meta
TEAL = "#018C94"          # 串行族·线 / P95 / 检索 / sql
PINK = "#EBB9BC"          # 并发族·面 / 精排rerank / corpus
PURPLE = "#764D9C"        # 并发族·线 / P99 / LLM生成
GRAY = "#7F7F7F"          # P50 / parent / 落选线
TEAL_TINT = "#7FD3D8"     # RRF融合
PURPLE_TINT = "#B89AD0"   # 多文档规划+检索
BLACK = "#000000"

ROWS = ["规划改写", "检索", "RRF融合", "规划执行+检索", "精排rerank", "parent上下文", "LLM生成"]
ROW_COLORS = [BLUE, TEAL, TEAL_TINT, PURPLE_TINT, PINK, GRAY, PURPLE]
HATCHES = ["/", "\\", "//", "x", "o", "+", "*"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica"],
    "font.size": 13,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.5,
    "svg.fonttype": "path",
})


def load_records():
    out = {}
    for name, key in [("serial_latency.json", "串行"), ("concurrent_latency.json", "并发")]:
        out[key] = json.loads((DATA / name).read_text(encoding="utf-8"))
    return out


def load_trace():
    return json.loads((DATA / "serial_trace.json").read_text(encoding="utf-8"))


def corrected_stages(r):
    ret = {row: float(r.get(row, 0.0)) for row in ROWS}
    ret["检索"] = max(float(r.get(f"检索_{k}", 0.0)) for k in ("faiss", "bm25", "meta", "sql"))
    if ret["检索"] == 0.0:
        ret["RRF融合"] = 0.0
        ret["规划执行+检索"] = float(r.get("RRF融合", 0.0))
    else:
        ret["规划执行+检索"] = 0.0
    return ret


def short_label(id_):
    parts = id_.removesuffix(".md").split("_")
    org = parts[2] if len(parts) > 2 else id_
    title = parts[4] if len(parts) > 4 else parts[-1]
    return f"{org}·{title[:8]}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(BASE / name)
    plt.close(fig)
    print(name)


# ── 图2 占比堆叠（修正口径 7 行）──
def fig1():
    recs = load_records()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, key in enumerate(["串行", "并发"]):
        corrected = [corrected_stages(r) for r in recs[key]]
        totals = [sum(c.values()) or 1.0 for c in corrected]
        fracs = np.array([[c[s] / t for s in ROWS] for c, t in zip(corrected, totals)])
        mean_frac = fracs.mean(axis=0)
        bottom = 0.0
        for si, s in enumerate(ROWS):
            ax.bar(i, mean_frac[si], 0.6, bottom=bottom,
                   color=ROW_COLORS[si], hatch=HATCHES[si], edgecolor="black",
                   linewidth=1.5, alpha=0.9)
            if mean_frac[si] > 0.05:
                ax.text(i, bottom + mean_frac[si] / 2, f"{mean_frac[si] * 100:.0f}%",
                        ha="center", va="center", fontsize=12, color="white",
                        path_effects=[pe.Stroke(linewidth=2.5, foreground="black"),
                                      pe.Normal()])
            bottom += mean_frac[si]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["串行", "并发"])
    ax.set_ylabel("耗时占比")
    ax.set_ylim(0, 1)
    handles = [Patch(facecolor=ROW_COLORS[i], edgecolor="black", hatch=HATCHES[i],
                     label=ROWS[i]) for i in range(len(ROWS))]
    ax.legend(handles=handles, frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.set_title("各阶段耗时占比（修正口径）", fontsize=14)
    fig.text(0.5, 0.01,
             "口径：检索=四路并行净耗时(max)；规划路径题（plan 为 parallel_subq/multi_hop）检索耗时计入「规划执行+检索」；llm首token 已计入 LLM生成，不单列",
             ha="center", fontsize=9, color=GRAY)
    save(fig, "fig1_stage_stack.svg")


# ── 图2 配对总耗时：GroupedViolin 风格（小提琴+抖动散点+显著性括号）──
def fig2():
    recs = load_records()
    ser = [r["总计"] for r in recs["串行"]]
    con = [r["总计"] for r in recs["并发"]]
    t_stat, p = stats.ttest_rel(ser, con)
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.2g}"

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [ser, con]
    positions = [1, 2]
    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False, widths=0.85)
    body_colors = [BLUE, PINK]
    y_tops = []
    bodies = cast(list, parts["bodies"])
    for i, body in enumerate(bodies):
        body.set_facecolor(body_colors[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(1.0)
        verts = body.get_paths()[0].vertices
        ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
        y_tops.append(ymax)
        ax.plot([positions[i], positions[i]], [ymin, ymax],
                color="black", lw=0.8, alpha=0.8, zorder=2)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        jitter = rng.normal(0, 0.06, len(vals))
        ax.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                   s=10, color=body_colors[i], alpha=0.6, zorder=3,
                   edgecolors="none")

    lo = min(min(ser), min(con))
    hi = max(max(ser), max(con))
    span = hi - lo
    yb = max(y_tops) + 0.05 * span
    ax.plot([1, 2], [yb, yb], lw=1.0, c="black", zorder=4)
    ax.plot([1, 1], [yb, yb - 0.012 * span], lw=1.0, c="black", zorder=4)
    ax.plot([2, 2], [yb, yb - 0.012 * span], lw=1.0, c="black", zorder=4)
    ax.text(1.5, yb + 0.01 * span, f"配对 t 检验，{p_text}",
            ha="center", va="bottom", fontsize=11)

    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(lo - 0.03 * span, yb + 0.06 * span)
    ax.set_xticks(positions)
    ax.set_xticklabels(["串行", "并发"])
    ax.set_ylabel("总耗时 (ms)")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("每题总耗时配对比较", fontsize=14)
    save(fig, "fig2_paired_box.svg")


# ── 图5 QPS ──
def fig3():
    rows = json.loads((DATA / "throughput_result.json").read_text(encoding="utf-8"))
    x = [r["并发"] for r in rows]
    qps = [r["QPS"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, np.zeros_like(qps), qps, color=BLUE, alpha=0.5, label="QPS")
    ax.plot(x, qps, lw=3, c=TEAL, marker="o", markersize=8,
            markerfacecolor=TEAL, markeredgecolor="black", markeredgewidth=0.8)
    for xi, yi in zip(x, qps):
        ax.annotate(f"{yi:.2f}",
                    xy=(xi, yi), xytext=(xi, yi + 0.55),
                    ha="center", va="bottom", fontsize=11,
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black",
                                    shrinkA=0, shrinkB=0, mutation_scale=12))
    ax.legend(frameon=False, loc="upper left")
    ax.set_xticks(x)
    ax.set_xlabel("并发数")
    ax.set_ylabel("QPS")
    ax.set_ylim(0, max(qps) * 1.3)
    ax.set_title("吞吐量随并发变化", fontsize=14)
    save(fig, "fig3_qps.svg")


# ── 图6 延迟分位 ──
def fig4():
    rows = json.loads((DATA / "throughput_result.json").read_text(encoding="utf-8"))
    x = [r["并发"] for r in rows]
    p50 = [r["P50(ms)"] for r in rows]
    p95 = [r["P95(ms)"] for r in rows]
    p99 = [r["P99(ms)"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, p50, p99, color=BLUE, alpha=0.4, label="P50–P99 区间")
    ax.plot(x, p50, lw=3, c=GRAY, marker="o", markersize=7,
            markerfacecolor=GRAY, markeredgecolor="black", markeredgewidth=0.8,
            label="P50")
    ax.plot(x, p95, lw=3, c=TEAL, marker="s", markersize=7,
            markerfacecolor=TEAL, markeredgecolor="black", markeredgewidth=0.8,
            label="P95")
    ax.plot(x, p99, lw=3, c=PURPLE, marker="^", markersize=8,
            markerfacecolor=PURPLE, markeredgecolor="black", markeredgewidth=0.8,
            label="P99")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xticks(x)
    ax.set_xlabel("并发数")
    ax.set_ylabel("延迟 (ms)")
    ax.set_title("延迟分位随并发变化", fontsize=14)
    save(fig, "fig4_latency_percentile.svg")


# ── 图9 CDF 延迟曲线 ──
def fig5():
    recs = load_records()
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, line_c, area_c in [("串行", TEAL, BLUE), ("并发", PURPLE, PINK)]:
        vals = sorted(r["总计"] for r in recs[key])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.fill_between(vals, 0, y, color=area_c, alpha=0.35, step="post")
        ax.plot(vals, y, drawstyle="steps-post", lw=2.5, c=line_c,
                label=f"{key} (n={len(vals)})")
    ser = sorted(r["总计"] for r in recs["串行"])
    for p, lab, ypos in [(50, "P50", 0.72), (95, "P95", 0.84), (99, "P99", 0.96)]:
        v = np.percentile(ser, p)
        ax.axvline(v, ls="--", lw=1.0, color=GRAY, alpha=0.9)
        ax.text(v, ypos, f"{lab} {v / 1000:.1f}s", rotation=90, va="top", ha="right",
                fontsize=9, color=BLACK)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel("端到端总耗时 (ms)")
    ax.set_ylabel("累计比例")
    ax.set_ylim(0, 1)
    ax.set_title("端到端延迟 CDF（串行 vs 并发，分位线取串行）", fontsize=14)
    save(fig, "fig5_cdf.svg")


# ── 图10 尾延迟归因瀑布 ──
def fig6():
    recs = load_records()
    ser = sorted(recs["串行"], key=lambda r: r["总计"], reverse=True)[:10]
    corr = [corrected_stages(r) for r in ser]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    x = np.arange(len(ser))
    bottom = np.zeros(len(ser))
    for si, s in enumerate(ROWS):
        vals = np.array([c[s] for c in corr])
        ax.bar(x, vals, 0.6, bottom=bottom, color=ROW_COLORS[si],
               hatch=HATCHES[si], edgecolor="black", linewidth=1.2, alpha=0.9)
        bottom += vals
    for xi, r in enumerate(ser):
        ax.text(xi, r["总计"] + 120, f"{r['总计'] / 1000:.1f}s",
                ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r["id"]) for r in ser], rotation=20, ha="right",
                       fontsize=10)
    handles = [Patch(facecolor=ROW_COLORS[i], edgecolor="black", hatch=HATCHES[i],
                     label=ROWS[i]) for i in range(len(ROWS))]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("尾延迟归因：串行最慢 10 题", fontsize=14)
    save(fig, "fig6_tail_waterfall.svg")


# ── 图11 rerank 重排幅度 ──
def bucket_of(sources):
    if "sql" in sources:
        return "sql"
    if "metadata" in sources:
        return "metadata"
    return "corpus"


def question_metrics(r):
    merged = r["trace"].get("rrf_merged") or []
    final = r["trace"].get("reranked") or []
    rank_of = {c["chunk_id"]: i + 1 for i, c in enumerate(merged)}
    scored = []
    for pos, c in enumerate(final, 1):
        if c.get("rerank_score") is None:
            continue
        if bucket_of(c.get("sources") or []) == "sql":
            continue
        if c["chunk_id"] not in rank_of:
            continue
        scored.append({
            "chunk_id": c["chunk_id"],
            "rrf_rank": rank_of.get(c["chunk_id"]),
            "final_rank": pos,
            "rerank_score": c["rerank_score"],
            "bucket": bucket_of(c.get("sources") or []),
        })
    if not scored:
        return None
    final_ids = {c["chunk_id"] for c in final}
    dropped = [
        i + 1 for i, c in enumerate(merged)
        if c["chunk_id"] not in final_ids and bucket_of(c.get("sources") or []) != "sql"
    ]
    return {
        "scored": scored,
        "rescue": max((s["rrf_rank"] for s in scored if s["final_rank"] <= 3), default=0),
        "veto": min(dropped, default=0),
        "disp": sum(abs(s["rrf_rank"] - s["final_rank"]) for s in scored),
        "n": len(scored),
    }


def fig7():
    recs = load_trace()
    metrics = [m for m in (question_metrics(r) for r in recs) if m]

    rescue_q = max(metrics, key=lambda m: m["rescue"])
    veto_q = max(
        (m for m in metrics if m["veto"] > 0),
        key=lambda m: (m["veto"] == 0, -m["veto"]), default=None)
    stable_q = min(metrics, key=lambda m: m["disp"])
    fallback_q = min((m for m in metrics if m is not stable_q),
                     key=lambda m: m["disp"], default=stable_q)

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 0.9])
    ax_main = fig.add_subplot(gs[:, 0])

    xs, ys, cols, filled = [], [], [], []
    for m in metrics:
        for s in m["scored"]:
            xs.append(s["rrf_rank"])
            ys.append(s["rerank_score"])
            cols.append(BLUE if s["bucket"] == "metadata" else PINK)
            filled.append(s["final_rank"] <= 10)
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    filled = np.array(filled)
    for f in (True, False):
        mask = filled == f
        ax_main.scatter(
            xs[mask], ys[mask], s=30,
            facecolors=[c for c, m_ in zip(cols, mask) if m_] if f else "none",
            edgecolors=[c for c, m_ in zip(cols, mask) if m_],
            linewidth=1.0, alpha=0.75, zorder=3,
            label="入选 top-10" if f else "落选（补位）")
    r_stat, _ = stats.spearmanr(xs, ys)
    ax_main.text(0.03, 0.97, f"Spearman r = {r_stat:.2f}",
                 transform=ax_main.transAxes, fontsize=12, va="top")
    ax_main.set_xlabel("RRF 序位（rrf_merged 中名次）")
    ax_main.set_ylabel("rerank_score")
    ax_main.set_ylim(0, 1.05)
    ax_main.set_title("Rerank 重排幅度：全题池候选（x=RRF 序位）", fontsize=14)
    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor=BLUE,
               markeredgecolor="black", markersize=8, label="metadata"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=PINK,
               markeredgecolor="black", markersize=8, label="corpus"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="white",
               markeredgecolor=BLACK, markersize=8, label="落选（补位）"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=BLACK,
               markeredgecolor="black", markersize=8, label="入选 top-10"),
    ]
    ax_main.legend(handles=handles, frameon=False, loc="center right", fontsize=10)

    panels = [
        (gs[0, 1], rescue_q, "救回最显著"),
        (gs[1, 1], veto_q if veto_q else fallback_q,
         "否决最显著" if veto_q else "次小位移"),
        (gs[2, 1], stable_q, "最小位移"),
    ]
    for cell, m, tag in panels:
        ax = fig.add_subplot(cell)
        for s in m["scored"]:
            in_final = s["final_rank"] <= 10
            color = (BLUE if s["bucket"] == "metadata" else PINK) if in_final else GRAY
            ax.plot([0, 1], [s["rrf_rank"], s["final_rank"]],
                    color=color, lw=1.3, ls="-" if in_final else "--",
                    alpha=0.85, zorder=2)
            ax.scatter([0], [s["rrf_rank"]], s=26,
                       facecolors=color if in_final else "white",
                       edgecolors=color, linewidth=1.0, zorder=3)
            ax.scatter([1], [s["final_rank"]], s=26,
                       facecolors=color if in_final else "white",
                       edgecolors=color, linewidth=1.0, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RRF 序", "rerank 序"])
        ax.invert_yaxis()
        ax.set_ylabel("名次")
        ax.set_title(tag, fontsize=11)
        ax.set_ylim(bottom=max(s["rrf_rank"] for s in m["scored"]) + 0.5, top=0.5)

    fig.text(0.5, 0.01,
             "主图一点=一候选（sql 直通不参与 rerank 打分，未入图）；侧图虚线=落选补位，实线=入选 top-10",
             ha="center", fontsize=9, color=GRAY)
    save(fig, "fig7_rerank.svg")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("7 幅图完成")
