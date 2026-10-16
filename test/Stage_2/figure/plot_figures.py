# -*- coding: utf-8 -*-
"""Stage_2 检索质量评测可视化 —— 顶刊风格 + 品牌五色版，6 幅图。

设计基线（取自本仓库 academic-figure-skill/references/）
--------------------------------------------------------
typography.md       Arial 主体 + 中文字形回退；轴线 0.6 pt；仅保留左/下框线；刻度外向
journal-specs.md    单栏 89 mm / 双栏 183 mm；尺寸在创建时设定，导出后不再缩放
common-pitfalls.md  图例移出绘图区；柱状图叠加原始点；避免过度装饰

配色：用户品牌五色（跨阶段同色同义）
--------------------------------------------------------
TEAL  #018C94  混合检索（最强档）/ 主曲线 / 命中
BLUE  #89B2F4  向量单路 / 回落面 / 边际分布
GRAY  #7F7F7F  BM25 单路 / 无变化 / 未召回
PURP  #764D9C  Rerank / 提升 / 唯一强调色
PINK  #EBB9BC  增益区间 / 正区间填充
三档检索策略按「词法 -> 稠密 -> 混合」能力顺序取 灰 -> 浅蓝 -> 青绿，构成有序色阶。

字体回退注意
------------
matplotlib 仅在 ``font.family`` 收到「具体字体名列表」时才做逐字形回退；
故此处固定使用显式列表。

输出
----
figure/fig*.svg  矢量主文件（仅 SVG，文字保持可编辑）
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")            # 无显示环境（服务器 / CI）同样可出图
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE.parent
STAGE1 = DATA.parent / "Stage_1"        # 跨阶段数据：延迟 timings + trace

MM = 1.0 / 25.4
W_SINGLE = 89 * MM               # 单栏
W_DOUBLE = 183 * MM              # 双栏

# ── 品牌五色 + 中性色 ──
TEAL = "#018C94"                 # 混合 / 主曲线 / 命中
BLUE = "#89B2F4"                 # 向量 / 回落面
BLUE_L = "#BBD3F9"
BLUE_XL = "#E3EDFD"
PURP = "#764D9C"                 # Rerank / 提升 / 强调
PINK = "#EBB9BC"                 # 增益面
PINK_XL = "#F7E4E5"
GRAY = "#7F7F7F"                 # BM25 / 无变化 / 未召回
GRAY_L = "#C9C9CE"
GRAY_XL = "#EFEFF1"

INK = "#1A1A1A"
INK_SOFT = "#6B6B70"
INK_FAINT = "#9A9AA0"
SPINE = "#4A4A4F"
GRID = "#E7E7EA"
PAPER = "#FFFFFF"

STRAT_COLOR = {"BM25": GRAY, "向量": BLUE, "混合": TEAL}
STRAT_FALLBACK = [TEAL, PURP, BLUE, GRAY, "#762A83", "#F1A340"]

mpl.rcParams.update({
    "font.family": ["Arial", "Microsoft YaHei"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.titlelocation": "left",
    "axes.titlepad": 5.0,
    "axes.labelcolor": INK,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handlelength": 1.7,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.5,
    "legend.borderpad": 0.2,
    "lines.solid_capstyle": "round",
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "svg.fonttype": "none",      # 文字保持可编辑；若期刊要求转曲改为 "path"
    "pdf.fonttype": 42,
})


# ────────────────────────── 通用工具 ──────────────────────────
def load(name):
    """读取 Stage_2 目录下的评测结果 JSON。"""
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_stage1(name):
    """读取 Stage_1 目录下的 JSON（跨阶段联合图数据源）。"""
    return json.loads((STAGE1 / name).read_text(encoding="utf-8"))


def new_fig(w_mm, h_mm):
    """按期刊栏宽（毫米）创建画布——尺寸在创建时定死，导出后不再缩放。"""
    return plt.subplots(figsize=(w_mm * MM, h_mm * MM))


def tidy(ax, grid_axis=None):
    """统一坐标轴：仅左/下框线 0.6 pt，刻度外向，可选极淡网格。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(SPINE)
    ax.tick_params(length=2.4, width=0.6, color=SPINE, pad=2.4)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    return ax


def strat_colors(names):
    """策略名 -> 颜色；未知策略按固定次序回退，保证跨图同色同义。"""
    return [STRAT_COLOR.get(n, STRAT_FALLBACK[i % len(STRAT_FALLBACK)])
            for i, n in enumerate(names)]


def save(fig, name):
    """仅导出 SVG 矢量主文件。"""
    fig.savefig(BASE / name, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {name}")


# ────────────────────── 图1 分层召回率 Recall@K ──────────────────────
def fig1():
    """RRF 融合 vs 精排 Rerank 的 Recall@K；标注 K=1 回落与 K=3 增益。"""
    d = load("recall_at_k.json")
    ks = np.arange(1, 21)
    rrf = np.array([d["rrf"][str(k)] for k in ks])
    rer = np.array([d["rerank"][str(k)] for k in ks])
    gain3 = (rer[2] - rrf[2]) * 100
    loss1 = (rer[0] - rrf[0]) * 100

    fig, ax = new_fig(126, 86)
    tidy(ax, grid_axis="y")

    ax.fill_between(ks, rrf, rer, where=rer >= rrf, interpolate=True,
                    color=PINK_XL, lw=0, zorder=1)
    ax.fill_between(ks, rrf, rer, where=rer < rrf, interpolate=True,
                    color=BLUE_XL, lw=0, zorder=1)
    ax.plot(ks, rrf, ls=(0, (3.6, 1.7)), lw=1.4, color=TEAL, marker="s",
            ms=2.9, mfc=TEAL, mec=PAPER, mew=0.5, zorder=3)
    ax.plot(ks, rer, ls="-", lw=1.8, color=PURP, marker="o", ms=3.3,
            mfc=PURP, mec=PAPER, mew=0.5, zorder=4)

    ax.annotate(f"K=3   +{gain3:.1f} pp", xy=(3, rer[2]), xytext=(4.3, 0.968),
                fontsize=7.2, color=PURP, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PURP, shrinkA=1,
                                shrinkB=3, connectionstyle="arc3,rad=-0.20"))
    ax.annotate(f"K=1   {loss1:.1f} pp", xy=(1, (rrf[0] + rer[0]) / 2),
                xytext=(1.85, 0.640), fontsize=7.2, color=TEAL,
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=TEAL, shrinkA=1,
                                shrinkB=3, connectionstyle="arc3,rad=0.16"))

    ax.set_xlim(0.4, 20.6)
    ax.set_ylim(0.60, 1.02)
    ax.set_xticks([1, 3, 5, 10, 15, 20])
    ax.set_yticks([0.60, 0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("Top-K")
    ax.set_ylabel("文档级召回率 Recall@K")
    ax.set_title("分层召回率 Recall@K")

    handles = [
        Line2D([], [], color=PURP, lw=1.8, marker="o", ms=3.3, mec=PAPER,
               mew=0.5, label="精排 Rerank"),
        Line2D([], [], color=TEAL, lw=1.4, ls=(0, (3.6, 1.7)), marker="s",
               ms=2.9, mec=PAPER, mew=0.5, label="RRF 融合"),
        Patch(facecolor=PINK_XL, edgecolor="none", label="Rerank 增益区间"),
        Patch(facecolor=BLUE_XL, edgecolor="none", label="回落区间"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=4, handlelength=1.9, columnspacing=1.3)
    save(fig, "fig1_recall_k.svg")


# ─────────────── 图2 排名分布（阶梯矩阵 + 累计命中） ───────────────
def fig2():
    """正确文档在重排后候选列表中的排名分布：左侧按排名排序的矩阵，
    右侧累计命中占比曲线。同一批正确文档贯穿两栏，故共用 y 轴。"""
    rows = load("rank_distribution.json")
    ranks = [r["rank"] for r in rows]
    n_doc = len(rows)
    max_rank = 10
    n_row = max_rank + 1                      # 排名 1..10 + 未进入候选

    counts = np.zeros(n_row, dtype=int)
    for r in ranks:
        counts[max_rank if r is None else min(r, max_rank) - 1] += 1
    miss_row = max_rank                       # 最后一行 = 未召回

    # 按排名升序排序 -> 阶梯状，分布形状一眼可读
    order = sorted(range(n_doc),
                   key=lambda i: (max_rank + 1) if ranks[i] is None else ranks[i])
    img = np.ones((n_row, n_doc, 3))
    for pos, i in enumerate(order):
        r = ranks[i]
        img[miss_row if r is None else r - 1, pos] = to_rgb(
            GRAY if r is None else TEAL)

    fig, (ax, ax_c) = plt.subplots(
        1, 2, figsize=(W_DOUBLE * 1.0, 86 * MM), sharey=True,
        gridspec_kw={"width_ratios": [2.05, 1.0]})

    ax.imshow(img, aspect="auto", interpolation="nearest")
    for y in np.arange(0.5, n_row - 0.5, 1):
        ax.axhline(y, color=PAPER, lw=0.9, zorder=3)
    for x in np.arange(4.5, n_doc, 5):
        ax.axvline(x, color=PAPER, lw=0.5, zorder=3)

    step = max(1, n_doc // 12)
    ax.set_xticks(np.arange(0, n_doc, step))
    ax.set_xticklabels(np.arange(1, n_doc + 1, step))
    ax.set_yticks(range(n_row))
    ax.set_yticklabels([str(i) for i in range(1, max_rank + 1)] + ["未召回"])
    ax.set_xlabel("正确文档序号（按重排后排名排序）")
    ax.set_ylabel("重排后排名")
    ax.set_title("a  排名分布（n = %d 篇正确文档）" % n_doc)
    ax.tick_params(length=0, pad=3)

    cum = np.cumsum(counts) / n_doc * 100
    ypos = np.arange(n_row)
    ax_c.axvline(100, color=GRAY_L, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax_c.plot(cum, ypos, "-", color=TEAL, lw=1.6, zorder=3)
    ax_c.plot(cum[:max_rank], ypos[:max_rank], "o", ms=3.0, mfc=TEAL,
              mec=PAPER, mew=0.5, zorder=4)
    ax_c.plot(cum[miss_row], ypos[miss_row], "o", ms=3.0, mfc=GRAY,
              mec=PAPER, mew=0.5, zorder=4)

    for row in range(n_row):
        if row == miss_row or row == 0 or counts[row] > 0:
            val = cum[row]
            ax_c.text(val + 1.8, row, f"{val:.1f}%", va="center", ha="left",
                      fontsize=6.8,
                      color=GRAY if row == miss_row else INK_SOFT)

    ax_c.set_xlim(0, 124)
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.set_xlabel("累计命中占比（%）")
    ax_c.set_title("b  累计命中")
    ax_c.text(0.035, 0.055, "未召回 %d 篇（%.1f%%）"
              % (counts[miss_row], counts[miss_row] / n_doc * 100),
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=7,
              color=INK_FAINT)
    save(fig, "fig2_rank_distribution.svg")


# ────────────────────── 图3 PR 曲线 ──────────────────────
def fig3():
    """精排分数对「相关 / 不相关」的区分度：PR 曲线 + 无技能基线
    （pi = 正类先验），并标出最佳 F1 工作点。"""
    d = load("pr_curve.json")
    pts = sorted(d["points"], key=lambda p: p["recall"])
    rec = np.array([p["recall"] for p in pts])
    pre = np.array([p["precision"] for p in pts])
    thr = np.array([p["threshold"] for p in pts])
    prev = d["n_pos"] / (d["n_pos"] + d["n_neg"])       # 无技能基线
    f1 = 2 * pre * rec / (pre + rec)
    k = int(np.argmax(f1))
    y0 = 0.36

    fig, ax = new_fig(92, 92)
    tidy(ax, grid_axis="both")

    ax.axhline(prev, color=GRAY, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(0.613, prev + 0.012, f"无技能基线  π = {prev:.3f}",
            fontsize=6.8, color=INK_SOFT, ha="left", va="bottom")

    ax.plot(rec, pre, "-", color=TEAL, lw=1.7, zorder=4)
    ax.plot(rec, pre, "o", ms=3.0, mfc=TEAL, mec=PAPER, mew=0.5, zorder=5)
    ax.plot(rec[k], pre[k], "o", ms=7.5, mfc=PAPER, mec=PURP, mew=1.3,
            zorder=6)

    ax.annotate(f"最佳 F1 = {f1[k]:.3f}\n（阈值 {thr[k]:.2f}）",
                xy=(rec[k], pre[k]), xytext=(0.615, 0.998), fontsize=7,
                color=PURP, ha="left", va="top", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PURP, shrinkA=4,
                                shrinkB=5, connectionstyle="arc3,rad=-0.18"))
    ax.text(0.998, 0.372, f"AUC-PR = {d['auc_pr']:.3f}",
            ha="right", va="bottom", fontsize=6.8, color=INK_SOFT)

    ax.set_xlim(0.60, 1.01)
    ax.set_ylim(y0, 1.01)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("精排分数的区分度：Precision-Recall 曲线")
    save(fig, "fig3_pr_curve.svg")


# ────────── 图4 噪声鲁棒性（逐题结果 + 性能衰减） ──────────
def fig4():
    """5 类查询扰动下的鲁棒性：左栏逐题命中/未命中，右栏 Recall@5 相对
    标准查询的衰减。逐题明细让「9/10」这类比例有据可查。"""
    d = load("noise_decay.json")
    conds = [c for c in ["标准", "错别字", "口语化", "截断", "指代"]
             if c in d["summary"]]
    rows = d["rows"]
    n_q = len(rows)
    vals = np.array([d["summary"][c] for c in conds])
    base = vals[0]
    miss = {c: sum(1 for r in rows if not r.get(c)) for c in conds}

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 74 * MM),
        gridspec_kw={"width_ratios": [1.12, 1.0]})

    tidy(ax_a)
    for i in range(n_q):
        for j in range(len(conds)):
            ax_a.plot([j - 0.5, j + 0.5], [i, i], color=GRID, lw=0.5, zorder=0)
    for i, r in enumerate(rows):
        for j, c in enumerate(conds):
            if r.get(c):
                ax_a.scatter(j, i, s=30, facecolor=TEAL, edgecolor="none",
                             zorder=3)
            else:
                ax_a.scatter(j, i, s=42, marker="x", color=PURP, lw=1.4,
                             zorder=4)
    ax_a.set_xticks(range(len(conds)))
    ax_a.set_xticklabels(conds)
    ax_a.set_yticks(range(n_q))
    ax_a.set_yticklabels([f"Q{i + 1}" for i in range(n_q)])
    ax_a.set_xlim(-0.6, len(conds) - 0.4)
    ax_a.set_ylim(n_q - 0.5, -0.5)
    ax_a.tick_params(length=0, pad=3)
    ax_a.set_xlabel("查询扰动类型")
    ax_a.set_title("a  逐题命中情况（n = %d 题）" % n_q)
    ax_a.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4.2, mfc=TEAL, mec="none",
               label="命中"),
        Line2D([], [], marker="x", ls="", ms=4.6, mec=PURP, mew=1.4,
               label="未命中"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2,
        handletextpad=0.3, columnspacing=1.4)

    tidy(ax_b, grid_axis="x")
    delta = vals - base
    y = np.arange(len(conds))[::-1]
    for yi, dl in zip(y, delta):
        if abs(dl) < 1e-9:
            ax_b.plot([0], [yi], marker="|", ms=9, color=GRAY, mew=1.2,
                      zorder=3)
        else:
            ax_b.barh(yi, dl, height=0.5, color=PINK_XL, edgecolor=PURP,
                      lw=0.9, zorder=3)
            ax_b.text(dl - 0.005, yi, f"{dl:+.3f}", ha="right", va="center",
                      fontsize=7.4, color=PURP)
    for yi, c in zip(y, conds):
        ok = n_q - miss[c]
        ax_b.text(0.006, yi, f"{ok}/{n_q} 题命中", fontsize=6.6,
                  color=PURP if miss[c] else INK_FAINT, ha="left",
                  va="center")
    ax_b.axvline(0, color=SPINE, lw=0.7, zorder=2)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(conds)
    ax_b.set_xlim(-0.135, 0.058)
    ax_b.set_xticks([-0.10, -0.05, 0])
    ax_b.set_xlabel("ΔRecall@5（相对标准查询）")
    ax_b.set_title("b  性能衰减")
    save(fig, "fig4_noise_decay.svg")


# ────────── 图5 MRR + 95% Bootstrap CI（GroupedBarChart 风格） ──────────
def fig5(n_boot=2000, seed=42):
    """柱 + 误差棒 + 原始点散布：各策略 MRR 点估计与不确定度。"""
    d = load("mrr.json")
    names = d["strategies"]
    cols = strat_colors(names)
    rng = np.random.default_rng(seed)

    fig, ax = new_fig(104, 84)
    tidy(ax, grid_axis="y")

    for i, s in enumerate(names):
        v = np.array([r[s] for r in d["rows"]])
        boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        m = v.mean()

        ax.bar(i, m, 0.56, color=cols[i], alpha=0.55, edgecolor=cols[i],
               lw=0.8, zorder=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK,
                    elinewidth=1.1, capsize=3.0, capthick=1.1, zorder=5)
        jx = rng.normal(i, 0.09, len(v))
        ax.scatter(jx, v, s=16, color=cols[i], alpha=0.65, edgecolor=INK,
                   linewidth=0.4, zorder=4)
        ax.text(i, hi + 0.06, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.62, len(names) - 0.38)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("MRR")
    ax.set_title("各策略 MRR 与 95% Bootstrap 置信区间")
    save(fig, "fig5_mrr_ci.svg")


# ────────── 图6 成本-召回曲线（上下文字符 vs RR） ──────────
def fig6():
    """生产链路的上下文成本与召回质量：每题一点（x=送入 LLM 的上下文
    总字符数，y=混合策略 RR），分箱均值曲线展示成本换召回的趋势。"""
    trace = load_stage1("serial_trace.json")
    mrr = load("mrr.json")
    rr_by_id = {r["id"]: r["混合"] for r in mrr["rows"]}
    pts = [(t["trace"].get("context_chars", 0), rr_by_id[t["id"]])
           for t in trace
           if t["id"] in rr_by_id and t["trace"].get("n_context", 0) > 0]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # 分箱均值曲线（按字符数分 5 箱）
    order = np.argsort(xs)
    xs_s, ys_s = xs[order], ys[order]
    edges = np.array_split(np.arange(len(xs_s)), 5)
    bx = [float(xs_s[e].mean()) for e in edges]
    by = [float(ys_s[e].mean()) for e in edges]
    bn = [len(e) for e in edges]

    fig, ax = new_fig(126, 82)
    tidy(ax, grid_axis="both")

    rng = np.random.default_rng(7)
    ax.scatter(xs, ys + rng.uniform(-0.03, 0.03, len(ys)), s=14,
               facecolor=TEAL, edgecolor=PAPER, lw=0.3, alpha=0.55,
               zorder=3, label="单题（混合策略）")
    ax.plot(bx, by, "-", color=PURP, lw=1.8, zorder=5,
            label="分箱均值曲线")
    ax.plot(bx, by, "o", ms=3.6, mfc=PURP, mec=PAPER, mew=0.5, zorder=6)
    for xb, yb, n in zip(bx, by, bn):
        ax.text(xb, yb + 0.05, f"n={n}", ha="center", va="bottom",
                fontsize=6.4, color=INK_FAINT)

    ax.set_xlabel("送入 LLM 的上下文总字符数")
    ax.set_ylabel("RR（混合策略）")
    ax.set_ylim(-0.06, 1.10)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_title("成本-召回：上下文规模与命中质量")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    save(fig, "fig6_cost_recall.svg")




if __name__ == "__main__":
    print("Stage_2 顶刊风格 + 品牌五色可视化：")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print(f"完成，输出目录：{BASE}")
