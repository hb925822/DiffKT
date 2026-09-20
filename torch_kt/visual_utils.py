#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/20 0020 下午 6:46
# @Author  : hb
# @File    : visual_utils.py
# ml_plots.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Dict, Any
# ml_plot_style.py
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any

NATURE_COLORS = [
    "#E64B35",  # Red
    "#4DBBD5",  # Blue
    "#00A087",  # Green
    "#3C5488",  # Dark Blue
    "#F39B7F",  # Light Orange
    "#8491B4",  # Purple Gray
    "#91D1C2",  # Teal
    "#DC0000",  # Strong Red (for emphasis)
]

COLORBREWER_SET1 = [
    "#E41A1C",  # Red
    "#377EB8",  # Blue
    "#4DAF4A",  # Green
    "#984EA3",  # Purple
    "#FF7F00",  # Orange
    "#FFFF33",  # Yellow (慎用，打印易丢失)
    "#A65628",  # Brown
    "#F781BF",  # Pink
]

PAUL_TOL_BRIGHT = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Magenta
]

PAUL_TOL_VIBRANT = [
    "#0077BB",  # Blue
    "#33BBEE",  # Cyan
    "#009988",  # Teal
    "#EE7733",  # Orange
    "#CC3311",  # Red
    "#EE3377",  # Magenta
]
# ----------------------------
# SCI 顶刊标准 rcParams 配置
# ----------------------------
BASE_RC = {
    # ──────────────── 字体 ────────────────
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif'],
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.monospace': ['Courier New', 'DejaVu Sans Mono'],
    'text.usetex': False,  # 默认关闭，函数中可动态开启
    'mathtext.fontset': 'stix',  # STIX ≈ Times 风格数学符号
    'axes.formatter.use_mathtext': True,

    # ──────────────── 字号（单位：pt）────────────────
    # Science/Nature 要求图内文字 ≤ 正文（通常 8–10 pt）
    'font.size': 9,  # 全局默认（用于 tick labels）
    'axes.titlesize': 10,  # 标题（建议慎用，多数期刊不要标题）
    'axes.labelsize': 9,  # x/y 轴标签
    'xtick.labelsize': 8,  # x 轴刻度文字
    'ytick.labelsize': 8,  # y 轴刻度文字
    'legend.fontsize': 8,  # 图例
    'figure.titlesize': 10,  # figure.suptitle

    # ──────────────── 线条与标记 ────────────────
    'lines.linewidth': 0.8,  # Science: 0.5–1.0 pt；0.8 pt ≈ 0.28 mm（推荐）
    'lines.markersize': 3,  # 小而清晰
    'lines.markeredgewidth': 0.4,
    'errorbar.capsize': 2,  # 误差线帽子长度（pt）

    # ──────────────── 坐标轴 ────────────────
    'axes.linewidth': 0.6,  # 轴线宽度（pt）
    'axes.labelpad': 3,  # 轴标签与轴的距离（pt）

    # 刻度
    'xtick.major.width': 0.6,  # 主刻度线宽
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,  # 次刻度线宽
    'ytick.minor.width': 0.4,

    'xtick.major.size': 3,  # 主刻度长度（pt）
    'ytick.major.size': 3,
    'xtick.minor.size': 2,  # 次刻度长度
    'ytick.minor.size': 2,

    'xtick.direction': 'in',  # 刻度朝内（Science/Nature 标准）
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.top': True,  # 上方也显示刻度
    'ytick.right': True,

    # ──────────────── 图例 ────────────────
    'legend.frameon': False,  # 无边框（Nature/Science 要求）
    'legend.loc': 'best',
    'legend.columnspacing': 0.8,
    'legend.handletextpad': 0.3,
    'legend.handlelength': 1.0,

    # ──────────────── 颜色循环（Nature 推荐 + ColorBrewer Qualitative）────────────────
    # 来源：Nature Methods "Points of View: Color" (2011), ColorBrewer Set1/Set2
    'axes.prop_cycle': mpl.cycler(color=[
        "#E64B35",  # Red (Nature red)
        "#4DBBD5",  # Blue (Nature blue)
        "#00A087",  # Green (Nature green)
        "#3C5488",  # Dark Blue
        "#F39B7F",  # Light Orange
        "#8491B4",  # Purple Gray
        "#91D1C2",  # Teal
        "#DC0000",  # Strong Red (for emphasis)
        "#7E61AF",  # Violet
    ]),

    # ──────────────── 图形尺寸（英寸）────────────────
    # Science 单栏宽度：～8.7 cm = 3.425 in
    # Nature 单栏宽度：86 mm = 3.386 in
    # IEEE 单栏：3.5 in
    'figure.figsize': (3.4, 2.3),  # 默认：单栏，高宽比 ≈ 0.68（适合曲线图）
    # 双栏示例：(7.0, 2.5) 或 (7.0, 4.0)

    # ──────────────── 输出设置 ────────────────
    'savefig.dpi': 600,  # Science 要求 ≥ 300 DPI（线图建议 600）
    'savefig.bbox': 'tight',  # 自动裁剪空白
    'savefig.pad_inches': 0.02,  # 裁剪后保留微小边距
    'savefig.transparent': False,  # 通常白底
    'pdf.fonttype': 42,  # PDF 嵌入 TrueType 字体（避免 Type 3）
    'ps.fonttype': 42,

    # ──────────────── 其他 ────────────────
    'axes.spines.top': True,  # 保留上边框（配合 inward ticks）
    'axes.spines.right': True,
    'image.cmap': 'viridis',  # 默认 colormap（优于 jet）
}


def _apply_rc(rc_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并基础配置与用户覆盖项"""
    rc = BASE_RC.copy()
    if rc_overrides:
        rc.update(rc_overrides)
    return rc


# -----------------------------
# 1. 多折线图（通用版，支持训练曲线、算法对比等）
# -----------------------------
def plot_multi_line(
        ax: plt.Axes,
        x: List[float],
        y_list: List[List[float]],
        labels: Optional[List[str]] = None,
        colors: Union[str, List[str]] = 'auto',  # 'auto' 使用 rcParams 中的颜色循环
        markers: Union[str, List[str], None] = None,
        linestyles: Union[str, List[str]] = '-',
        title: str = '',
        xlabel: str = 'Epoch',
        ylabel: str = 'Loss',
        grid: bool = True,
        rc_overrides: Optional[Dict[str, Any]] = None,
        use_latex: bool = False,
) -> plt.Axes:
    """
    在给定 axes 上绘制多条折线。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标坐标轴。
    x : List[float]
        X 轴数据（共享）。
    y_list : List[List[float]]
        每条曲线的 Y 值列表。
    labels : List[str], optional
        每条曲线的标签（用于 legend）。
    colors : str or List[str] or 'auto'
        - 'auto': 使用当前 color cycle（推荐）
        - str: 所有曲线同色
        - List[str]: 每条曲线指定颜色
    markers : str or List[str] or None
        标记符号（如 'o', 's'），None 表示无标记。
    linestyles : str or List[str]
        线型（如 '-', '--', '-.'）。
    title, xlabel, ylabel : str
        图标题和轴标签。
    grid : bool
        是否显示网格。
    rc_overrides, use_latex : ...
        样式控制。

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        raise ValueError("ax must be provided.")

    n_lines = len(y_list)
    if n_lines == 0:
        return ax

    # 默认标签
    if labels is None:
        labels = [f"Line {i + 1}" for i in range(n_lines)]
    elif len(labels) != n_lines:
        raise ValueError(f"Length of labels ({len(labels)}) must match number of lines ({n_lines}).")

    # 处理 colors
    if colors == 'auto':
        # 不传 color，让 Matplotlib 自动使用 prop_cycle
        color_list = [None] * n_lines
    elif isinstance(colors, str):
        color_list = [colors] * n_lines
    else:
        if len(colors) != n_lines:
            raise ValueError(f"Length of colors ({len(colors)}) must match number of lines ({n_lines}).")
        color_list = colors

    # 处理 markers
    if markers is None:
        marker_list = [None] * n_lines
    elif isinstance(markers, str):
        marker_list = [markers] * n_lines
    else:
        if len(markers) != n_lines:
            raise ValueError(f"Length of markers ({len(markers)}) must match number of lines ({n_lines}).")
        marker_list = markers

    # 处理 linestyles
    if isinstance(linestyles, str):
        linestyle_list = [linestyles] * n_lines
    else:
        if len(linestyles) != n_lines:
            raise ValueError(f"Length of linestyles ({len(linestyles)}) must match number of lines ({n_lines}).")
        linestyle_list = linestyles

    rc = _apply_rc(rc_overrides)
    rc['text.usetex'] = use_latex

    with plt.rc_context(rc):
        for i in range(n_lines):
            ax.plot(
                x,
                y_list[i],
                label=labels[i],
                color=color_list[i],
                marker=marker_list[i],
                linestyle=linestyle_list[i],
                markersize=3
            )
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if grid:
            ax.grid(True, linestyle='--', alpha=0.5)

    return ax

# -----------------------------
# 1. 训练曲线（Loss / Accuracy）
# -----------------------------
def plot_training_curve(
        ax: plt.Axes,
        epochs: List[int],
        train_vals: List[float],
        val_vals: Optional[List[float]] = None,
        title: str = '',
        xlabel: str = 'Epoch',
        ylabel: str = 'Loss',
        label_train: str = 'Train',
        label_val: str = 'Validation',
        rc_overrides: Optional[Dict[str, Any]] = None,
        use_latex: bool = False
) -> plt.Axes:
    if ax is None:
        raise ValueError("ax must be provided.")

    rc = _apply_rc(rc_overrides)
    rc['text.usetex'] = use_latex

    with plt.rc_context(rc):
        ax.plot(epochs, train_vals, label=label_train, marker='o', markersize=3)
        if val_vals is not None:
            ax.plot(epochs, val_vals, label=label_val, marker='s', markersize=3)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True, linestyle='--', alpha=0.5)
    return ax


# -----------------------------
# 2. 多算法对比折线图
# -----------------------------
def plot_comparison_lines(
        ax: plt.Axes,
        x: List[float],
        y_dict: Dict[str, List[float]],
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        markers: Optional[List[str]] = None,
        rc_overrides: Optional[Dict[str, Any]] = None,
        use_latex: bool = False
) -> plt.Axes:
    if ax is None:
        raise ValueError("ax must be provided.")

    rc = _apply_rc(rc_overrides)
    rc['text.usetex'] = use_latex

    with plt.rc_context(rc):
        names = list(y_dict.keys())
        for i, name in enumerate(names):
            marker = markers[i] if markers else None
            ax.plot(x, y_dict[name], label=name, marker=marker)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True, linestyle='--', alpha=0.5)
    return ax


# -----------------------------
# 3. 消融实验柱状图（带误差线）
# -----------------------------
def plot_ablation_bar(
        ax: plt.Axes,
        labels: List[str],
        values: List[float],
        errors: Optional[List[float]] = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = 'Score',
        color: Union[str, List[str]] = '#4DBBD5',  # ← 支持单色或颜色列表
        rc_overrides: Optional[Dict[str, Any]] = None,
        use_latex: bool = False,
) -> plt.Axes:
    """
    绘制消融实验柱状图，支持单色或多色。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标坐标轴。
    labels : List[str]
        柱子的标签（如 ['w/o A', 'w/o B', 'Full']）。
    values : List[float]
        每个柱子的高度。
    errors : Optional[List[float]], optional
        误差线长度（对称），默认 None。
    title, xlabel, ylabel : str
        图标题和轴标签。
    color : str or List[str]
        - 若为 str：所有柱子使用该颜色（单色）
        - 若为 List[str]：每个柱子对应一个颜色（长度必须 == len(labels)）
    rc_overrides : dict, optional
        样式覆盖。
    use_latex : bool, optional
        是否启用 LaTeX。

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        raise ValueError("ax must be provided.")

    n_bars = len(labels)

    # 验证 color 输入
    if isinstance(color, list):
        if len(color) != n_bars:
            raise ValueError(f"Length of color list ({len(color)}) must match number of bars ({n_bars}).")
        colors = color
    else:
        # 单色：广播为列表
        colors = [color] * n_bars

    rc = _apply_rc(rc_overrides)
    rc['text.usetex'] = use_latex

    with plt.rc_context(rc):
        x_pos = np.arange(n_bars)
        ax.bar(
            x_pos,
            values,
            yerr=errors,
            capsize=3,
            color=colors,  # ← 支持多色
            edgecolor='black',
            linewidth=0.3
        )
        ax.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            xticks=x_pos,
            xticklabels=labels
        )
        ax.tick_params(axis='x', rotation=30)

    return ax


# -----------------------------
# 4. 混淆矩阵热力图
# -----------------------------
def plot_confusion_matrix(
        ax: plt.Axes,
        cm: np.ndarray,
        classes: List[str],
        title: str = 'Confusion Matrix',
        normalize: bool = False,
        cmap: Union[str, mpl.colors.Colormap] = 'Blues',
        add_colorbar: bool = True,
        cbar_label: Optional[str] = None,
        cbar_shrink: float = 0.8,
        cbar_pad: float = 0.05,
        rc_overrides: Optional[Dict[str, Any]] = None,
        use_latex: bool = False,
) -> Tuple[plt.Axes, Optional[mpl.colorbar.Colorbar]]:
    """
    在给定的 Axes 上绘制混淆矩阵热力图。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标坐标轴（必须提供）。
    cm : np.ndarray
        混淆矩阵（二维数组）。
    classes : List[str]
        类别名称列表。
    title : str, optional
        图标题。
    normalize : bool, optional
        是否归一化（行归一化）。
    cmap : str or Colormap, optional
        colormap，默认 'Blues'。
    add_colorbar : bool, optional
        是否添加 colorbar。
    cbar_label : str, optional
        colorbar 标签（如 'Frequency' 或 'Probability'）。
    cbar_shrink : float, optional
        colorbar 缩放比例。
    cbar_pad : float, optional
        colorbar 与主图间距。
    rc_overrides : dict, optional
        样式覆盖配置。
    use_latex : bool, optional
        是否启用 LaTeX 渲染。

    Returns
    -------
    ax : matplotlib.axes.Axes
        绘图后的 axes。
    cbar : matplotlib.colorbar.Colorbar or None
        如果 add_colorbar=True，返回 colorbar 对象；否则 None。
    """
    if ax is None:
        raise ValueError("ax must be provided as the first argument.")

    # 应用样式
    rc = _apply_rc(rc_overrides)
    rc['text.usetex'] = use_latex
    rc['font.size'] = 9  # 热力图文字小一点

    with plt.rc_context(rc):
        # 归一化
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘制热力图
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # 添加 colorbar（如果需要）
        cbar = None
        if add_colorbar:
            fig = ax.get_figure()
            cbar = fig.colorbar(im, ax=ax, shrink=cbar_shrink, pad=cbar_pad)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=rc['axes.labelsize'])

        # 设置 ticks 和 labels
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel='True Label',
            xlabel='Predicted Label'
        )

        # 在每个格子中显示数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm[i, j] > thresh else "black"
                value_str = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
                ax.text(
                    j, i, value_str,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=rc['font.size']
                )

        # 旋转 x 轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    return ax, cbar


# -----------------------------
# 5. t-SNE / UMAP 降维散点图
# -----------------------------
def plot_scatter_2d(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        labels: Optional[Union[List[int], np.ndarray]] = None,
        title: str = '',
        xlabel: str = 'Dim 1',
        ylabel: str = 'Dim 2',
        alpha: float = 0.7,
        s: float = 10,
        rc_overrides: Optional[Dict[str, Any]] = None,
        use_latex: bool = False,
) -> plt.Axes:
    if ax is None:
        raise ValueError("ax must be provided.")

    rc = _apply_rc(rc_overrides)
    rc['text.usetex'] = use_latex

    with plt.rc_context(rc):
        if labels is not None:
            scatter = ax.scatter(x, y, c=labels, cmap='tab10', alpha=alpha, s=s)
            # 注意：不再自动加 legend！由用户处理
        else:
            ax.scatter(x, y, alpha=alpha, s=s)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return ax


# -----------------------------
# 6. PR 曲线 / ROC 曲线
# -----------------------------
def plot_curve(
        ax: plt.Axes,
        x_vals: List[np.ndarray],
        y_vals: List[np.ndarray],
        labels: List[str],
        title: str = '',
        xlabel: str = 'Recall',
        ylabel: str = 'Precision',
        linestyle: str = '-',
        rc_overrides: Optional[Dict[str, Any]] = None,
        use_latex: bool = False
) -> plt.Axes:
    if ax is None:
        raise ValueError("ax must be provided.")

    rc = _apply_rc(rc_overrides)
    rc['text.usetex'] = use_latex

    with plt.rc_context(rc):
        for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
            ax.plot(x, y, label=label, linestyle=linestyle)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True, linestyle='--', alpha=0.5)
    return ax


if __name__ == "__main__":
    # -----------------------------
    # 示例1：训练曲线（需先创建 fig, ax）
    # -----------------------------
    epochs = list(range(1, 21))
    train_loss = np.exp(-0.3 * np.array(epochs)) + 0.01 * np.random.randn(20)
    val_loss = np.exp(-0.25 * np.array(epochs)) + 0.02 * np.random.randn(20)

    fig, ax = plt.subplots(figsize=(4, 2.8))  # 注意：样式中的 figsize 不再生效，需在此设置
    plot_training_curve(
        epochs=epochs,
        train_vals=train_loss,
        val_vals=val_loss,
        title='Training Loss',
        ylabel='Cross-Entropy Loss',
        xlabel='Epoch',
        rc_overrides=None,  # 若需样式覆盖，仍可传入（但 figsize 需在 plt.subplots 中设）
        use_latex=False,
        ax=ax  # ← 必须显式传入
    )
    ax.legend()  # 图例由用户控制
    plt.show(block=False)

    # -----------------------------
    # 示例2：消融实验柱状图
    # -----------------------------
    ablation_labels = ['w/o A', 'w/o B', 'Full Model']
    scores = [0.72, 0.78, 0.85]
    errors = [0.02, 0.015, 0.01]

    fig, ax = plt.subplots()
    plot_ablation_bar(
        labels=ablation_labels,
        values=scores,
        errors=errors,
        ylabel='Accuracy',
        title='Ablation Study',
        ax=ax  # ← 必须传入
    )
    # 注意：此图通常不需要 legend，故未加
    plt.show(block=False)

    # -----------------------------
    # 示例3：混淆矩阵（ax 作为第一个位置参数！）
    # -----------------------------
    from sklearn.metrics import confusion_matrix
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Cat', 'Dog', 'Bird']

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax, cbar = plot_confusion_matrix(
        ax,  # ← 第一个参数是 ax！
        cm,
        classes,
        normalize=True,
        title='Confusion Matrix (Normalized)',
        add_colorbar=True,
        cbar_label='Probability',
        rc_overrides=None,
        use_latex=False
    )
    plt.show()
