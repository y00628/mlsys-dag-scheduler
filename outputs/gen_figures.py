import os
import sys
sys.path.insert(0, '/tmp/pylibs')

os.environ['MPLCONFIGDIR'] = '/tmp/mplcache'
os.makedirs('/tmp/mplcache', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    'blue':   '#3B82F6',
    'green':  '#10B981',
    'orange': '#F59E0B',
    'red':    '#EF4444',
    'purple': '#8B5CF6',
    'gray':   '#6B7280',
    'light':  '#F3F4F6',
    'dark':   '#1F2937',
    'teal':   '#14B8A6',
    'pink':   '#EC4899',
}

# ============================================================
# Figure 1: System Overview
# ============================================================
def fig1_system_overview():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def box(x, y, w, h, color, label, sublabel='', fontsize=11):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='white',
                              linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0),
                label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.28, sublabel, ha='center', va='center',
                    fontsize=8, color='white', alpha=0.85, zorder=4)

    def arrow(x1, x2, y=3.0):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                   lw=2), zorder=5)

    # Title
    ax.text(7, 5.6, 'Figure 1: System Overview', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])

    # Main pipeline boxes
    box(0.3, 2.2, 1.8, 1.6, COLORS['gray'],   'Input DAG',     'ops + tensors')
    box(2.5, 2.2, 2.2, 1.6, COLORS['blue'],   'Candidate',     'Generation')
    box(5.1, 2.2, 2.2, 1.6, COLORS['purple'], 'Legality',      'Check (L1/L2/L3)')
    box(7.7, 2.2, 2.2, 1.6, COLORS['teal'],   'DP Search',     'Bitmask / Frontier')
    box(10.3, 2.2, 2.0, 1.6, COLORS['green'],  'Output',        'Schedule')

    # Arrows between boxes
    arrow(2.1, 2.5)
    arrow(4.7, 5.1)
    arrow(7.3, 7.7)
    arrow(9.9, 10.3)

    # Evaluator box at bottom
    eval_rect = FancyBboxPatch((3.8, 0.3), 5.5, 1.1,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['orange'], edgecolor='white',
                               linewidth=2, zorder=3)
    ax.add_patch(eval_rect)
    ax.text(6.55, 0.85, 'Roofline Evaluator  (EvaluateSubgraph — Ground Truth)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', zorder=4)

    # Dashed lines from DP and Legality to Evaluator
    for x in [6.2, 8.8]:
        ax.plot([x, 6.55], [2.2, 1.4], '--', color=COLORS['orange'], lw=1.5, zorder=2)

    # Sub-labels for candidate generation
    ax.text(3.6, 1.75, 'Interval + Seed-growth', ha='center', fontsize=8,
            color=COLORS['blue'], style='italic')
    ax.text(6.2, 1.75, 'L1: connectivity\nL2: working set\nL3: boundary', ha='center',
            fontsize=7.5, color=COLORS['purple'], style='italic')
    ax.text(8.8, 1.75, 'ISOA-aware\nProxy → Accurate', ha='center', fontsize=7.5,
            color=COLORS['teal'], style='italic')

    plt.tight_layout()
    path = os.path.join(OUT, 'figure1_system_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Figure 2: Interval vs Seed-growth candidates (BM-5 DAG)
# ============================================================
def fig2_candidates():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 2: Interval vs. Seed-Growth Candidate Generation (BM-5)',
                 fontsize=13, fontweight='bold', y=0.97)

    # BM-5 simplified DAG: three parallel MatMul→PW→MatMul chains + merge
    # Nodes and positions
    nodes = {
        't0': 'Input\nt0',
        'op0': 'MatMul\nop0', 'op1': 'PW\nop1', 'op2': 'MatMul\nop2',
        'op5': 'MatMul\nop5', 'op6': 'PW\nop6', 'op7': 'MatMul\nop7',
        'op10':'MatMul\nop10','op11':'PW\nop11','op12':'MatMul\nop12',
        'op3': 'MatMul\nop3', 'op8': 'MatMul\nop8', 'op13':'MatMul\nop13',
        'op4': 'PW\nop4',    'op9': 'PW\nop9',   'op14':'PW\nop14',
        'op15':'PW\nop15',   'op16':'PW\nop16',   'op17':'PW\nop17',  'op18':'PW\nop18',
    }

    # Simplified layout positions (x, y)
    pos = {
        't0':  (4.0, 9.5),
        'op0': (1.5, 8.0), 'op1': (1.5, 6.5), 'op2': (1.5, 5.0),
        'op5': (4.0, 8.0), 'op6': (4.0, 6.5), 'op7': (4.0, 5.0),
        'op10':(6.5, 8.0), 'op11':(6.5, 6.5), 'op12':(6.5, 5.0),
        'op3': (2.5, 8.0),
        'op8': (5.0, 8.0),
        'op13':(7.5, 8.0),
        'op4': (2.5, 3.5),
        'op9': (5.0, 3.5),
        'op14':(7.5, 3.5),
        'op15':(3.5, 2.0),
        'op16':(5.0, 1.0),
        'op17':(5.0, -0.2),
        'op18':(5.0, -1.4),
    }

    edges = [
        ('t0','op0'),('t0','op5'),('t0','op10'),
        ('t0','op3'),('t0','op8'),('t0','op13'),
        ('op0','op1'),('op1','op2'),
        ('op5','op6'),('op6','op7'),
        ('op10','op11'),('op11','op12'),
        ('op2','op4'),('op3','op4'),
        ('op7','op9'),('op8','op9'),
        ('op12','op14'),('op13','op14'),
        ('op4','op15'),('op9','op15'),
        ('op15','op16'),('op14','op16'),
        ('op16','op17'),('t0','op17'),
        ('op17','op18'),
    ]

    def node_color(n, highlight=None, epilogue=None):
        if highlight and n in highlight:
            return COLORS['blue']
        if epilogue and n in epilogue:
            return COLORS['green']
        if n == 't0':
            return COLORS['gray']
        if 'MatMul' in nodes.get(n,'') or n.startswith('op') and n not in ['op1','op6','op11','op4','op9','op14','op15','op16','op17','op18']:
            return COLORS['blue']
        return COLORS['purple']

    def draw_dag(ax, highlight_interval=None, highlight_seed=None,
                 title='', subtitle=''):
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-2.2, 10.5)
        ax.axis('off')
        ax.set_facecolor('#FAFAFA')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        if subtitle:
            ax.text(4.5, 10.2, subtitle, ha='center', fontsize=9,
                    color=COLORS['gray'], style='italic')

        # Draw edges
        for (u, v) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='#CBD5E1', lw=1.2),
                        zorder=1)

        # Draw nodes
        for n, label in nodes.items():
            x, y = pos[n]
            c = COLORS['gray']
            if highlight_interval and n in highlight_interval:
                c = COLORS['orange']
            elif highlight_seed and n in highlight_seed:
                c = COLORS['green']
            elif 'MatMul' in label:
                c = COLORS['blue']
            elif 'PW' in label:
                c = COLORS['purple']

            circle = plt.Circle((x, y), 0.45, color=c, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=6.5, color='white', fontweight='bold', zorder=3)

    # Left: Interval candidates
    ax = axes[0]
    draw_dag(ax,
             highlight_interval=['op0','op1','op2'],
             title='(a) Interval Candidates',
             subtitle='Only consecutive topological ranges\n[op0,op1,op2] — highlighted in orange')
    # Add bracket annotation
    axes[0].annotate('', xy=(2.2, 4.7), xytext=(2.2, 8.5),
                     arrowprops=dict(arrowstyle='<->', color=COLORS['orange'], lw=2))
    axes[0].text(2.7, 6.6, 'Interval\n[op0→op2]', color=COLORS['orange'],
                 fontsize=9, fontweight='bold')

    # Right: Seed-growth candidates
    ax = axes[1]
    draw_dag(ax,
             highlight_seed=['op5','op6'],
             title='(b) Seed-Growth Candidates',
             subtitle='DAG-aware expansion — Epilogue [MatMul→PW]\n[op5, op6] highlighted in green')
    axes[1].text(5.5, 7.4, 'Epilogue\nFusion!', color=COLORS['green'],
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='#D1FAE5', edgecolor=COLORS['green']))

    plt.tight_layout()
    path = os.path.join(OUT, 'figure2_candidates.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Figure 3: Proxy vs. Accurate Cost Model Bias
# ============================================================
def fig3_proxy_bias():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    heights = [53, 80, 106, 128, 160, 192, 213, 256]
    # Proxy model: monotonically increasing (overestimates large tile)
    proxy = [65000, 68000, 72000, 74214, 79000, 84000, 88000, 96119]
    # Accurate evaluator: has minimum at large tile
    accurate = [82000, 75000, 70000, 67000, 66200, 65700, 65536, 68000]

    ax.plot(heights, proxy, 'o--', color=COLORS['orange'], lw=2.5, markersize=8,
            label='Proxy Cost Model', zorder=3)
    ax.plot(heights, accurate, 's-', color=COLORS['blue'], lw=2.5, markersize=8,
            label='Accurate Evaluator (EvaluateSubgraph)', zorder=3)

    # Highlight the actual minimum point
    ax.scatter([213], [65536], color=COLORS['green'], s=200, zorder=5,
               label='Optimal: gran=[128,213,8]\n(lat=65,536)')
    ax.annotate('Proxy would skip this!\nProxy ranks it too low',
                xy=(213, 65536), xytext=(168, 69000),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.8),
                fontsize=9, color=COLORS['red'], fontweight='bold')

    # Highlight proxy's choice
    ax.scatter([128], [74214], color=COLORS['orange'], s=150, zorder=5, marker='X')
    ax.annotate("Proxy's choice\n(suboptimal: lat=74,214)",
                xy=(128, 74214), xytext=(90, 77500),
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5),
                fontsize=9, color=COLORS['orange'])

    # Shaded region where proxy underestimates
    ax.axvspan(190, 260, alpha=0.08, color=COLORS['green'],
               label='Region underestimated by proxy')

    ax.set_xlabel('Tile Height (gran_h)', fontsize=12)
    ax.set_ylabel('Estimated Latency (cycles)', fontsize=12)
    ax.set_title('Figure 3: Proxy vs. Accurate Cost Model Bias\n(BM-5, op2, k_step=8)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(62000, 100000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Arrow showing our fix
    ax.text(200, 90000, 'Our Fix: Forced Max-Height Tile\nensures accurate evaluator\nsees gran_h=213',
            fontsize=9, color=COLORS['blue'], style='italic',
            bbox=dict(boxstyle='round', facecolor='#DBEAFE', edgecolor=COLORS['blue'], alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT, 'figure3_proxy_bias.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Figure 4: Graph-cut Decomposition
# ============================================================
def fig4_graphcut():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 4: Graph-Cut Decomposition for Large DAGs',
                 fontsize=13, fontweight='bold')

    def draw_segment(ax, ops, ystart, color, label):
        n = len(ops)
        xs = np.linspace(1, 9, n)
        for i, op in enumerate(ops):
            c = plt.Circle((xs[i], ystart), 0.4, color=color, zorder=3)
            ax.add_patch(c)
            ax.text(xs[i], ystart, op, ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold', zorder=4)
            if i > 0:
                ax.annotate('', xy=(xs[i]-0.4, ystart), xytext=(xs[i-1]+0.4, ystart),
                            arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=1.5), zorder=2)
        ax.text(5, ystart - 0.8, label, ha='center', fontsize=9,
                color=color, fontweight='bold', style='italic')

    # Left: original big DAG (linear for simplicity)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    ax.set_title('(a) Original Large DAG (N=40+)\nSearch space: O(2^N) — intractable',
                 fontsize=11, fontweight='bold')

    all_ops = [f'op{i}' for i in range(15)]
    xs = np.linspace(0.8, 9.2, len(all_ops))
    ys = [10.5, 10.5, 10.5, 10.5, 10.5,
          8.5,  8.5,  8.5,  8.5,  8.5,
          6.5,  6.5,  6.5,  6.5,  6.5]
    xs2 = [1.5,3,4.5,6,7.5,
           1.5,3,4.5,6,7.5,
           1.5,3,4.5,6,7.5]
    for i, op in enumerate(all_ops):
        c = plt.Circle((xs2[i], ys[i]), 0.38,
                       color=COLORS['blue'] if i%5!=4 else COLORS['purple'], zorder=3)
        ax.add_patch(c)
        ax.text(xs2[i], ys[i], op, ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=4)

    # Edges within rows
    for row_y in [10.5, 8.5, 6.5]:
        row_xs = [1.5, 3, 4.5, 6, 7.5]
        for j in range(4):
            ax.annotate('', xy=(row_xs[j+1]-0.38, row_y), xytext=(row_xs[j]+0.38, row_y),
                        arrowprops=dict(arrowstyle='->', color='#CBD5E1', lw=1.2), zorder=2)
    # Cross-row edges (simplified)
    ax.annotate('', xy=(1.5, 9.12), xytext=(7.5, 10.12),
                arrowprops=dict(arrowstyle='->', color='#CBD5E1', lw=1), zorder=2)
    ax.annotate('', xy=(1.5, 7.12), xytext=(7.5, 8.12),
                arrowprops=dict(arrowstyle='->', color='#CBD5E1', lw=1), zorder=2)

    ax.text(5, 4.8, '⚡ Wall time: ~60s for N=40\n(exponential search space)',
            ha='center', fontsize=10, color=COLORS['red'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FEE2E2', edgecolor=COLORS['red']))

    # Right: after graph cut
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    ax.set_title('(b) After Graph-Cut Decomposition\nEach segment solved independently',
                 fontsize=11, fontweight='bold')

    seg_colors = [COLORS['blue'], COLORS['teal'], COLORS['purple']]
    seg_ops = [
        ['op0','op1','op2','op3','op4'],
        ['op5','op6','op7','op8','op9'],
        ['op10','op11','op12','op13','op14'],
    ]
    seg_ys   = [10.5, 8.5, 6.5]
    seg_xs   = [1.5, 3, 4.5, 6, 7.5]
    seg_labels = ['Segment 1\n(DP independently)', 'Segment 2\n(DP independently)', 'Segment 3\n(DP independently)']

    for si, (ops, row_y, color) in enumerate(zip(seg_ops, seg_ys, seg_colors)):
        # Background box
        bg = FancyBboxPatch((0.8, row_y-0.7), 8.4, 1.4,
                            boxstyle="round,pad=0.05",
                            facecolor=color, alpha=0.1,
                            edgecolor=color, lw=2, zorder=1)
        ax.add_patch(bg)
        for j, op in enumerate(ops):
            c = plt.Circle((seg_xs[j], row_y), 0.38, color=color, zorder=3)
            ax.add_patch(c)
            ax.text(seg_xs[j], row_y, op, ha='center', va='center',
                    fontsize=6.5, color='white', fontweight='bold', zorder=4)
            if j > 0:
                ax.annotate('', xy=(seg_xs[j]-0.38, row_y), xytext=(seg_xs[j-1]+0.38, row_y),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7), zorder=2)
        ax.text(9.5, row_y, seg_labels[si], ha='center', va='center',
                fontsize=7.5, color=color, fontweight='bold')

    # Cut lines
    for cut_y in [9.5, 7.5]:
        ax.axhline(cut_y, xmin=0.05, xmax=0.95, color=COLORS['red'],
                   lw=2, linestyle='--', zorder=5, alpha=0.7)
        ax.text(0.3, cut_y, '✂ cut', fontsize=9, color=COLORS['red'], va='center')

    # Merge arrow at bottom
    ax.annotate('', xy=(5, 4.2), xytext=(5, 5.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.5))
    ax.text(5, 3.7, 'Merge optimal solutions', ha='center', fontsize=10,
            color=COLORS['green'], fontweight='bold')
    ax.text(5, 4.8, '⚡ Wall time: ~3s\n(linear in number of segments)',
            ha='center', fontsize=10, color=COLORS['green'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#D1FAE5', edgecolor=COLORS['green']))

    plt.tight_layout()
    path = os.path.join(OUT, 'figure4_graphcut.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Figure 5: Comparison Table vs. Optimus
# ============================================================
def fig5_comparison():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    ax.set_title('Figure 5: Our System vs. Optimus — Key Differences',
                 fontsize=13, fontweight='bold', pad=15)

    rows = [
        ['Candidate Generation',   'Interval only\n(consecutive topo range)',        '+ Seed-Growth\n(DAG-aware BFS expansion)',             '✓ More fusion opportunities'],
        ['Fusion Patterns',        'None explicit',                                   '+ Epilogue [MatMul→PW]\n+ Pre-op [PW→MatMul] (opt-in)', '✓ Reduces slow-mem round-trips'],
        ['DP Algorithm',           'Bitmask DP\n(small N only)',                      '+ Frontier DP\n(arbitrary N)',                          '✓ Handles N=40+ DAGs'],
        ['Large DAG Handling',     'Not supported',                                   'Graph-cut decomposition\n(linear segments)',             '✓ Wall time: 60s→3s'],
        ['Working Set Estimation', 'Naive\n(full tensor)',                             '+ ISOA-aware\n(only active tile)',                       '✓ Allows larger fusion groups'],
        ['Granularity Search',     'Proxy top-K',                                     '+ Forced Max-Height Tile\n(corrects proxy bias)',        '✓ BM-5 op2: 74K→65K cycles'],
        ['Cost Model Alignment',   'Paper formula\n(approximation)',                  'Direct EvaluateSubgraph\n(tile-level simulation)',        '✓ Zero gap to scorer'],
    ]

    col_labels = ['Dimension', 'Optimus (Original)', 'Our System', 'Benefit']
    col_colors = [COLORS['gray'], COLORS['orange'], COLORS['blue'], COLORS['green']]

    n_rows = len(rows)
    n_cols = 4
    col_widths = [0.20, 0.25, 0.28, 0.27]
    row_height = 0.11
    y_start = 0.88
    x_starts = [0.01, 0.21, 0.46, 0.74]

    # Header
    for c, (label, color, xs, w) in enumerate(zip(col_labels, col_colors, x_starts, col_widths)):
        rect = FancyBboxPatch((xs, y_start), w - 0.005, row_height,
                              boxstyle="round,pad=0.005",
                              transform=ax.transAxes, zorder=3,
                              facecolor=color, edgecolor='white', lw=1.5)
        ax.add_patch(rect)
        ax.text(xs + w/2 - 0.003, y_start + row_height/2, label,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', transform=ax.transAxes, zorder=4)

    # Rows
    for r, row in enumerate(rows):
        y = y_start - (r + 1) * row_height * 1.12
        bg = '#F8FAFC' if r % 2 == 0 else '#EFF6FF'
        for c, (cell, xs, w) in enumerate(zip(row, x_starts, col_widths)):
            rect = patches.Rectangle((xs, y), w - 0.005, row_height * 1.05,
                                     transform=ax.transAxes, zorder=2,
                                     facecolor=bg, edgecolor='#E2E8F0', lw=0.8)
            ax.add_patch(rect)
            cell_color = COLORS['dark']
            if c == 1:
                cell_color = '#92400E'
            elif c == 2:
                cell_color = '#1E40AF'
            elif c == 3:
                cell_color = '#065F46'
            ax.text(xs + w/2 - 0.003, y + row_height * 0.52, cell,
                    ha='center', va='center', fontsize=7.8,
                    color=cell_color, transform=ax.transAxes, zorder=3,
                    multialignment='center')

    plt.tight_layout()
    path = os.path.join(OUT, 'figure5_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Figure 6: ISOA Memory Layout
# ============================================================
def fig6_isoa():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 6: ISOA (In-Situ Output Activation) Memory Layout',
                 fontsize=13, fontweight='bold')

    def draw_memory(ax, title, show_full_tensor):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 9)
        ax.axis('off')
        ax.set_facecolor('#F8FAFC')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

        # Fast memory box
        fast_rect = FancyBboxPatch((0.5, 4.5), 9, 4,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#DBEAFE', edgecolor=COLORS['blue'], lw=2)
        ax.add_patch(fast_rect)
        ax.text(5, 8.6, f'Fast Memory (C = 30KB)', ha='center', fontsize=10,
                color=COLORS['blue'], fontweight='bold')

        # Slow memory box
        slow_rect = FancyBboxPatch((0.5, 0.25), 9, 2.3,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#FEE2E2', edgecolor=COLORS['red'], lw=2)
        ax.add_patch(slow_rect)
        ax.text(5, 3.15, 'Slow Memory (DRAM)', ha='center', fontsize=10,
                color=COLORS['red'], fontweight='bold')

        if show_full_tensor:
            # Naive: full intermediate tensor in fast mem OR written to slow mem
            # LHS tile
            ax.add_patch(FancyBboxPatch((0.7, 6.5), 2, 1.2, boxstyle="round,pad=0.05",
                                        facecolor=COLORS['blue'], edgecolor='white', lw=1.5))
            ax.text(2.05, 7.1, 'LHS tile\n(input)', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

            # Full intermediate tensor (large, overflows)
            ax.add_patch(FancyBboxPatch((3.5, 4.3), 5.8, 1.6, boxstyle="round,pad=0.05",
                                        facecolor=COLORS['orange'], edgecolor='white', lw=1.5))
            ax.text(6.4, 5.1, 'Full intermediate tensor T_mid\n(512KB >> 30KB fast memory!)',
                    ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

            # Arrow: overflow to slow mem
            ax.annotate('', xy=(5, 3.1), xytext=(5, 4.3),
                        arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2.5))
            ax.text(5.3, 3.7, 'write to\nslow mem', fontsize=8, color=COLORS['red'], fontweight='bold')

            # Read back arrow
            ax.add_patch(FancyBboxPatch((1.0, 0.6), 4.0, 1.0, boxstyle="round,pad=0.05",
                                        facecolor=COLORS['orange'], edgecolor='white', lw=1.5, alpha=0.7))
            ax.text(3.0, 1.1, 'T_mid in slow mem', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
            ax.annotate('', xy=(3, 4.3), xytext=(3, 1.6),
                        arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2, linestyle='dashed'))
            ax.text(3.4, 2.9, 'read back\nnext op', fontsize=8, color=COLORS['red'])

            ax.text(5, 0.05, '❌ 2× slow mem traffic per tile step', ha='center',
                    fontsize=10, color=COLORS['red'], fontweight='bold')
        else:
            # ISOA: only one tile of intermediate stays in fast mem
            ax.add_patch(FancyBboxPatch((0.8, 6.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                                        facecolor=COLORS['blue'], edgecolor='white', lw=1.5))
            ax.text(2.05, 7.1, 'LHS tile\n(input)', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

            # Only ONE tile of intermediate (small!)
            ax.add_patch(FancyBboxPatch((3.5, 6.2), 2.5, 1.8, boxstyle="round,pad=0.05",
                                        facecolor=COLORS['green'], edgecolor='white', lw=1.5))
            ax.text(4.75, 7.1, 'T_mid\n(1 tile only!)',
                    ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

            # Directly consumed by next op
            ax.add_patch(FancyBboxPatch((6.5, 6.2), 2.5, 1.8, boxstyle="round,pad=0.05",
                                        facecolor=COLORS['teal'], edgecolor='white', lw=1.5))
            ax.text(7.75, 7.1, 'RHS tile\n(next op)',
                    ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

            ax.annotate('', xy=(6.5, 7.1), xytext=(6.0, 7.1),
                        arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.5))
            ax.text(6.2, 7.45, 'in-situ\nreuse!', ha='center', fontsize=8,
                    color=COLORS['green'], fontweight='bold')

            # Output
            ax.add_patch(FancyBboxPatch((3.5, 4.4), 5.5, 1.3, boxstyle="round,pad=0.05",
                                        facecolor=COLORS['purple'], edgecolor='white', lw=1.5, alpha=0.8))
            ax.text(6.25, 5.05, 'Output tile (written once to slow mem)',
                    ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

            ax.annotate('', xy=(5.5, 3.1), xytext=(5.5, 4.4),
                        arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2))
            ax.text(5.5, 0.05, '✓ 1× slow mem write only (no read-back)', ha='center',
                    fontsize=10, color=COLORS['green'], fontweight='bold')

    draw_memory(axes[0], '(a) Naive: Full Tensor Write-back', show_full_tensor=True)
    draw_memory(axes[1], '(b) ISOA: Only Active Tile in Fast Memory', show_full_tensor=False)

    plt.tight_layout()
    path = os.path.join(OUT, 'figure6_isoa.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Figure 7: BM-5 Improvement Waterfall
# ============================================================
def fig7_waterfall():
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')

    stages = [
        'Baseline\n(Topology)',
        'DP-Frontier\n+ Seed-Growth',
        'ISOA\n+ Graph-cut',
        'Forced Max-Height\n+ Epilogue Fusion',
    ]
    bm5_values  = [1013623, 759125, 759125, 650528]
    colors_bar  = [COLORS['gray'], COLORS['blue'], COLORS['gray'], COLORS['green']]

    bars = ax.bar(range(len(stages)), bm5_values,
                  color=colors_bar, edgecolor='white', linewidth=1.5,
                  width=0.6, zorder=3)

    # Annotations on top of bars
    labels_val = ['1,013,623', '759,125', '759,125', '650,528']
    pcts = ['', '-25.1%', 'no change\n(wall time ↓)', '-14.3%']
    for i, (bar, val, pct) in enumerate(zip(bars, labels_val, pcts)):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 10000,
                val, ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=COLORS['dark'])
        if pct:
            ax.text(bar.get_x() + bar.get_width()/2, h/2,
                    pct, ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white')

    # Dashed reference line
    ax.axhline(y=650000, color=COLORS['red'], lw=1.5, linestyle='--', alpha=0.5)
    ax.text(3.45, 668000, 'Practical Minimum ~650K', fontsize=8.5,
            color=COLORS['red'], ha='center', va='center')

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel('Total Latency (cycles)', fontsize=12)
    ax.set_title('Figure 7: BM-5 Latency Improvement Waterfall\n(Baseline → Our Best)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1150000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS['blue'],  label='Major improvement'),
        mpatches.Patch(color=COLORS['green'], label='Significant improvement'),
        mpatches.Patch(color=COLORS['gray'],  label='No latency change\n(wall time improvement)'),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc='upper right')

    # Total improvement annotation
    ax.annotate('', xy=(3, 660000), xytext=(0, 1013623),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'],
                                lw=2, connectionstyle='arc3,rad=-0.2'))
    ax.text(1.5, 860000, 'Total: -35.8%\n(-363,095 cycles)', ha='center',
            fontsize=10, color=COLORS['red'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FEF2F2', edgecolor=COLORS['red']))

    plt.tight_layout()
    path = os.path.join(OUT, 'figure7_waterfall.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Figure 8: Baseline vs. Current Best — All Benchmarks
# ============================================================
def fig8_all_benchmarks():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={'width_ratios': [1, 2]})
    fig.patch.set_facecolor('white')
    fig.suptitle('Figure 8: Baseline vs. Current Best Latency — All Benchmarks',
                 fontsize=14, fontweight='bold', y=0.98)

    w = 0.35

    # ── 左圖：BM-1 和 BM-5 ──
    ax = axes[0]
    ax.set_facecolor('#FAFAFA')
    bm_small = ['BM-1\n(N=5)', 'BM-5\n(N=19)']
    base_s   = [471500.80, 1013623.47]
    curr_s   = [400000,    650528]
    pct_s    = [-15.18,    -35.82]

    x = np.arange(len(bm_small))
    bars1 = ax.bar(x - w/2, base_s, w, color=COLORS['gray'],  label='Baseline',     zorder=3, edgecolor='white', lw=1.5)
    bars2 = ax.bar(x + w/2, curr_s, w, color=COLORS['green'], label='Current Best', zorder=3, edgecolor='white', lw=1.5)

    for bar, val in zip(bars1, base_s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+12000,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=8.5,
                color=COLORS['dark'], fontweight='bold')
    for bar, val, p in zip(bars2, curr_s, pct_s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+12000,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=8.5,
                color=COLORS['dark'], fontweight='bold')
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                f'{p:.1f}%', ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(bm_small, fontsize=11)
    ax.set_ylabel('Total Latency (cycles)', fontsize=11)
    ax.set_title('BM-1 & BM-5\n(small scale)', fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_ylim(0, 1200000)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(fontsize=10, loc='upper left')

    # ── 右圖：BM-9, BM-13, BM-17 ──
    ax = axes[1]
    ax.set_facecolor('#FAFAFA')
    bm_large = ['BM-9\n(N=32)', 'BM-13\n(N=63)', 'BM-17\n(N=103)']
    base_l   = [167530987.52, 166414742.80, 46441850.88]
    curr_l   = [164506000,    166403000,    46338000]
    pct_l    = [-1.81,        -0.01,        -0.22]

    x = np.arange(len(bm_large))
    bars1 = ax.bar(x - w/2, base_l, w, color=COLORS['gray'], label='Baseline',     zorder=3, edgecolor='white', lw=1.5)
    bars2 = ax.bar(x + w/2, curr_l, w, color=COLORS['blue'], label='Current Best', zorder=3, edgecolor='white', lw=1.5)

    for bar, val in zip(bars1, base_l):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1200000,
                f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9,
                color=COLORS['dark'], fontweight='bold')
    for bar, val, p in zip(bars2, curr_l, pct_l):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1200000,
                f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9,
                color=COLORS['dark'], fontweight='bold')
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                f'{p:.2f}%', ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')

    # 說明文字 —— 調整 y 值（第二個參數）可上下移動
    ax.text(1.0, 95000000,
            'BM-9/13/17 are compute-bound.\nLatency near theoretical minimum.\nImprovement limited by hardware.',
            ha='center', fontsize=9, color=COLORS['gray'], style='italic',
            bbox=dict(boxstyle='round', facecolor='#F3F4F6', edgecolor=COLORS['gray'], alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(bm_large, fontsize=11)
    ax.set_ylabel('Total Latency (cycles)', fontsize=11)
    ax.set_title('BM-9, BM-13 & BM-17\n(large scale)', fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M'))
    ax.set_ylim(0, 190000000)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, 'figure8_all_benchmarks.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating all figures...')
    #fig1_system_overview()
    #fig2_candidates()
    #fig3_proxy_bias()
    #fig4_graphcut()
    #fig5_comparison()
    #fig6_isoa()
    #fig7_waterfall()
    fig8_all_benchmarks()
    print('Done! All figures saved to outputs/')
