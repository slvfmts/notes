#!/usr/bin/env python3
"""Generate analysis charts for LLM editorial edits."""

import re
import difflib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ──────────────────────────────────────────────────────────────
EXCEL_PATH = '/Users/slava/Downloads/Текст, сгенерированный GEMINI.xlsx'
OUT_DIR = '/Users/slava/projects/llm-content/charts'

COLOR_NC = '#4A90D9'
COLOR_PUSH = '#E8734A'
LABEL_NC = 'Email'
LABEL_PUSH = 'Пуш'

# Font that supports Cyrillic
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# ── Load data ───────────────────────────────────────────────────────────
print("Loading data...")
df_nc = pd.read_excel(EXCEL_PATH, sheet_name='NC')
df_push = pd.read_excel(EXCEL_PATH, sheet_name='Пуш')

# Drop rows where before or after is missing
df_nc = df_nc.dropna(subset=['before', 'after']).reset_index(drop=True)
df_push = df_push.dropna(subset=['before', 'after']).reset_index(drop=True)

print(f"Email pairs: {len(df_nc)}, Пуш pairs: {len(df_push)}")

# ── Helper functions ────────────────────────────────────────────────────

def clean_text_for_jaccard(text):
    """Lowercase, remove markdown and punctuation, split into words."""
    text = str(text).lower()
    # Remove markdown formatting
    text = re.sub(r'[#*_\[\]\(\)>`~\-|]', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return set(words)

def jaccard_similarity(text1, text2):
    s1 = clean_text_for_jaccard(text1)
    s2 = clean_text_for_jaccard(text2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def edit_distance_ratio(text1, text2):
    return difflib.SequenceMatcher(None, str(text1), str(text2)).ratio()

def count_emoji(text):
    return len(re.findall(r'[\U0001F300-\U0001FAFF\U00002702-\U000027B0]', str(text)))

def count_md_headers(text):
    return len(re.findall(r'^#+\s', str(text), re.MULTILINE))

def count_words(text):
    # Remove markdown, then count words
    t = re.sub(r'[#*_\[\]\(\)>`~\-|]', ' ', str(text))
    return len(t.split())

def compression_ratio(before, after):
    lb = len(str(before))
    if lb == 0:
        return 1.0
    return len(str(after)) / lb


# ── Compute metrics ────────────────────────────────────────────────────
print("Computing Jaccard & edit distance...")
for df, label in [(df_nc, 'Email'), (df_push, 'Пуш')]:
    df['jaccard'] = df.apply(lambda r: jaccard_similarity(r['before'], r['after']), axis=1)
    df['edit_ratio'] = df.apply(lambda r: edit_distance_ratio(r['before'], r['after']), axis=1)
    df['compression'] = df.apply(lambda r: compression_ratio(r['before'], r['after']), axis=1)
    df['emoji_before'] = df['before'].apply(count_emoji)
    df['emoji_after'] = df['after'].apply(count_emoji)
    df['headers_before'] = df['before'].apply(count_md_headers)
    df['headers_after'] = df['after'].apply(count_md_headers)
    df['words_before'] = df['before'].apply(count_words)
    df['words_after'] = df['after'].apply(count_words)
    print(f"  {label}: jaccard mean={df['jaccard'].mean():.3f}, edit_ratio mean={df['edit_ratio'].mean():.3f}")

print("Computing cosine similarity with sentence-transformers...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

for df, label in [(df_nc, 'Email'), (df_push, 'Пуш')]:
    befores = df['before'].tolist()
    afters = df['after'].tolist()
    emb_b = model.encode(befores, show_progress_bar=False)
    emb_a = model.encode(afters, show_progress_bar=False)
    sims = [cosine_similarity([emb_b[i]], [emb_a[i]])[0][0] for i in range(len(df))]
    df['cosine_sim'] = sims
    print(f"  {label}: cosine_sim mean={df['cosine_sim'].mean():.3f}")


# ── Chart 1: Gap between meaning and text ───────────────────────────────
print("Generating Chart 1...")
fig1, ax1 = plt.subplots(figsize=(10, 6))

metrics = ['Cosine similarity\n(смысл)', 'Jaccard по словам\n(лексика)', 'Edit distance ratio\n(текст)']
nc_vals = [df_nc['cosine_sim'].mean(), df_nc['jaccard'].mean(), df_nc['edit_ratio'].mean()]
push_vals = [df_push['cosine_sim'].mean(), df_push['jaccard'].mean(), df_push['edit_ratio'].mean()]

x = np.arange(len(metrics))
w = 0.32
bars1 = ax1.bar(x - w/2, nc_vals, w, label=LABEL_NC, color=COLOR_NC, edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x + w/2, push_vals, w, label=LABEL_PUSH, color=COLOR_PUSH, edgecolor='white', linewidth=0.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel('Среднее значение', fontsize=11)
ax1.set_title('Разрыв между смыслом и текстом', fontsize=14, fontweight='bold', pad=15)

# Dynamic subtitle
cos_avg = (df_nc['cosine_sim'].mean() + df_push['cosine_sim'].mean()) / 2
edit_change = 1 - (df_nc['edit_ratio'].mean() + df_push['edit_ratio'].mean()) / 2
ax1.text(0.5, -0.15, f'Смысл сохраняется на {cos_avg*100:.0f}%, но текст переписан на {edit_change*100:.0f}%+',
         transform=ax1.transAxes, ha='center', fontsize=10, color='#666666', style='italic')

ax1.legend(fontsize=10, frameon=False)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
fig1.tight_layout()
fig1.savefig(f'{OUT_DIR}/chart1_meaning_vs_text.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("  Saved chart1_meaning_vs_text.png")


# ── Chart 2: Cosine similarity distribution ─────────────────────────────
print("Generating Chart 2...")
fig2, ax2 = plt.subplots(figsize=(10, 6))

bins = np.linspace(
    min(df_nc['cosine_sim'].min(), df_push['cosine_sim'].min()) - 0.02,
    1.0, 25
)

ax2.hist(df_nc['cosine_sim'], bins=bins, alpha=0.6, color=COLOR_NC, label=LABEL_NC, edgecolor='white', linewidth=0.5)
ax2.hist(df_push['cosine_sim'], bins=bins, alpha=0.6, color=COLOR_PUSH, label=LABEL_PUSH, edgecolor='white', linewidth=0.5)

# Medians
med_nc = df_nc['cosine_sim'].median()
med_push = df_push['cosine_sim'].median()
ax2.axvline(med_nc, color=COLOR_NC, linestyle='--', linewidth=2, label=f'Медиана Email: {med_nc:.2f}')
ax2.axvline(med_push, color=COLOR_PUSH, linestyle='--', linewidth=2, label=f'Медиана Пуш: {med_push:.2f}')

ax2.set_xlabel('Cosine similarity', fontsize=11)
ax2.set_ylabel('Количество пар', fontsize=11)
ax2.set_title('Распределение cosine similarity', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10, frameon=False)
fig2.tight_layout()
fig2.savefig(f'{OUT_DIR}/chart2_cosine_distribution.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("  Saved chart2_cosine_distribution.png")


# ── Chart 3: Compression ratio ──────────────────────────────────────────
print("Generating Chart 3...")
fig3, ax3 = plt.subplots(figsize=(10, 6))

all_comp = pd.concat([df_nc['compression'], df_push['compression']])
lo = max(0, all_comp.quantile(0.01) - 0.05)
# Clip at 2.0 so outliers don't stretch the chart; note outlier count
hi_clip = 2.0
n_outliers = (all_comp > hi_clip).sum()
bins3 = np.linspace(lo, hi_clip, 30)

# Clip values for histogram display
nc_comp_clipped = df_nc['compression'].clip(upper=hi_clip)
push_comp_clipped = df_push['compression'].clip(upper=hi_clip)

ax3.hist(nc_comp_clipped, bins=bins3, alpha=0.6, color=COLOR_NC, label=LABEL_NC, edgecolor='white', linewidth=0.5)
ax3.hist(push_comp_clipped, bins=bins3, alpha=0.6, color=COLOR_PUSH, label=LABEL_PUSH, edgecolor='white', linewidth=0.5)

# Zones
ax3.axvspan(lo, 0.95, alpha=0.07, color='green', zorder=0)
ax3.axvspan(0.95, 1.05, alpha=0.07, color='gray', zorder=0)
ax3.axvspan(1.05, hi_clip, alpha=0.07, color='red', zorder=0)

# Zone labels at top
y_top = ax3.get_ylim()[1] * 0.92
ax3.text((lo + 0.95)/2, y_top, 'Сокращает', ha='center', fontsize=9, color='green', fontweight='bold', alpha=0.7)
ax3.text(1.0, y_top, 'Не меняет', ha='center', fontsize=9, color='gray', fontweight='bold', alpha=0.7)
ax3.text((1.05 + hi_clip)/2, y_top, 'Раздувает', ha='center', fontsize=9, color='red', fontweight='bold', alpha=0.7)
if n_outliers > 0:
    ax3.text(0.98, 0.85, f'{n_outliers} выбросов > {hi_clip:.0f}x\nне показаны',
             transform=ax3.transAxes, ha='right', fontsize=8, color='#999999', style='italic')

# Medians
med_nc_c = df_nc['compression'].median()
med_push_c = df_push['compression'].median()
ax3.axvline(med_nc_c, color=COLOR_NC, linestyle='--', linewidth=2, label=f'Медиана Email: {med_nc_c:.2f}')
ax3.axvline(med_push_c, color=COLOR_PUSH, linestyle='--', linewidth=2, label=f'Медиана Пуш: {med_push_c:.2f}')

ax3.set_xlabel('Compression ratio (len(after) / len(before))', fontsize=11)
ax3.set_ylabel('Количество пар', fontsize=11)
ax3.set_title('Что делает редактор с объёмом текста', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=10, frameon=False)
fig3.tight_layout()
fig3.savefig(f'{OUT_DIR}/chart3_compression_ratio.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("  Saved chart3_compression_ratio.png")


# ── Chart 4: Stylistic edits ────────────────────────────────────────────
print("Generating Chart 4...")
fig4, ax4 = plt.subplots(figsize=(10, 6))

style_metrics = ['Эмодзи', 'Markdown-заголовки (#)', 'Длина (слова)']

# Email before/after
nc_emoji_b = df_nc['emoji_before'].mean()
nc_emoji_a = df_nc['emoji_after'].mean()
nc_head_b = df_nc['headers_before'].mean()
nc_head_a = df_nc['headers_after'].mean()
nc_words_b = df_nc['words_before'].mean()
nc_words_a = df_nc['words_after'].mean()

# Push before/after
push_emoji_b = df_push['emoji_before'].mean()
push_emoji_a = df_push['emoji_after'].mean()
push_head_b = df_push['headers_before'].mean()
push_head_a = df_push['headers_after'].mean()
push_words_b = df_push['words_before'].mean()
push_words_a = df_push['words_after'].mean()

# We have 4 groups: Email before, Email after, Push before, Push after
# For each metric. Use grouped bar chart with sub-groups.
# Better approach: for each metric, 4 bars (Email before, Email after, Push before, Push after)

vals_matrix = [
    [nc_emoji_b, nc_emoji_a, push_emoji_b, push_emoji_a],
    [nc_head_b, nc_head_a, push_head_b, push_head_a],
    [nc_words_b, nc_words_a, push_words_b, push_words_a],
]

# Since the scales are very different (emoji ~2-5, headers ~5-10, words ~80-200),
# use a normalized approach or separate axes. Better: use subplots within chart 4.
plt.close(fig4)

fig4, axes4 = plt.subplots(1, 3, figsize=(10, 6))
fig4.suptitle('Стилистические правки: до и после редактуры', fontsize=14, fontweight='bold', y=1.02)

bar_labels = ['Email\nдо', 'Email\nпосле', 'Пуш\nдо', 'Пуш\nпосле']
colors_4 = [COLOR_NC, COLOR_NC, COLOR_PUSH, COLOR_PUSH]
alphas_4 = [0.5, 1.0, 0.5, 1.0]
hatches_4 = ['///', '', '///', '']

for idx, (metric_name, vals) in enumerate(zip(style_metrics, vals_matrix)):
    ax = axes4[idx]
    x4 = np.arange(4)
    bars = ax.bar(x4, vals, color=colors_4, edgecolor='white', linewidth=0.5, width=0.65)
    for b, a, h in zip(bars, alphas_4, hatches_4):
        b.set_alpha(a)
        b.set_hatch(h)
        b.set_edgecolor('#666666')
        b.set_linewidth(0.5)

    # Value labels
    for b in bars:
        height = b.get_height()
        fmt = f'{height:.1f}' if height < 20 else f'{height:.0f}'
        ax.text(b.get_x() + b.get_width()/2, height + max(vals)*0.02,
                fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x4)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_title(metric_name, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig4.tight_layout()
fig4.savefig(f'{OUT_DIR}/chart4_stylistic_edits.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print("  Saved chart4_stylistic_edits.png")


# ── Combined 2x2 chart ──────────────────────────────────────────────────
print("Generating combined chart...")
import matplotlib.gridspec as gridspec

fig_all = plt.figure(figsize=(16, 12))
fig_all.suptitle('Анализ редакторских правок текстов нейросети', fontsize=18, fontweight='bold', y=0.98)

# Main 2x2 grid: top-left, top-right, bottom-left take 1 slot each;
# bottom-right is subdivided into 3 sub-axes
outer_gs = gridspec.GridSpec(2, 2, figure=fig_all, hspace=0.35, wspace=0.3)

# Chart 1 replica
ax = fig_all.add_subplot(outer_gs[0, 0])
bars1 = ax.bar(x - w/2, nc_vals, w, label=LABEL_NC, color=COLOR_NC, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, push_vals, w, label=LABEL_PUSH, color=COLOR_PUSH, edgecolor='white', linewidth=0.5)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Среднее значение', fontsize=10)
ax.set_title('Разрыв между смыслом и текстом', fontsize=12, fontweight='bold')
ax.text(0.5, -0.18, f'Смысл сохраняется на {cos_avg*100:.0f}%, но текст переписан на {edit_change*100:.0f}%+',
        transform=ax.transAxes, ha='center', fontsize=9, color='#666666', style='italic')
ax.legend(fontsize=9, frameon=False)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

# Chart 2 replica
ax = fig_all.add_subplot(outer_gs[0, 1])
ax.hist(df_nc['cosine_sim'], bins=bins, alpha=0.6, color=COLOR_NC, label=LABEL_NC, edgecolor='white', linewidth=0.5)
ax.hist(df_push['cosine_sim'], bins=bins, alpha=0.6, color=COLOR_PUSH, label=LABEL_PUSH, edgecolor='white', linewidth=0.5)
ax.axvline(med_nc, color=COLOR_NC, linestyle='--', linewidth=2, label=f'Медиана Email: {med_nc:.2f}')
ax.axvline(med_push, color=COLOR_PUSH, linestyle='--', linewidth=2, label=f'Медиана Пуш: {med_push:.2f}')
ax.set_xlabel('Cosine similarity', fontsize=10)
ax.set_ylabel('Количество пар', fontsize=10)
ax.set_title('Распределение cosine similarity', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, frameon=False)

# Chart 3 replica
ax = fig_all.add_subplot(outer_gs[1, 0])
ax.hist(nc_comp_clipped, bins=bins3, alpha=0.6, color=COLOR_NC, label=LABEL_NC, edgecolor='white', linewidth=0.5)
ax.hist(push_comp_clipped, bins=bins3, alpha=0.6, color=COLOR_PUSH, label=LABEL_PUSH, edgecolor='white', linewidth=0.5)
ax.axvspan(lo, 0.95, alpha=0.07, color='green', zorder=0)
ax.axvspan(0.95, 1.05, alpha=0.07, color='gray', zorder=0)
ax.axvspan(1.05, hi_clip, alpha=0.07, color='red', zorder=0)
y_top3 = ax.get_ylim()[1] * 0.92
ax.text((lo + 0.95)/2, y_top3, 'Сокращает', ha='center', fontsize=8, color='green', fontweight='bold', alpha=0.7)
ax.text(1.0, y_top3, 'Не меняет', ha='center', fontsize=8, color='gray', fontweight='bold', alpha=0.7)
ax.text((1.05 + hi_clip)/2, y_top3, 'Раздувает', ha='center', fontsize=8, color='red', fontweight='bold', alpha=0.7)
ax.axvline(med_nc_c, color=COLOR_NC, linestyle='--', linewidth=2, label=f'Медиана Email: {med_nc_c:.2f}')
ax.axvline(med_push_c, color=COLOR_PUSH, linestyle='--', linewidth=2, label=f'Медиана Пуш: {med_push_c:.2f}')
ax.set_xlabel('Compression ratio (len(after) / len(before))', fontsize=10)
ax.set_ylabel('Количество пар', fontsize=10)
ax.set_title('Что делает редактор с объёмом текста', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, frameon=False)

# Chart 4 replica: 3 sub-axes in bottom-right
inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1, 1], wspace=0.4)
sub_axes = [fig_all.add_subplot(inner_gs[0, j]) for j in range(3)]

for idx, (metric_name, vals) in enumerate(zip(style_metrics, vals_matrix)):
    ax = sub_axes[idx]
    x4 = np.arange(4)
    bars = ax.bar(x4, vals, color=colors_4, edgecolor='white', linewidth=0.5, width=0.65)
    for b, a, hh in zip(bars, alphas_4, hatches_4):
        b.set_alpha(a)
        b.set_hatch(hh)
        b.set_edgecolor('#666666')
        b.set_linewidth(0.5)
    for b in bars:
        height = b.get_height()
        fmt = f'{height:.1f}' if height < 20 else f'{height:.0f}'
        ax.text(b.get_x() + b.get_width()/2, height + max(vals)*0.02,
                fmt, ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x4)
    ax.set_xticklabels(bar_labels, fontsize=6)
    ax.set_title(metric_name, fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add a shared title for Chart 4 sub-axes area
# Position it above the sub-axes
bbox_left = sub_axes[0].get_position()
bbox_right = sub_axes[2].get_position()
mid_x = (bbox_left.x0 + bbox_right.x1) / 2
fig_all.text(mid_x, bbox_left.y1 + 0.03, 'Стилистические правки: до и после редактуры',
             ha='center', fontsize=12, fontweight='bold')

fig_all.savefig(f'{OUT_DIR}/llm_analysis_all.png', dpi=300, bbox_inches='tight')
plt.close(fig_all)
print("  Saved llm_analysis_all.png")

print("\nAll charts generated successfully!")
print(f"\nSummary statistics:")
print(f"  Email pairs: {len(df_nc)}")
print(f"  Push pairs: {len(df_push)}")
print(f"  Email - cosine: {df_nc['cosine_sim'].mean():.3f}, jaccard: {df_nc['jaccard'].mean():.3f}, edit_ratio: {df_nc['edit_ratio'].mean():.3f}")
print(f"  Push - cosine: {df_push['cosine_sim'].mean():.3f}, jaccard: {df_push['jaccard'].mean():.3f}, edit_ratio: {df_push['edit_ratio'].mean():.3f}")
print(f"  Email compression median: {df_nc['compression'].median():.3f}")
print(f"  Push compression median: {df_push['compression'].median():.3f}")
