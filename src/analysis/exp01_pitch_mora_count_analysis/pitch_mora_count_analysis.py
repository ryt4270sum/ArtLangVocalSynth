"""Module for analyzing pitch and mora patterns in musical scores.

This module processes MusicXML files to analyze the relationship between relative pitch and mora occurrences.
"""

# ---- stdlib ----
import sys
from pathlib import Path

# ---- path hack (before local imports) ----
sys.path.append(str((Path(__file__).parent.parent.parent).resolve()))

# ---- third-party ----
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

# ---- local ----
from utils.constants import (  # noqa: F401
    _app,
    _con,
    _fric,
    _nas,
    _plo,
    _ss,
    _tap,
    _vow,
    approximant,
    fricative,
    nasal,
    plosive,
    second_mora_exclude,
    tap,
)
from utils.exp01_plot import (  # noqa: F401
    ThreeLevelArtConfig,
    count_phoneme_occurrence,
    plot_three_levels_art,
    plot_three_levels_count,
    save_table,
)
from utils.mora_analyzer import MoraAnalyzer
from utils.musicxml_analyzer import MusicXmlData


plt.rcParams["font.size"] = 20

mora_analyzer = MoraAnalyzer()

# ここから実行部分
data_folder = Path(Path(__file__).parent) / "data"
print("Data Folder: ", data_folder)
csv_folder = Path(data_folder) / "01exp_csv"
csv_folder.mkdir(exist_ok=True)
row_csv = csv_folder / "exp01_all_songs_row.csv" #生データ
normalized_csv = csv_folder / "normalized_lyrics.csv" #正規化だけしたデータ
valid_csv = csv_folder/ "all_mora_analysis.csv" #分析対象データ
invalid_csv = csv_folder / "invalid_mora.csv"

if Path(row_csv).is_file():
    print("exp01_all_songs_row.csv exists")
    df_row = pd.read_csv(row_csv)
else:
    print("exp01_all_songs_row.csv doesn't exist")
    folder = r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文\東北きりたん歌唱データベース\kiritan_singing-master\musicxml"  # noqa: E501
    xml_data = MusicXmlData(str(folder))
    df_row = xml_data.exp01_load()
    print("=== exp01 df (with relative_pitch) ===")
    print(df_row.head())
    print(df_row.tail())
    print(df_row.columns)
    #とりあえずバックアップ
    df_row.to_csv(row_csv, index=False, encoding="utf-8-sig")

# 文字の正規化処理をすでにしているか確認
if Path(normalized_csv).is_file():
    print("normalized_lyrics.csv exists")
    df_valid = pd.read_csv(normalized_csv)
# 文字の正規化処理をしていない場合、ここで実行
else:
    print("normalized_lyrics.csv doesn't exist")
    df_row["lyric"] = df_row["lyric"].fillna("").astype(str).str.strip()
    lyrics = df_row["lyric"].tolist()
    norm_lyrics: list[str] = []
    for i ,mora in enumerate(lyrics):
        add_mora = mora
        if "う゛" in mora:
            add_mora = mora.replace("う゛", "ヴ")
        elif "ゔ" in mora:
            add_mora = mora.replace("ゔ", "ヴ")
        elif mora == "を":
            add_mora = "お"

        if mora =="ー":
            try:
                prev_mora = norm_lyrics[i-1]
                prev_vowel = mora_analyzer.vowel_of_mora(prev_mora)
                add_mora = prev_vowel
            except(IndexError, ValueError) as e:
                print(f"Chouon fallback: {e}")
                add_mora = "ー"
        norm_lyrics.append(add_mora)
    lyrics = norm_lyrics

    #モーラを正規化したリストで上書き
    df_row["lyric"] = lyrics

    # 1) 有効行だけ残すmaskを作る(空白とnotna(Nan)を除外)、有効行だけ抽出
    mask = df_row["lyric"].notna() & df_row["lyric"].str.strip().ne("")
    df_valid = df_row.loc[mask, ["song", "pitch", "relative_pitch", "lyric"]].copy()
    df_invalid = df_row.loc[~mask, ["song", "pitch", "relative_pitch", "lyric"]].copy()
    # --- 前後空白を落としておくと後段が安定 ---
    df_valid["lyric"] = df_valid["lyric"].str.strip()
    df_invalid["lyric"] = df_invalid["lyric"].str.strip()
    df_valid.to_csv(normalized_csv, index=False, encoding="utf-8-sig")

if Path(valid_csv).is_file():
    print("all_mora_analysis.csv exists")
    df_mora_analysis = pd.read_csv(valid_csv)
else:
    print("all_mora_analysis.csv doesn't exist")
    # 2) 歌詞を音素に変換
# all_mora は object 配列で
df_valid["lyric"] = df_valid["lyric"].fillna("").astype(str).str.strip()
all_mora = df_valid.to_numpy(object)

valid, excluded = [], []

for song, pitch, rel_pitch, lyric in all_mora:
    moras, mora_count = mora_analyzer.split_moras(lyric)  # ← lyric は文字列化済み
    single_mora = 1
    double_mora = 2

    if mora_count == single_mora:
        mora = moras[0]
        ph = mora_analyzer.get_phonemes(mora)                     # ★音素化
        consonant = mora_analyzer.get_consonant(ph)               # ★音素配列を渡す
        vowel     = mora_analyzer.get_vowel(ph, mode="first")     # ★同上
        special   = mora_analyzer.get_special_symbol(ph)
        valid.append([song, mora, pitch, rel_pitch, consonant, vowel, special])

    elif mora_count == double_mora:
        if moras[1] in second_mora_exclude:
            # 1 つ目
            ph1 = mora_analyzer.get_phonemes(moras[0])
            valid.append([
                song,
                moras[0],
                pitch,
                rel_pitch,
                mora_analyzer.get_consonant(ph1),
                mora_analyzer.get_vowel(ph1, mode="first"),
                mora_analyzer.get_special_symbol(ph1)
            ])
            # 2 つ目
            ph2 = mora_analyzer.get_phonemes(moras[1])
            valid.append([
                song,
                moras[1],
                pitch,
                rel_pitch,
                mora_analyzer.get_consonant(ph1),
                mora_analyzer.get_vowel(ph1, mode="first"),
                mora_analyzer.get_special_symbol(ph1)
            ])
        else:
            excluded.append([song, lyric, pitch, rel_pitch])

    else:  # count >= 3
        excluded.append([song, lyric, pitch, rel_pitch])

fig_folder = Path(data_folder) / "01exp_figures"
fig_folder.mkdir(exist_ok=True)

df_mora_analysis = pd.DataFrame(valid, columns=["song", "lyric", "pitch", "relative_pitch", "consonant", "vowel", "special"])  # noqa: E501
df_mora_analysis.to_csv(valid_csv, index=False, encoding="utf-8-sig")
df_excluded = pd.DataFrame(excluded, columns=["song", "lyric", "pitch", "rel_pitch"])
df_excluded.to_csv(invalid_csv, index=False, encoding="utf-8-sig")

# ======================================================================================
# 3) 相対音高を取得して、音程分布のヒストグラム描画
rel_pitch = df_mora_analysis["relative_pitch"].astype(int)

bins = np.arange(rel_pitch.min() - 0.5, rel_pitch.max() + 1.5, 1).tolist()
plt.figure(figsize=(10, 8))
plt.hist(rel_pitch, bins=bins, edgecolor="white", color="0.2")
plt.xlabel("Normalized Pitch (MIDI Note Number - median)")
plt.ylabel("Count")
plt.title("Relative Pitch Count")
plt.xticks(range(int(rel_pitch.min()), int(rel_pitch.max() + 1)),  rotation=45, fontsize=14)
plt.tight_layout()
histgram_fig_path = fig_folder /  "relative_pitch_count.png"
plt.savefig(histgram_fig_path, bbox_inches="tight")
plt.close()


High = []
Mid = []
Low = []

hi_count = mid_count = lo_count = 0

for song, mora, pitch, rel_pitch, consonant, vowel, special in df_mora_analysis.to_numpy(object):
    if rel_pitch > 1:
        High.append([song, mora, pitch, rel_pitch, consonant, vowel, special])
    elif rel_pitch < -1:
        Low.append([song, mora, pitch, rel_pitch, consonant, vowel, special])
    else:
        Mid.append([song, mora, pitch, rel_pitch, consonant, vowel, special])

print(f"High Count: {len(High)}, Mid Count: {len(Mid)}, Low Count: {len(Low)}")

# ======================================================================================

df_high = pd.DataFrame(High, columns=["song", "lyric", "pitch", "relative_pitch", "consonant", "vowel", "special"])
df_mid = pd.DataFrame(Mid, columns=["song", "lyric", "pitch", "relative_pitch", "consonant", "vowel", "special"])
df_low = pd.DataFrame(Low, columns=["song", "lyric", "pitch", "relative_pitch", "consonant", "vowel", "special"])

save_table(df_mora_analysis, "all")
save_table(df_high, "High")
save_table(df_mid, "Mid")
save_table(df_low, "Low")

high_c, high_v, high_spec = count_phoneme_occurrence(High)
mid_c, mid_v, mid_spec = count_phoneme_occurrence(Mid)
low_c, low_v,  low_spec = count_phoneme_occurrence(Low)

base_title = "Count Distribution"

# --- vowel (母音) -------------------------------------------------------------------------------
vow_fig_dir = fig_folder / _vow
plot_three_levels_count(
    high=high_v,
    mid=mid_v,
    low=low_v,
    folder=vow_fig_dir,
    mode=_vow,
    base_title=base_title,
)

# --- consonant (子音) ---------------------------------------------------------------------------
con_fig_dir = fig_folder / _con
con_fig_dir.mkdir(exist_ok=True)

nas_cfg = ThreeLevelArtConfig(
    con_fig_dir=con_fig_dir,
    key_list=nasal,
    label=_nas,
    base_title=base_title,
    ylim=(0, 20),
)

plo_cfg = ThreeLevelArtConfig(
    con_fig_dir=con_fig_dir,
    key_list=plosive,
    label=_plo,
    base_title=base_title,
    ylim=(0, 20),
)

fri_cfg = ThreeLevelArtConfig(
    con_fig_dir=con_fig_dir,
    key_list=fricative,
    label=_fric,
    base_title=base_title,
    ylim=(0, 10),
)

tap_cfg = ThreeLevelArtConfig(
    con_fig_dir=con_fig_dir,
    key_list=tap,
    label=_tap,
    base_title=base_title,
    ylim=(0, 20),
)

app_cfg = ThreeLevelArtConfig(
    con_fig_dir=con_fig_dir,
    key_list=approximant,
    label=_app,
    base_title=base_title,
    ylim=(0, 10),
)

# - consonant (子音全体) -
plot_three_levels_count(
    high=high_c,
    mid=mid_c,
    low=low_c,
    folder=con_fig_dir,
    mode=_con,
    base_title=base_title,
)

# -- nasal (鼻音) --
plot_three_levels_art(
    high=high_c,
    mid=mid_c,
    low=low_c,
    cfg=nas_cfg,
)

# -- plosive (破裂音) --
plot_three_levels_art(
    high=high_c,
    mid=mid_c,
    low=low_c,
    cfg=plo_cfg,
)

# -- fricative (摩擦音) --
plot_three_levels_art(
    high=high_c,
    mid=mid_c,
    low=low_c,
    cfg=fri_cfg
)

# -- tap (弾き音) --
plot_three_levels_art(
    high=high_c,
    mid=mid_c,
    low=low_c,
    cfg=tap_cfg,
)

# -- approximant (接近音) --
plot_three_levels_art(
    high=high_c,
    mid=mid_c,
    low=low_c,
    cfg=app_cfg,
)

# --- special symbols (撥音・促音) ---
SS_fig_dir = fig_folder / _ss
SS_fig_dir.mkdir(exist_ok=True)

plot_three_levels_count(
    high=high_spec,
    mid=mid_spec,
    low=low_spec,
    folder=SS_fig_dir,
    mode=_ss,
    base_title=base_title,
)
