"""Module for analyzing note duration and mora patterns in musical scores.

This module processes MusicXML files to analyze the relationship between note
duration and mora.
"""

# ---- stdlib ----
import sys
from pathlib import Path

# ---- path hack (before local imports) ----
sys.path.append(str((Path(__file__).parent.parent.parent).resolve()))

# ---- third-party ----
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- local ----
from utils.constants import (
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
from utils.exp03_plot import (
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
csv_folder = Path(data_folder) / "03exp_csv"
csv_folder.mkdir(exist_ok=True)
row_csv = csv_folder / "exp03_all_songs_row.csv" #生データ
normalized_csv = csv_folder / "normalized_lyrics.csv" #正規化だけしたデータ
valid_csv = csv_folder/ "all_mora_analysis.csv" #分析対象データ
invalid_csv = csv_folder / "invalid_mora.csv"

if Path(row_csv).is_file():
    print("exp03_all_songs_row.csv exists")
    df_row = pd.read_csv(row_csv)
else:
    print("exp03_all_songs_row.csv doesn't exist")
    folder = r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文\東北きりたん歌唱データベース\kiritan_singing-master\musicxml"  # noqa: E501
    xml_data = MusicXmlData(str(folder))
    df_row = xml_data.exp03_load()
    print("=== exp03 df (with relative_pitch) ===")
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
    for i, mora in enumerate(lyrics):
        add_mora = mora
        if "う゛" in mora:
            add_mora = mora.replace("う゛", "ヴ")
        elif "ゔ" in mora:
            add_mora = mora.replace("ゔ", "ヴ")
        elif mora == "を":
            add_mora = "お"

        if mora == "ー":
            try:
                prev_mora = norm_lyrics[i-1]
                prev_vowel = mora_analyzer.vowel_of_mora(prev_mora)
                add_mora = prev_vowel
            except (IndexError, ValueError) as e:
                print(f"Chouon fallback: {e}")
                add_mora = "ー"

        norm_lyrics.append(add_mora)
    lyrics = norm_lyrics

    #モーラを正規化したリストで上書き
    df_row["lyric"] = lyrics

    # 1) 有効行だけ残すmaskを作る(空白とnotna(Nan)を除外)、有効行だけ抽出
    mask = df_row["lyric"].notna() & df_row["lyric"].str.strip().ne("")
    df_valid = df_row.loc[mask, ["song", "event_idx", "lyric", "interval_pitch"]].copy()
    df_invalid = df_row.loc[~mask, ["song", "event_idx", "lyric", "interval_pitch"]].copy()
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

for song, event_idx, lyric, interval_pitch in all_mora:
    moras, mora_count = mora_analyzer.split_moras(lyric)  # ← lyric は文字列化済み
    single_mora = 1
    double_mora = 2

    if mora_count == single_mora:
        mora = moras[0]
        ph = mora_analyzer.get_phonemes(mora)                     # 音素化
        consonant = mora_analyzer.get_consonant(ph)               # 音素配列を渡す
        vowel     = mora_analyzer.get_vowel(ph, mode="first")     # 同上
        special   = mora_analyzer.get_special_symbol(ph)
        valid.append([song, event_idx, mora, interval_pitch, consonant, vowel, special])

    elif mora_count == double_mora:
        if moras[1] in second_mora_exclude:
            # 1 つ目
            ph1 = mora_analyzer.get_phonemes(moras[0])
            valid.append([
                song,
                event_idx,
                moras[0],
                interval_pitch,
                mora_analyzer.get_consonant(ph1),
                mora_analyzer.get_vowel(ph1, mode="first"),
                mora_analyzer.get_special_symbol(ph1)
            ])
            # 2 つ目
            ph2 = mora_analyzer.get_phonemes(moras[1])
            valid.append([
                song,
                event_idx,
                moras[1],
                None,
                mora_analyzer.get_consonant(ph2),
                mora_analyzer.get_vowel(ph2, mode="first"),
                mora_analyzer.get_special_symbol(ph2)
            ])
        else:
            excluded.append([song, event_idx,lyric, interval_pitch])

    else:  # count >= 3
        excluded.append([song, event_idx,lyric, interval_pitch])
# 保存先フォルダ作成
fig_folder = Path(data_folder) / "03exp_figures"
fig_folder.mkdir(exist_ok=True)

df_mora_analysis = pd.DataFrame(valid, columns=["song", "event_idx", "lyric", "interval_pitch", "consonant", "vowel", "special"])  # noqa: E501
df_mora_analysis.to_csv(valid_csv, index=False, encoding="utf-8-sig")
df_excluded = pd.DataFrame(excluded, columns=["song", "event_idx", "lyric", "interval_pitch"])
df_excluded.to_csv(invalid_csv, index=False, encoding="utf-8-sig")

# ======================================================================================
# 3) 音程を取得して、音程分布のヒストグラム描画
interval = []
for interval_pitch in df_mora_analysis["interval_pitch"]:
    if np.isnan(interval_pitch):
        continue
    else:
        interval.append(interval_pitch)

interval_min = min(interval)
interval_max = max(interval)

bins = np.arange(interval_min - 0.5, interval_max + 1.5, 1).tolist()
plt.figure(figsize=(10, 8))
plt.hist(interval, bins=bins, edgecolor="white", color="0.2")
plt.xlabel("Interval Pitch (semitone)")
plt.ylabel("Count")
plt.title("Interval Pitch Count")
plt.xticks(range(int(interval_min), int(interval_max + 1)),  rotation=45, fontsize=14)
plt.tight_layout()
histgram_fig_path = fig_folder /  "interval_pitch_count.png"
plt.savefig(histgram_fig_path, bbox_inches="tight")
plt.close()

# ======================================================================================
# 4) 音程変化ごとにデータを分類

up = []
same = []
down = []

for song, event_idx, lyric, interval, con, vow, spec in df_mora_analysis.to_numpy(object):

    if not np.isnan(interval):
        if  interval >= 1.0:
            up.append([song, event_idx, lyric, interval, con, vow, spec])
        elif interval == 0.0:
            same.append([song, event_idx, lyric, interval, con, vow, spec])
        else:
            down.append([song, event_idx, lyric, interval, con, vow, spec])
    else:
        continue

print("up: ", len(up), "same: ", len(same), "down: ", len(down))
# ======================================================================================

df_up = pd.DataFrame(up, columns=["song", "event_idx", "lyric", "interval_pitch", "consonant", "vowel", "special"])
df_same = pd.DataFrame(same, columns=["song", "event_idx", "lyric", "interval_pitch", "consonant", "vowel", "special"])
df_down = pd.DataFrame(down, columns=["song", "event_idx", "lyric", "interval_pitch", "consonant", "vowel", "special"])

save_table(df_mora_analysis, "all")
save_table(df_up, "Up")
save_table(df_same, "Same")
save_table(df_down, "Down")

up_c, up_v, up_spec = count_phoneme_occurrence(up)
same_c, same_v, same_spec = count_phoneme_occurrence(same)
down_c, down_v, down_spec = count_phoneme_occurrence(down)

base_title = "Count Distribution"

# # --- vowel (母音) ---
vow_fig_dir = fig_folder / _vow
plot_three_levels_count(
    up=up_v,
    same=same_v,
    down=down_v,
    folder=vow_fig_dir,
    mode=_vow,
    base_title=base_title,
)

# --- consonant (子音) ---
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
    up=up_c,
    same=same_c,
    down=down_c,
    folder=con_fig_dir,
    mode=_con,
    base_title=base_title,
)

# -- nasal (鼻音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    cfg=nas_cfg
)

# -- plosive (破裂音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    cfg=plo_cfg
)

# -- fricative (摩擦音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    cfg=fri_cfg
)

# -- tap (弾き音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    cfg=tap_cfg
)

# -- approximant (接近音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    cfg=app_cfg
)

# --- special symbols (撥音・促音) ---
SS_fig_dir = fig_folder / _ss
SS_fig_dir.mkdir(exist_ok=True)

plot_three_levels_count(
    up=up_spec,
    same=same_spec,
    down=down_spec,
    folder=SS_fig_dir,
    mode=_ss,
    base_title=base_title,
)
