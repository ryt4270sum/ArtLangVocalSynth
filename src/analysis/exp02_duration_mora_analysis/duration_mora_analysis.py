"""Module for analyzing note duration and mora patterns in musical scores.

This module processes MusicXML files to analyze the relationship between note duration and mora.
"""

import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent.parent).resolve()))

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from utils.exp02_plot import (
    count_phoneme_durations,
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
row_csv = Path(data_folder) / "exp02_all_songs_row.csv" #生データ
normalized_csv = Path(data_folder) / "normalized_lyrics.csv" #正規化だけしたデータ
valid_csv = Path(data_folder) / "all_mora_analysis.csv" #分析対象データ
invalid_csv = Path(data_folder) / "invalid_mora.csv"

if Path(row_csv).is_file():
    print("exp02_all_songs_row.csv exists")
    df = pd.read_csv(row_csv)
else:
    print("exp02_all_songs_row.csv doesn't exist")
    folder = r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文\東北きりたん歌唱データベース\kiritan_singing-master\musicxml"  # noqa: E501
    xml_data = MusicXmlData(str(folder))
    df_row = xml_data.exp02_load()
    print("=== exp02 df (with relative_pitch) ===")
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
    df_valid = df_row.loc[mask, ["song", "lyric", "dur", "BPM"]].copy()
    df_invalid = df_row.loc[~mask, ["song", "lyric", "dur", "BPM"]].copy()
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

for song, lyric, dur, bpm in all_mora:
    moras, mora_count = mora_analyzer.split_moras(lyric)  # ← lyric は文字列化済み
    single_mora = 1
    double_mora = 2

    #↓文献引用欲しいよね[]
    n_weight = 0.5 #二文字目が「ん」の場合の時間分割の割合
    cl_weight = 0.5 #二文字目が「っ」の場合の時間分割の割合
    dvow_weight = 0.5 #二文字目が母音の場合の時間分割の割合

    if mora_count == single_mora:
        mora = moras[0]
        ph = mora_analyzer.get_phonemes(mora)                     # 音素化
        consonant = mora_analyzer.get_consonant(ph)               # 音素配列を渡す
        vowel     = mora_analyzer.get_vowel(ph, mode="first")     # 同上
        special   = mora_analyzer.get_special_symbol(ph)
        valid.append([song, mora, dur, bpm, consonant, vowel, special])

    elif mora_count == double_mora:
        if moras[1] == "っ":
            # 1 つ目
            ph1 = mora_analyzer.get_phonemes(moras[0])
            valid.append([
                song,
                moras[0],
                dur*(1-cl_weight),
                bpm,
                mora_analyzer.get_consonant(ph1),
                mora_analyzer.get_vowel(ph1, mode="first"),
                mora_analyzer.get_special_symbol(ph1)
            ])
            # 2 つ目
            ph2 = mora_analyzer.get_phonemes(moras[1])
            valid.append([
                song,
                moras[1],
                dur*cl_weight,
                bpm,
                mora_analyzer.get_consonant(ph2),
                mora_analyzer.get_vowel(ph2, mode="first"),
                mora_analyzer.get_special_symbol(ph2)
            ])
        elif moras[1] == "ん":
            ph1 = mora_analyzer.get_phonemes(moras[0])
            valid.append([
                song,
                moras[0],
                dur*(1-n_weight),
                bpm,
                mora_analyzer.get_consonant(ph1),
                mora_analyzer.get_vowel(ph1, mode="first"),
                mora_analyzer.get_special_symbol(ph1)
            ])
            # 2 つ目
            ph2 = mora_analyzer.get_phonemes(moras[1])
            valid.append([
                song,
                moras[1],
                dur*n_weight,
                bpm,
                mora_analyzer.get_consonant(ph2),
                mora_analyzer.get_vowel(ph2, mode="first"),
                mora_analyzer.get_special_symbol(ph2)
            ])
        elif moras[1] in second_mora_exclude:
            ph1 = mora_analyzer.get_phonemes(moras[0])
            valid.append([
                song,
                moras[0],
                dur*(1-dvow_weight),
                bpm,
                mora_analyzer.get_consonant(ph1),
                mora_analyzer.get_vowel(ph1, mode="first"),
                mora_analyzer.get_special_symbol(ph1)
            ])
            # 2 つ目
            ph2 = mora_analyzer.get_phonemes(moras[1])
            valid.append([
                song,
                moras[1],
                dur*dvow_weight,
                bpm,
                mora_analyzer.get_consonant(ph2),
                mora_analyzer.get_vowel(ph2, mode="first"),
                mora_analyzer.get_special_symbol(ph2)
            ])
        else:
            excluded.append([song, lyric, dur, bpm])

    else:  # count >= 3
        excluded.append([song, lyric, dur, bpm])

fig_folder = Path(data_folder) / "02exp_figures"
fig_folder.mkdir(exist_ok=True)

df_mora_analysis = pd.DataFrame(valid, columns=["song", "lyric","dur", "BPM", "consonant", "vowel", "special"])
df_mora_analysis.to_csv(valid_csv, index=False, encoding="utf-8-sig")
df_excluded = pd.DataFrame(excluded, columns=["song", "lyric", "dur", "bpm"])
df_excluded.to_csv(invalid_csv, index=False, encoding="utf-8-sig")

# ======================================================================================
# 3) 持続時間を取得して、持続時間別のヒストグラム描画

dur = df_mora_analysis["dur"]
bpm = df_mora_analysis["BPM"]
dur_msec =(60 * 1000 * dur / bpm).astype(int)

bin_width =10 # msec
long_threshold = 800 # 超長い音符の場合(ms)、ロングトーンとして別扱いにする

# 音符ごとのBPMのヒストグラム描画
unique_bpm = df_mora_analysis.drop_duplicates(subset="song")["BPM"].copy()
plt.figure(figsize=(10, 6))
b_bin = np.arange(80, unique_bpm.max()+5, 5).tolist()
plt.hist(unique_bpm, bins=b_bin, edgecolor="white", color="0.2")
plt.xlabel("BPM")
plt.ylabel("Number of Songs")
plt.title("BPM Distribution per Song")
plt.tight_layout()
figures_02exp = Path(data_folder) / "02exp_figures"
figures_02exp.mkdir(exist_ok=True)
b_histgram_fig_path = figures_02exp /  "BPM_distribution.png"
plt.savefig(b_histgram_fig_path, bbox_inches="tight")
plt.close()

# 3) ヒストグラム描画
plt.figure(figsize=(10, 6))
n_bin = np.arange(0,dur.max(), 0.5).tolist()
plt.hist(dur, bins=n_bin, edgecolor="white", color="0.2")
plt.xlabel("Note Value (Quarter Note = 1.0)")
plt.ylabel("Count")
plt.title("Distribution of Note Durations (beat units)")
plt.tight_layout()
n_histgram_fig_path = Path(data_folder) / "02exp_figures" /  "NoteDuration_distribution.png"
plt.savefig(n_histgram_fig_path, bbox_inches="tight")
plt.close()

# 実時間のヒストグラム描画
bins = np.arange(0, long_threshold, bin_width).tolist()
plt.figure(figsize=(10, 6))
plt.hist(dur_msec, bins=bins, edgecolor="white", color="0.2")
plt.xlabel("Note Duration [ms]")
plt.ylabel("Count")
plt.title(f"Duration Histogram (bin width = {bin_width} ms)")
plt.tight_layout()
histgram_fig_path = Path(data_folder) / "02exp_figures" /  "NoteDuration_ms_distribution.png"
plt.savefig(histgram_fig_path, bbox_inches="tight")
plt.close()

# ======================================================================================
# 4) 持続時間ごとにデータを分類

superlong = []
long = []
short = []

s_long_count = long_count = short_count = 0
dur_threshold =300

for song, lyric, dur, bpm, con, vow, spec in df_mora_analysis.to_numpy(object):
    duration_msec = int (1000 * 60 * dur / bpm ) #ms単位にそろえるために1000をかける
    if  duration_msec >= long_threshold:
        superlong.append([song, lyric, duration_msec, con, vow, spec])
        s_long_count += 1
    elif duration_msec >= dur_threshold:
        long.append([song, lyric, duration_msec, con, vow, spec])
        long_count += 1
    else:
        short.append([song, lyric, duration_msec, con, vow, spec])
        short_count += 1

print(f"Super Long Count: {s_long_count}, Long Count: {long_count}, Short Count: {short_count}")
# ======================================================================================

df_slong = pd.DataFrame(superlong, columns=["song", "lyric", "dur_msec", "consonant", "vowel", "special"])
df_long = pd.DataFrame(long, columns=["song", "lyric", "dur_msec", "consonant", "vowel", "special"])
df_short = pd.DataFrame(short, columns=["song", "lyric", "dur_msec", "consonant", "vowel", "special"])

save_table(df_mora_analysis, "all")
save_table(df_slong, "Super Long")
save_table(df_long, "Long")
save_table(df_short, "Short")

slong_c, slong_v, slong_spec = count_phoneme_durations(superlong)
long_c, long_v, long_spec = count_phoneme_durations(long)
short_c, short_v,  short_spec = count_phoneme_durations(short)
base_title = "Count Distribution"

# # --- vowel (母音) ---
vow_fig_dir = Path(data_folder) / "02exp_figures" / _vow
plot_three_levels_count(
    slong=slong_v,
    long=long_v,
    short=short_v,
    folder=vow_fig_dir,
    mode=_vow,
    base_title=base_title,
)

# --- consonant (子音) ---
con_fig_dir = fig_folder / _con
con_fig_dir.mkdir(exist_ok=True)

# - consonant (子音全体) -
plot_three_levels_count(
    slong=slong_c,
    long=long_c,
    short=short_c,
    folder=con_fig_dir,
    mode=_con,
    base_title=base_title,
)

# -- nasal (鼻音) --
plot_three_levels_art(
    slong=slong_c,
    long=long_c,
    short=short_c,
    con_fig_dir=con_fig_dir,
    key_list=nasal,
    label=_nas,
    base_title=base_title,
    ylim=(0, 20),
)

# -- plosive (破裂音) --
plot_three_levels_art(
    slong=slong_c,
    long=long_c,
    short=short_c,
    con_fig_dir=con_fig_dir,
    key_list=plosive,
    label=_plo,
    base_title=base_title,
    ylim=(0, 20),
)

# -- fricative (摩擦音) --
plot_three_levels_art(
    slong=slong_c,
    long=long_c,
    short=short_c,
    con_fig_dir=con_fig_dir,
    key_list=fricative,
    label=_fric,
    base_title=base_title,
    ylim=(0, 10),
)

# -- tap (弾き音) --
plot_three_levels_art(
    slong=slong_c,
    long=long_c,
    short=short_c,
    con_fig_dir=con_fig_dir,
    key_list=tap,
    label=_tap,
    base_title=base_title,
    ylim=(0, 20),
)

# -- approximant (接近音) --
plot_three_levels_art(
    slong=slong_c,
    long=long_c,
    short=short_c,
    con_fig_dir=con_fig_dir,
    key_list=approximant,
    label=_app,
    base_title=base_title,
    ylim=(0, 10),
)

# --- special symbols (撥音・促音) ---
SS_fig_dir = fig_folder / _ss
SS_fig_dir.mkdir(exist_ok=True)

plot_three_levels_count(
    slong=slong_spec,
    long=long_spec,
    short=short_spec,
    folder=SS_fig_dir,
    mode=_ss,
    base_title=base_title,
)
