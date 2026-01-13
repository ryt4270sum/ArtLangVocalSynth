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

from utils.mora_analyzer import MoraAnalyzer
from utils.musicxml_analyzer import MusicXmlData

plt.rcParams["font.size"] = 20

mora_analyzer = MoraAnalyzer()

# ここから実行部分
data_folder = Path(Path(__file__).parent) / "data"
print("Data Folder: ", data_folder)
row_csv = Path(data_folder) / "exp03_all_songs_row.csv" #生データ
normalized_csv = Path(data_folder) / "normalized_lyrics.csv" #正規化だけしたデータ
valid_csv = Path(data_folder) / "all_mora_analysis.csv" #分析対象データ
invalid_csv = Path(data_folder) / "invalid_mora.csv"

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
second_mora_exclude = {"っ", "ん", "あ", "い", "う", "え", "お", "ー"}

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

fig_folder = Path(data_folder) / "02exp_figures"
fig_folder.mkdir(exist_ok=True)

df_mora_analysis = pd.DataFrame(valid, columns=["song", "event_idx", "lyric", "interval_pitch", "consonant", "vowel", "special"])
df_mora_analysis.to_csv(valid_csv, index=False, encoding="utf-8-sig")
df_excluded = pd.DataFrame(excluded, columns=["song", "event_idx", "lyric", "interval_pitch"])
df_excluded.to_csv(invalid_csv, index=False, encoding="utf-8-sig")


#print("=== df_mora_analysis ===")
#print(df_mora_analysis.iloc[0:50])
# --- ここから分析部分 ---
# 音程を取得
interval = []
for interval_pitch in df_mora_analysis["interval_pitch"]:
    if np.isnan(interval_pitch):
        continue
    else:
        interval.append(interval_pitch)

interval_min = min(interval)
interval_max = max(interval)
#print("interval_pitch", interval)

bins = np.arange(interval_min - 0.5, interval_max + 1.5, 1).tolist()

# 3) ヒストグラム描画
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

up = []
same = []
down = []

up_count = same_count = down_count = 0

for song, _, lyric, interval, con, vow, spec in df_mora_analysis.to_numpy(object):

    if not np.isnan(interval):
        if  interval >= 1.0:
            up.append([song, lyric, interval, con, vow, spec])
            up_count += 1
        elif interval == 0.0:
            same.append([song, lyric, interval, con, vow, spec])
            same_count += 1
        else:
            down.append([song, lyric, interval, con, vow, spec])
            down_count += 1
    else:
        continue

print("up: ", up_count, "same: ", same_count, "down: ", down_count)

VOWELS = ["a", "i", "u", "e", "o"]
CONSONANT = [
    "k", "s", "t", "n", "h", "m", "y", "r", "w",
    "g", "z", "d", "b", "p", "ch", "j", "ts", "f", "sh",
    "ky", "gy", "ny", "hy", "by", "my", "py", "ry", "dy", "ty", "v"
]

nasal = ["n", "m", "ny", "my"] # nasal >> 鼻音(マ行・ナ行・鼻濁音のガ(カ゜)行)
plosive = ["b", "p", "d", "t", "g", "k", "by", "py", "dy", "ty", "gy", "ky"] # plosive >> 破裂音(カ行・ガ行・タテト・ダデド・パ行・バ行)  # noqa: E501
#affricate = ["ch", "j", "ts"] # affricate >> 破擦音(チ・ツ・語頭のザ行(ヂズも))  # noqa: ERA001
fricative = ["s", "sh", "z", "f", "h", "hy", "v", "ch", "j", "ts"] # fricative >> 摩擦音(サ行・ハ行・語中のザ行(ヂズも))
tap = ["r", "ry"] # tap >> 弾き音(ラ行)# nasal >> 鼻音(マ行・ナ行・鼻濁音のガ(カ゜)行)
approximant = ["y", "w"] # approximant >> 接近音(ヤ行、ワ行)

SPECIALS = ["cl", "N"]

def save_table(df_mora_analysis: pd.DataFrame, name: str) -> None:
    """子音-母音の組み合わせ表を保存する.

    与えられたデータを走査し，子音と母音の組み合わせの出現回数をカウントし，
    その結果を表形式で表示する.

    Args:
        df_mora_analysis: 音素情報を含むデータフレーム.
        name: 保存するファイル名のベース部分.
    Returns:
        なし (表をコンソールに出力し、csvにして保存).
    """

    vowel_order = ["a", "i", "u", "e", "o"]
    cons_order = ["#", "b", "by", "ch", "d", "dy", "f", "g", "gy", "h", "hy", "j", "k", "ky", "m", "my", "n", "ny", "p",
               "py", "r", "ry", "s", "sh", "t", "ts", "ty", "v", "w", "y", "z", "cl", "N"]
    table = pd.DataFrame(0, index=cons_order, columns=vowel_order)
    for row in df_mora_analysis.to_numpy(object):
        con, vow, spec = row[4], row[5], row[6]
        if spec == "cl":
            table.loc["cl", "u"] += 1
        elif spec == "N":
            table.loc["N", "u"] += 1
        else:
            if con not in cons_order:
                con = "#"
            table.loc[con, vow] += 1
    print("Consonant-Vowel Table:")
    print(table)
    filename = f"{name} consonant_vowel_table.csv"
    save_path = fig_folder / filename
    table.to_csv(save_path, encoding="utf-8-sig")

def count_phoneme_occurrence(data: list) -> tuple[dict, dict, dict]:
    """音素データ中の子音, 母音, 特殊記号の総出現時間をカウントする.

    与えられたデータ (音声学的解析結果など) を走査し,
    各子音, 母音, 特殊記号の出現頻度を辞書形式で数える.

    Args:
        data: 音素情報のリスト.
            各要素は以下のような形式で構成される.
                [song, lyric, interval, con, vow, spec]

    Returns:
        3つの辞書からなるタプル.
            - consonant_counts: 子音ごとの総出現時間
            - vowel_counts: 母音ごとの総出現時間
            - special_counts: 特殊記号 (撥音, 促音など) の総出現時間
    """
    # カウント辞書初期化
    v_durs = dict.fromkeys(VOWELS, 0)
    c_durs = dict.fromkeys(CONSONANT, 0)
    spec_durs = dict.fromkeys(SPECIALS, 0)

    # 各行を走査してカウント
    for row in data:
        con, vow, spec = row[3], row[4], row[5]
        others_c = others_v = others_spec = 0
        if con in c_durs:
            c_durs[con] += 1
        else:
            others_c += 1
        if vow in v_durs:
            v_durs[vow] += 1
        else:
            others_v += 1
        if spec in spec_durs:
            spec_durs[spec] += 1
        else:
            others_spec += 1

    #print(f"Others - Consonant: {others_c}, Vowel: {others_v}, Special: {others_spec}")  # noqa: ERA001
    return c_durs, v_durs, spec_durs

def plot_count_histogram(
    count_dict: dict[str, float],
    title: str,
    mode: str,
    folder: Path | None = None
) -> None:
    """音素の出現回数をヒストグラムとして描画・保存する.

    与えられたカウント辞書をもとに, 子音, 母音, 特殊記号などの
    出現頻度を棒グラフで可視化し, PNG 画像として保存する.

    Args:
        count_dict: 音素とその出現回数を対応付けた辞書.
        xlabel: x軸ラベル (例: "Vowel", "Consonant").
        title: グラフのタイトルおよび保存ファイル名 (拡張子を除く).
        folder: 画像の保存先フォルダ (指定しない場合はスクリプトのあるフォルダ).
        mode: プロットモード (Vowel, Consonant, Special Symbol).
    """
    figure_folder = Path(__file__).parent if folder is None else folder

    keys = list(count_dict.keys())
    values = list(count_dict.values())
    length = sum(values)
    rate=[]
    for v in values:
        prob = int(v*100/length)
        rate.append(prob)

    mode_settings: dict[str, dict[str, any]] = {
        "Consonant": {
            "figsize": (12, 8),
            "xlabel": "Consonant",
            "ylim": (0, 30),
        },
        "Vowel": {
            "figsize": (8, 8),
            "xlabel": "Vowel",
            "ylim": (0, 40),
        },
        "Special Symbol": {
            "figsize": (8, 8),
            "xlabel": "Special Symbol",
            "ylim": (0, 100),
        }
    }

    if mode not in mode_settings:
        print("適切な mode を選んでください")
    else:
        cfg = mode_settings[mode]

        plt.figure(figsize=cfg["figsize"])
        plt.bar(keys, rate, color="0.2")
        plt.xlabel(cfg["xlabel"])
        plt.ylabel("Occurrence Rate [%]")
        plt.ylim(cfg["ylim"])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figure_folder / f"{title}.png")
        plt.close()

def  plot_art_histgram(
        full_dict: dict[str, int],
        key_list: list[str],
        title: str,
        folder: Path,
        ylim: tuple[int, int] | None = None
        ) -> None:
    """full_count_dict の中から target_list で指定した音素のみを抽出し、全体に対する割合 [%] を計算してプロットする関数.

    この関数は full_count_dict に含まれる音素の出現数をもとに、
    対象の音素だけを取り出して割合を算出し、棒グラフとして保存します。

    Parameters:
    ----------
    full_count_dict : dict[str, int]
        全音素の出現回数を保持する辞書。
        例:{"k": 1200, "s": 900, ...}

    target_list : list[str]
        抽出したい音素のリスト。
        例:["k", "s", "t"]

    title : str
        プロットにつけるタイトル。

    folder : Path
        画像を保存するフォルダのパス。

    Returns:
    -------
    None
        プロット画像を保存するだけで、計算結果は返さない。
    """
    if ylim is None:
        ylim = (0, 100)
    total = sum(full_dict.values())
    #print("Total Mora Count: ", total)  # noqa: ERA001
    selected: dict[str, int] = {}
    # ① 抽出 (select_articulation と同じ役割)
    for key, value in full_dict.items():
        if key in key_list:
            selected[key] = value
        else:
            continue
    selected_rate: dict[str, float] = {}
    for key, value in selected.items():
        rate = round(value*100/total, 2)
        selected_rate[key] = rate
    key_num = len(key_list)
    thresh_key_num = 5

    if key_num > thresh_key_num:
        plt.figure(figsize=(8, 10))
        plt.xticks(rotation=45, fontsize=14)
        plt.xlabel("Consonant")
        plt.ylim(ylim)
        plt.bar(list(selected_rate.keys()), list(selected_rate.values()) , color="0.2")
        plt.ylabel("Occurrence Rate [%]")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(folder / f"{title}.png")
        plt.close()
    else:
        plt.figure(figsize=(8, 10))
        plt.xticks(rotation=45, fontsize=14)
        plt.xlabel("Consonant")
        plt.ylim(ylim)
        plt.bar(list(selected_rate.keys()), list(selected_rate.values()) , color="0.2")
        plt.ylabel("Occurrence Rate [%]")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(folder / f"{title}.png")
        plt.close()

def plot_three_levels_count(
    up: dict[str, int],
    same: dict[str, int],
    down: dict[str, int],
    folder: Path,
    mode: str,
    base_title: str,
) -> None:
    """slong/long/short の3種類について count ヒストグラムをまとめて描画."""
    folder.mkdir(exist_ok=True)
    plot_count_histogram(up, title=f"{_sl} {mode} {base_title}", folder=folder, mode=mode)
    plot_count_histogram(same,  title=f"{_l} {mode} {base_title}", folder=folder, mode=mode)
    plot_count_histogram(down,  title=f"{_s} {mode} {base_title}", folder=folder, mode=mode)


def plot_three_levels_art(
    up: dict[str, int],
    same: dict[str, int],
    down: dict[str, int],
    con_fig_dir: Path,
    key_list: list[str],
    label: str,          # 例: _nas, _plo など
    base_title: str,
    ylim: tuple[int, int],
) -> None:
    """slong/long/short の3種類について articulation ヒストグラムをまとめて描画."""
    sub_dir = con_fig_dir / label
    sub_dir.mkdir(exist_ok=True)

    plot_art_histgram(
        full_dict=up,
        key_list=key_list,
        title=f"{_sl} {label} {base_title}",
        folder=sub_dir,
        ylim=ylim,
    )
    plot_art_histgram(
        full_dict=same,
        key_list=key_list,
        title=f"{_l} {label} {base_title}",
        folder=sub_dir,
        ylim=ylim,
    )
    plot_art_histgram(
        full_dict=down,
        key_list=key_list,
        title=f"{_s} {label} {base_title}",
        folder=sub_dir,
        ylim=ylim,
)


def convert_csv(data:list, folder:Path, filename:str) -> None:
    """リスト(または辞書のリスト)をDataFrameに変換し、CSVとして保存する関数。.

    Parameters
    ----------
    data : list
        リスト、または辞書のリスト(例:[{"a":1, "b":2}, {"a":3, "b":4}])
    folder : Path
        CSVを保存するフォルダのパス(Path形式)
    filename : str
        保存するCSVファイル名(例:"output.csv")
    """
    # DataFrameに変換
    df = pd.DataFrame(data)
    filepath = folder / filename
    # CSVとして保存(BOM付きUTF-8でExcelでも文字化けしにくい)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    filepath = folder / filename
    print(f"✅ CSVファイルを保存しました: {filepath}")

up_c, up_v, up_spec = count_phoneme_occurrence(up)
same_c, same_v, same_spec = count_phoneme_occurrence(same)
down_c, down_v, down_spec = count_phoneme_occurrence(down)

_sl, _l, _s = "Up", "Same", "Down"
_vow, _con, _ss = "Vowel", "Consonant", "Special Symbol"
base_title = "Count Distribution"
_nas, _plo, _fric, _tap, _app = "Nasal", "Plosive", "Fricative", "Tap", "Approximant"

# # --- vowel (母音) ---
vow_fig_dir = Path(data_folder) / "02exp_figures" / _vow
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
    con_fig_dir=con_fig_dir,
    key_list=nasal,
    label=_nas,
    base_title=base_title,
    ylim=(0, 20),
)

# -- plosive (破裂音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    con_fig_dir=con_fig_dir,
    key_list=plosive,
    label=_plo,
    base_title=base_title,
    ylim=(0, 20),
)

# -- fricative (摩擦音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    con_fig_dir=con_fig_dir,
    key_list=fricative,
    label=_fric,
    base_title=base_title,
    ylim=(0, 10),
)

# -- tap (弾き音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
    con_fig_dir=con_fig_dir,
    key_list=tap,
    label=_tap,
    base_title=base_title,
    ylim=(0, 20),
)

# -- approximant (接近音) --
plot_three_levels_art(
    up=up_c,
    same=same_c,
    down=down_c,
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
    up=up_spec,
    same=same_spec,
    down=down_spec,
    folder=SS_fig_dir,
    mode=_ss,
    base_title=base_title,
)