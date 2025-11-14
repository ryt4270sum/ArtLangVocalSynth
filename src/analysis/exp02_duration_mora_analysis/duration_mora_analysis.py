"""Module for analyzing pitch and mora patterns in musical scores.

This module processes MusicXML files to analyze the relationship between pitch heights and mora durations.
"""

import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent.parent).resolve()))

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from music21 import converter, note, tempo

from utils.mora_analyzer import MoraAnalyzer
from utils.musicxml_analyzer import MusicXmlAnalyzer

plt.rcParams["font.size"] = 20


#xml_path = r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文\東北きりたん歌唱データベース\kiritan_singing-master\musicxml\01.xml"  # noqa: E501, ERA001

musicxml_analyzer = MusicXmlAnalyzer(pic_dir=None, strip_ties=True, include_grace=False)
mora_analyzer = MoraAnalyzer()


def extract_notes_from_xml(xml_path: str) -> list[dict[str, object]]:
    """1つの MusicXML ファイルから音符情報を抽出する.

    指定された MusicXML ファイルを解析し, 各音符の長さ(BPMと音符の長さから計算),
    歌詞テキストなどを収集する.

    Args:
        xml_path: MusicXML ファイルへのパス.

    Returns:
        各音符を表す辞書のリスト.
        各辞書には以下の情報を含む.
            - song: 曲名 (拡張子を除いたファイル名)
            - note_idx: 曲内での音符インデックス
            - dur: 音価 (4分音符を1とする)
            - BPM: BPM
            - lyric: 歌詞テキスト (存在しない場合は空文字)
    """
    path = Path(xml_path)
    song = path.stem
    score = converter.parse(xml_path)
    data = []
    note_idx = 0

    if musicxml_analyzer.strip_ties:
        score = score.stripTies(inPlace=False)

    for n in score.recurse().notes:
        if not isinstance(n, note.Note):
            continue
        if n.duration.isGrace or (n.duration.quarterLength or 0) == 0:
            continue

        mm = n.getContextByClass(tempo.MetronomeMark)
        bpm_value = mm.getQuarterBPM() if mm is not None else None
        bpm = int(bpm_value) if bpm_value is not None else None

        data.append({
            "song": song,
            "note_idx": note_idx,
            "dur": float(n.duration.quarterLength),
            "BPM": bpm,
            "lyric": (n.lyric or "").strip()
        })
        note_idx += 1

    return data

def aggregate_note_data(folder: str) -> pd.DataFrame:
    """指定フォルダ内の MusicXML ファイルをすべて解析し, 音符情報を集約する.

    指定されたフォルダ内の `.xml` ファイルを走査し,
    各ファイルから抽出した音符情報を連結して DataFrame として返す.

    Args:
        folder: MusicXML ファイルを含むフォルダのパス.

    Returns:
        すべてのファイルから抽出された音符データをまとめた pandas.DataFrame.
    """
    all_data = []
    for file in Path(folder).iterdir():
        if file.suffix.lower() == ".xml":
            xml_path = Path(folder) / file
            data = extract_notes_from_xml(str(xml_path))
            all_data += data

    return pd.DataFrame(all_data)

# ここから実行部分
data_folder = Path(Path(__file__).parent) / "data"
print(data_folder)
row_csv = Path(data_folder) / "exp02_all_songs_row.csv" #生データ
normalized_csv = Path(data_folder) / "normalized_lyrics.csv" #正規化だけしたデータ
valid_csv = Path(data_folder) / "all_mora_analysis.csv" #分析対象データ

if Path(row_csv).is_file():
    print("exp02_all_songs_row.csv exists")
    df = pd.read_csv(row_csv)
else:
    print("exp02_all_songs_row.csv doesn't exist")
    folder = r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文\東北きりたん歌唱データベース\kiritan_singing-master\musicxml"  # noqa: E501
    df = aggregate_note_data(folder=folder)
    #とりあえずバックアップ
    df.to_csv(row_csv, index=False, encoding="utf-8-sig")

# 文字の正規化処理をすでにしているか確認
if Path(normalized_csv).is_file():
    print("normalized_lyrics.csv exists")
    df_valid = pd.read_csv(normalized_csv)
# 文字の正規化処理をしていない場合、ここで実行
else:
    print("normalized_lyrics.csv doesn't exist")
    lyrics = df["lyric"].tolist()
    for i in range(1, len(lyrics)):
        mora = lyrics[i]
        if mora == "ー":
            prev_vowel = mora_analyzer.vowel_of_mora(lyrics[i-1])
            lyrics[i] = prev_vowel
        elif mora in {"う゛" , "ゔ"}:
            lyrics[i] = "ヴ"
        elif mora == "を":
            lyrics[i] = "お"

    #モーラを正規化したリストで上書き
    df["lyric"] = lyrics

    # 1) 有効行だけ残すmaskを作る(空白とnotna(Nan)を除外)、有効行だけ抽出
    mask = df["lyric"].notna() & df["lyric"].str.strip().ne("")
    df_valid = df.loc[mask, ["song", "lyric", "dur", "BPM"]].copy()
    # --- 前後空白を落としておくと後段が安定 ---
    df_valid["lyric"] = df_valid["lyric"].str.strip()
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

print(df_valid.columns)

valid, excluded = [], []
second_mora_exclude = {"っ", "ん", "あ", "い", "う", "え", "お", "ー"}

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

df_mora_analysis = pd.DataFrame(valid, columns=["song", "lyric","dur", "BPM", "consonant", "vowel", "special"])
df_mora_analysis.to_csv(valid_csv, index=False, encoding="utf-8-sig")

# 各ヒストグラムを描画

dur = df_mora_analysis["dur"]
bpm = df_mora_analysis["BPM"]
dur_msec =(60 * 1000 * dur / bpm).astype(int)

bin_width =5 # msec
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
b_histgram_fig_path = Path(data_folder) / "02exp_figures" /  "BPM_distribution.png"
plt.savefig(b_histgram_fig_path, bbox_inches="tight")
plt.close()
print("BPM (Mean): ", np.mean(unique_bpm))
print("BPM (Median): ", np.median(unique_bpm))

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


SuperLong = []
Long = []
Short = []

s_long_count = long_count = short_count = 0
dur_threshold =250

for song, lyric, dur, bpm, con, vow, spec in df_mora_analysis.to_numpy(object):
    duration_msec = int (1000 * 60 * dur / bpm ) #ms単位にそろえるために1000をかける
    if  duration_msec >= long_threshold:
        SuperLong.append([song, lyric, duration_msec, con, vow, spec])
    elif duration_msec >= dur_threshold:
        Long.append([song, lyric, duration_msec, con, vow, spec])
    else:
        Short.append([song, lyric, duration_msec, con, vow, spec])

print("SuperLong: ", len(SuperLong))
print("Long: ", len(Long))
print("Short: ", len(Short))
print("Total: ", len(SuperLong)+len(Long)+len(Short))

VOWELS = ["a", "i", "u", "e", "o"]
CONSONANT = [
    "k", "s", "t", "n", "h", "m", "y", "r", "w",
    "g", "z", "d", "b", "p", "ch", "j", "ts", "f", "sh",
    "ky", "gy", "ny", "hy", "by", "my", "py", "ry"
]
SPECIALS = ["cl", "N"]

def count_phoneme_durations(data: list) -> tuple[dict, dict, dict]:
    """音素データ中の子音, 母音, 特殊記号の総出現時間をカウントする.

    与えられたデータ (音声学的解析結果など) を走査し,
    各子音, 母音, 特殊記号の出現頻度を辞書形式で数える.

    Args:
        data: 音素情報のリスト.
            各要素は以下のような形式で構成される.
                [song, lyric, duration_msec, con, vow, spec]

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
    xlabel: str,
    title: str,
    folder: Path | None = None,
    mode: str = "default",
) -> None:
    """音素の出現回数をヒストグラムとして描画・保存する.

    与えられたカウント辞書をもとに, 子音, 母音, 特殊記号などの
    出現頻度を棒グラフで可視化し, PNG 画像として保存する.

    Args:
        count_dict: 音素とその出現回数を対応付けた辞書.
        xlabel: x軸ラベル (例: "Vowel", "Consonant").
        title: グラフのタイトルおよび保存ファイル名 (拡張子を除く).
        folder: 画像の保存先フォルダ (指定しない場合はスクリプトのあるフォルダ).
        mode: プロットモード ("default" または "consonant").
            "consonant" の場合は x軸ラベルを45°回転して描画する.
    """
    figure_folder = Path(__file__).parent if folder is None else folder

    keys = list(count_dict.keys())
    values = list(count_dict.values())
    length = sum(values)
    rate=[]
    for v in values:
        prob = int(v*100/length)
        rate.append(prob)

    if mode == "consonant":
        plt.figure(figsize=(8, 8))
        plt.xticks(rotation=45, fontsize=14)
        plt.bar(keys, rate, color="0.2")
        plt.xlabel(xlabel)
        plt.ylabel("Occurrence Rate [%]")
        plt.ylim([0,50])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figure_folder /  f"{title}.png")
        plt.close()
    else:
        plt.figure(figsize=(8, 8))
        plt.xlabel(xlabel)
        plt.bar(keys, rate, color="0.2")
        plt.xlabel(xlabel)
        plt.ylabel("Occurrence Rate [%]")
        plt.ylim([0,50])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figure_folder /  f"{title}.png")
        plt.close()

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

convert_csv(Long, folder=data_folder, filename="exp02_Long.csv")
convert_csv(SuperLong, folder=data_folder, filename="exp02_SuperLong.csv")
convert_csv(Short, folder=data_folder, filename="exp02_Short.csv")

slong_c, slong_v, slong_spec = count_phoneme_durations(SuperLong)
long_c, long_v, long_spec = count_phoneme_durations(Long)
short_c, short_v,  short_spec = count_phoneme_durations(Short)

_sl, _l, _s = "SuperLong", "Long", "Short"
_vow, _con, _ss = "Vowel", "Consonant", "Special Symbol"
base_title = "Count Distribution"
# --- vowel (母音) ---
vow_fig_dir = Path(data_folder) / "02exp_figures" / _vow
plot_count_histogram(slong_v, xlabel=_vow, title=f"{_sl} {_vow} {base_title}", folder=vow_fig_dir)
plot_count_histogram(long_v, xlabel=_vow, title=f"{_l} {_vow} {base_title}", folder=vow_fig_dir)
plot_count_histogram(short_v, xlabel=_vow, title=f"{_s} {_vow} {base_title}", folder=vow_fig_dir)

# --- consonant (子音) ---
con_fig_dir = Path(data_folder) / "02exp_figures" / _con
plot_count_histogram(slong_c, xlabel=_con, title=f"{_sl} {_con} {base_title}", folder=con_fig_dir, mode="consonant")
plot_count_histogram(long_c, xlabel=_con, title=f"{_l} {_con} {base_title}", folder=con_fig_dir, mode="consonant")
plot_count_histogram(short_c, xlabel=_con, title=f"{_s} {_con} {base_title}", folder=con_fig_dir, mode="consonant")

# --- special symbols (撥音・促音) ---
SS_fig_dir = Path(data_folder) / "02exp_figures" / _ss
plot_count_histogram(slong_spec, xlabel=_ss, title=f"{_sl} {_ss} {base_title}", folder=SS_fig_dir)
plot_count_histogram(long_spec, xlabel=_ss, title=f"{_l} {_ss} {base_title}", folder=SS_fig_dir)
plot_count_histogram(short_spec, xlabel=_ss, title=f"{_s} {_ss} {base_title}", folder=SS_fig_dir)

