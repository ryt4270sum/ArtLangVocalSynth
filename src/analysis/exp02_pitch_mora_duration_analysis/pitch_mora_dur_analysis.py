"""Module for analyzing pitch and mora patterns in musical scores.

This module processes MusicXML files to analyze the relationship between pitch heights and mora occurrences.
"""

import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent.parent).resolve()))

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from music21 import converter, note

from utils.mora_analyzer import MoraAnalyzer
from utils.musicxml_analyzer import MusicXmlAnalyzer

plt.rcParams["font.size"] = 20


xml_path = r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文\東北きりたん歌唱データベース\kiritan_singing-master\musicxml\01.xml"  # noqa: E501

musicxml_analyzer = MusicXmlAnalyzer(pic_dir=None, strip_ties=True, include_grace=False)
mora_analyzer = MoraAnalyzer()


def extract_notes_from_xml(xml_path: str) -> list[dict[str, object]]:
    """1つの MusicXML ファイルから音符情報を抽出する.

    指定された MusicXML ファイルを解析し, 各音符の音高 (MIDI 値),
    音価, 中央値からの相対音高, 歌詞テキストなどを収集する.

    Args:
        xml_path: MusicXML ファイルへのパス.

    Returns:
        各音符を表す辞書のリスト.
        各辞書には以下の情報を含む.
            - song: 曲名 (拡張子を除いたファイル名)
            - note_idx: 曲内での音符インデックス
            - pitch: 音高 (MIDI 値, int)
            - relative_pitch: 曲全体の中央値からの相対音高
            - dur: 音価 (4分音符を1とする)
            - lyric: 歌詞テキスト (存在しない場合は空文字)
    """
    path = Path(xml_path)
    song = path.stem
    score = converter.parse(xml_path)
    data = []
    note_idx = 0

    if musicxml_analyzer.strip_ties:
        score = score.stripTies(inPlace=False)

    _, pitches, _ = musicxml_analyzer.collect_pitches(xml_path)
    median_pitch = np.median(pitches)

    for n in score.recurse().notes:
        if not isinstance(n, note.Note):
            continue
        if n.duration.isGrace or (n.duration.quarterLength or 0) == 0:
            continue

        data.append({
            "song": song,
            "note_idx": note_idx,
            "pitch": int(n.pitch.midi),
            "relative_pitch": int(n.pitch.midi) - median_pitch,
            "dur": float(n.duration.quarterLength),
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
row_csv = Path(data_folder) / "all_songs_row.csv" #生データ
normalized_csv = Path(data_folder) / "normalized_lyrics.csv" #正規化だけしたデータ
valid_csv = Path(data_folder) / "all_mora_analysis.csv" #分析対象データ

if Path(row_csv).is_file():
    print("all_songs_row.csv exists")
    df = pd.read_csv(row_csv)
else:
    print("all_songs_row.csv doesn't exist")
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
    df_valid = df.loc[mask, ["relative_pitch", "lyric"]].copy()
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

valid, excluded = [], []
second_mora_exclude = {"っ", "ん", "あ", "い", "う", "え", "お", "ー"}

for pitch, lyric in all_mora:
    moras, mora_count = mora_analyzer.split_moras(lyric)  # ← lyric は文字列化済み
    single_mora = 1
    double_mora = 2

    if mora_count == single_mora:
        mora = moras[0]
        ph = mora_analyzer.get_phonemes(mora)                     # ★音素化
        consonant = mora_analyzer.get_consonant(ph)               # ★音素配列を渡す
        vowel     = mora_analyzer.get_vowel(ph, mode="first")     # ★同上
        special   = mora_analyzer.get_special_symbol(ph
                                                     )
        valid.append([pitch, mora, consonant, vowel, special])

    elif mora_count == double_mora:
        if moras[1] in second_mora_exclude:
            # 1 つ目
            ph1 = mora_analyzer.get_phonemes(moras[0])
            valid.append([
                pitch, moras[0],
                mora_analyzer.get_consonant(ph1),
                mora_analyzer.get_vowel(ph1, mode="first"),
                mora_analyzer.get_special_symbol(ph1)
            ])
            # 2 つ目
            ph2 = mora_analyzer.get_phonemes(moras[1])
            valid.append([
                pitch, moras[1],
                mora_analyzer.get_consonant(ph2),
                mora_analyzer.get_vowel(ph2, mode="first"),
                mora_analyzer.get_special_symbol(ph2)
            ])
        else:
            excluded.append([pitch, lyric])

    else:  # count >= 3
        excluded.append([pitch, lyric])

df_mora_analysis = pd.DataFrame(valid, columns=["relative_pitch", "lyric", "consonant", "vowel", "special"])
df_mora_analysis.to_csv(valid_csv, index=False, encoding="utf-8-sig")

# 相対音高のヒストグラムを調べる(音域ごとに偏りがないか確認するため)  # noqa: ERA001

valid_pitches = df_mora_analysis["relative_pitch"].astype(int)
bins = np.arange(valid_pitches.min() - 0.5, valid_pitches.max() + 1.5, 1).tolist()

# 3) ヒストグラム描画
plt.figure(figsize=(16, 12))
plt.hist(valid_pitches, bins=bins, edgecolor="white")
plt.xlabel("Normalized Pitch (MIDI Note Number - median)")
plt.ylabel("Count")
plt.title("Relative pitch distribution")
plt.xticks(range(int(valid_pitches.min()), int(valid_pitches.max() + 1)),  rotation=0, fontsize=16)
plt.tight_layout()
histgram_fig_path = Path(data_folder) / "01exp_figures" /  "relative_pitch_distribution.png"
plt.savefig(histgram_fig_path, bbox_inches="tight")
plt.close()


High = []
Mid = []
Low = []

for pitch, mora, consonant, vowel, special in df_mora_analysis.to_numpy(object):
    if pitch > 0:
        High.append([pitch, mora, consonant, vowel, special])
    elif pitch == 0:
        Mid.append([pitch, mora, consonant, vowel, special])
    else:
        Low.append([pitch, mora, consonant, vowel, special])

print(f"High: {len(High)}, Mid: {len(Mid)}, Low: {len(Low)}")

VOWELS = ["a", "i", "u", "e", "o"]
CONSONANT = [
    "k", "s", "t", "n", "h", "m", "y", "r", "w",
    "g", "z", "d", "b", "p", "ch", "j", "ts", "f", "sh",
    "ky", "gy", "ny", "hy", "by", "py", "ry"
]
SPECIALS = ["cl", "N"]



def count_phoneme_occurrences(data: list) -> tuple[dict, dict, dict]:
    """音素データ中の子音, 母音, 特殊記号の出現回数をカウントする.

    与えられたデータ (音声学的解析結果など) を走査し,
    各子音, 母音, 特殊記号の出現頻度を辞書形式で数える.

    Args:
        data: 音素情報のリスト.
            各要素は以下のような形式で構成される.
                [pitch, mora, consonant, vowel, special_symbol]

    Returns:
        3つの辞書からなるタプル.
            - consonant_counts: 子音ごとの出現回数
            - vowel_counts: 母音ごとの出現回数
            - special_counts: 特殊記号 (撥音, 促音など) の出現回数
    """
    # カウント辞書初期化
    v_counts = dict.fromkeys(VOWELS, 0)
    c_counts = dict.fromkeys(CONSONANT, 0)
    spec_counts = dict.fromkeys(SPECIALS, 0)

    # 各行を走査してカウント
    for row in data:
        con, vow, spec = row[2], row[3], row[4]
        others_c, others_v, others_spec = 0, 0, 0
        if con in c_counts:
            c_counts[con] += 1
        else:
            others_c += 1
        if vow in v_counts:
            v_counts[vow] += 1
        else:
            others_v += 1
        if spec in spec_counts:
            spec_counts[spec] += 1
        else:
            others_spec += 1

    #print(f"Others - Consonant: {others_c}, Vowel: {others_v}, Special: {others_spec}")  # noqa: ERA001
    return c_counts, v_counts, spec_counts

def plot_count_histogram(
    count_dict: dict[str, int],
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

    if mode == "consonant":
        plt.figure(figsize=(8, 8))
        plt.bar(keys, values)
        plt.xticks(rotation=45, fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figure_folder /  f"{title}.png")
        plt.close()
    else:
        plt.figure(figsize=(8, 8))
        plt.bar(keys, values)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figure_folder /  f"{title}.png")
        plt.close()

high_c, high_v, high_spec = count_phoneme_occurrences(High)
mid_c, mid_v, mid_spec = count_phoneme_occurrences(Mid)
low_c, low_v,  low_spec = count_phoneme_occurrences(Low)

_hi, _mid, _low = "High", "Mid", "Low"
_vow, _con, _SS = "Vowel", "Consonant", "Special Symbol"
base_title = "Occurrence Distribution"
# --- vowel (母音) ---
vow_fig_dir = Path(data_folder) / "01exp_figures" / _vow
plot_count_histogram(high_v, xlabel=_vow, title=f"{_hi} {_vow} {base_title}", folder=vow_fig_dir)
plot_count_histogram(mid_v, xlabel=_vow, title=f"{_mid} {_vow} {base_title}", folder=vow_fig_dir)
plot_count_histogram(low_v, xlabel=_vow, title=f"{_low} {_vow} {base_title}", folder=vow_fig_dir)

# --- consonant (子音) ---
con_fig_dir = Path(data_folder) / "01exp_figures" / _con
plot_count_histogram(high_c, xlabel=_con, title=f"{_hi} {_con} {base_title}", folder=con_fig_dir, mode="consonant")
plot_count_histogram(mid_c, xlabel=_con, title=f"{_mid} {_con} {base_title}", folder=con_fig_dir, mode="consonant")
plot_count_histogram(low_c, xlabel=_con, title=f"{_low} {_con} {base_title}", folder=con_fig_dir, mode="consonant")

# --- special symbols (撥音・促音) ---
SS_fig_dir = Path(data_folder) / "01exp_figures" / _SS
plot_count_histogram(high_spec, xlabel=_SS, title=f"{_hi} {_SS} {base_title}", folder=SS_fig_dir)
plot_count_histogram(mid_spec, xlabel=_SS, title=f"{_mid} {_SS} {base_title}", folder=SS_fig_dir)
plot_count_histogram(low_spec, xlabel=_SS, title=f"{_low} {_SS} {base_title}", folder=SS_fig_dir)

