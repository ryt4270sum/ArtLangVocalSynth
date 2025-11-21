"""Module for analyzing pitch and mora patterns in musical scores.

This module processes MusicXML files to analyze the relationship between pitch and mora occurrences.
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

    tied_score = score.stripTies(inPlace=False)

    _, pitches, _ = musicxml_analyzer.collect_pitches(xml_path)
    median_pitch = np.median(pitches)

    for n in tied_score.recurse().notes:
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

def merge_lyricless_notes(df: pd.DataFrame) -> pd.DataFrame:
    """歌詞が NaN / 空のノートを、同じ曲の直前ノートに dur だけ足して結合する.

    前のノートが存在しない(曲の先頭が歌詞なしなど)の行は捨てる。
    """
    df = df.sort_values(["song", "note_idx"]).reset_index(drop=True)
    print(df[0:5])

    merged_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        song = row["song"]
        note_idx = row["note_idx"]
        pitch = row["pitch"]
        relative_pitch = row["relative_pitch"]
        dur = row["dur"]
        lyric = row["lyric"]

        # 歌詞が空 or NaN なら前の行にマージ
        if (pd.isna(lyric) or str(lyric).strip() == ""):
            if merged_rows and merged_rows[-1]["song"] == song:
                # 同じ曲の直前ノートに dur だけ足す
                merged_rows[-1]["dur"] += dur
            else:
                # 曲頭が歌詞なし等 → モーラ分析には使わないので捨てる
                pass
        else:
            # 歌詞ありノートはそのまま追加
            merged_rows.append({
                "song": song,
                "note_idx": note_idx,
                "pitch": pitch,
                "relative_pitch": relative_pitch,
                "dur": dur,
                "lyric": str(lyric).strip(),
            })

    return pd.DataFrame(merged_rows)

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
print("Data Folder: ", data_folder)
row_csv = Path(data_folder) / "exp01_all_songs_row.csv" #生データ
normalized_csv = Path(data_folder) / "normalized_lyrics.csv" #正規化だけしたデータ
valid_csv = Path(data_folder) / "all_mora_analysis.csv" #分析対象データ
invalid_csv = Path(data_folder) / "invalid_mora.csv"

if Path(row_csv).is_file():
    print("exp01_all_songs_row.csv exists")
    df = pd.read_csv(row_csv)
else:
    print("exp01_all_songs_row.csv doesn't exist")
    folder = r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文\東北きりたん歌唱データベース\kiritan_singing-master\musicxml"  # noqa: E501
    df = aggregate_note_data(folder=folder)
    df_merged = merge_lyricless_notes(df=df)
    #とりあえずバックアップ
    df.to_csv(row_csv, index=False, encoding="utf-8-sig")

# 文字の正規化処理をすでにしているか確認
if Path(normalized_csv).is_file():
    print("normalized_lyrics.csv exists")
    df_valid = pd.read_csv(normalized_csv)
# 文字の正規化処理をしていない場合、ここで実行
else:
    print("normalized_lyrics.csv doesn't exist")
    df_merged["lyric"] = df_merged["lyric"].fillna("").astype(str).str.strip()
    lyrics = df_merged["lyric"].tolist()
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
    df_merged["lyric"] = lyrics

    # 1) 有効行だけ残すmaskを作る(空白とnotna(Nan)を除外)、有効行だけ抽出
    mask = df_merged["lyric"].notna() & df_merged["lyric"].str.strip().ne("")
    df_valid = df_merged.loc[mask, ["song", "pitch", "relative_pitch", "lyric"]].copy()
    df_invalid = df_merged.loc[~mask, ["song", "pitch", "relative_pitch", "lyric"]].copy()
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
# 相対音高のヒストグラムを調べる(音域ごとに偏りがないか確認するため)  # noqa: ERA001

valid_pitches = df_mora_analysis["relative_pitch"].astype(int)
bins = np.arange(valid_pitches.min() - 0.5, valid_pitches.max() + 1.5, 1).tolist()



# 3) ヒストグラム描画
plt.figure(figsize=(16, 8))
plt.hist(valid_pitches, bins=bins, edgecolor="white", color="0.2")
plt.xlabel("Normalized Pitch (MIDI Note Number - median)")
plt.ylabel("Count")
plt.title("Relative Pitch Count")
plt.xticks(range(int(valid_pitches.min()), int(valid_pitches.max() + 1)),  rotation=0, fontsize=16)
plt.tight_layout()
histgram_fig_path = fig_folder /  "relative_pitch_count.png"
plt.savefig(histgram_fig_path, bbox_inches="tight")
plt.close()


High = []
Mid = []
Low = []

for _, mora, _, rel_pitch, consonant, vowel, special in df_mora_analysis.to_numpy(object):
    if rel_pitch > 1:
        High.append([rel_pitch, mora, consonant, vowel, special])
    elif rel_pitch < -1:
        Low.append([rel_pitch, mora, consonant, vowel, special])
    else:
        Mid.append([rel_pitch, mora, consonant, vowel, special])

print("High: ", len(High))
print("Mid: ", len(Mid))
print("Low: ", len(Low))
print("Total: ", len(High)+len(Mid)+len(Low))

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
        mode: プロットモード ("default" または "consonant").
            "consonant" の場合は x軸ラベルを45°回転して描画する.
    """
    figure_folder = Path(__file__).parent if folder is None else folder

    keys = list(count_dict.keys())
    values = list(count_dict.values())
    total = sum(count_dict.values())
    rate = []
    for v in values:
        prob = round(v*100/total, 2)
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
        ylim: list[int, int] | None = None
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
        ylim = [0, 100]
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
        plt.figure(figsize=(12, 9))
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
        plt.figure(figsize=(10, 8))
        plt.xticks(rotation=45, fontsize=14)
        plt.xlabel("Consonant")
        plt.ylim(ylim)
        plt.bar(list(selected_rate.keys()), list(selected_rate.values()) , color="0.2")
        plt.ylabel("Occurrence Rate [%]")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(folder / f"{title}.png")
        plt.close()

high_c, high_v, high_spec = count_phoneme_occurrences(High)
mid_c, mid_v, mid_spec = count_phoneme_occurrences(Mid)
low_c, low_v,  low_spec = count_phoneme_occurrences(Low)

_hi, _mid, _low = "High", "Mid", "Low"
_vow, _con, _SS = "Vowel", "Consonant", "Special Symbol"
base_title = "Occurrence Distribution"
_nas, _plo, _fric, _tap, _app = "Nasal", "Plosive", "Fricative", "Tap", "Approximant"

# # --- vowel (母音) ---
vow_fig_dir = fig_folder / _vow
vow_fig_dir.mkdir(exist_ok=True)
plot_count_histogram(high_v, title=f"{_hi} {_vow} {base_title}", folder=vow_fig_dir, mode=_vow)
plot_count_histogram(mid_v,  title=f"{_mid} {_vow} {base_title}", folder=vow_fig_dir, mode=_vow)
plot_count_histogram(low_v, title=f"{_low} {_vow} {base_title}", folder=vow_fig_dir, mode=_vow)

# --- consonant (子音) ---
con_fig_dir = fig_folder / _con
con_fig_dir.mkdir(exist_ok=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# - consonant (子音全体) -
plot_count_histogram(high_c, title=f"{_hi} {_con} {base_title}", folder=con_fig_dir, mode=_con)
plot_count_histogram(mid_c,  title=f"{_mid} {_con} {base_title}", folder=con_fig_dir, mode=_con)
plot_count_histogram(low_c,  title=f"{_low} {_con} {base_title}", folder=con_fig_dir, mode=_con)

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# -- nasal (鼻音) --
nas_dir = con_fig_dir / _nas
nas_dir.mkdir(exist_ok=True)
plot_art_histgram(full_dict=high_c, key_list=nasal,
                   title=f"{_hi} {_nas} {base_title}", folder=nas_dir, ylim=[0,20])
plot_art_histgram(full_dict=mid_c, key_list=nasal,
                   title=f"{_mid} {_nas} {base_title}", folder=nas_dir, ylim=[0,20])
plot_art_histgram(full_dict=low_c, key_list=nasal,
                   title=f"{_low} {_nas} {base_title}", folder=nas_dir, ylim=[0,20])

# -- plosive (破裂音) --
plo_dir = con_fig_dir / _plo
plo_dir.mkdir(exist_ok=True)
plot_art_histgram(full_dict=high_c, key_list=plosive,
                   title=f"{_hi} {_plo} {base_title}", folder=plo_dir, ylim=[0,20])
plot_art_histgram(full_dict=mid_c, key_list=plosive,
                   title=f"{_mid} {_plo} {base_title}", folder=plo_dir, ylim=[0,20])
plot_art_histgram(full_dict=low_c, key_list=plosive,
                   title=f"{_low} {_plo} {base_title}", folder=plo_dir, ylim=[0,20])

# -- fricative (摩擦音, 破擦音を一時的にまとめたもの) --
fric_dir = con_fig_dir / _fric
fric_dir.mkdir(exist_ok=True)
plot_art_histgram(full_dict=high_c, key_list=fricative,
                  title=f"{_hi} {_fric} {base_title}", folder=fric_dir, ylim=[0,10])
plot_art_histgram(full_dict=mid_c, key_list=fricative,
                   title=f"{_mid} {_fric} {base_title}", folder=fric_dir, ylim=[0,10])
plot_art_histgram(full_dict=low_c, key_list=fricative,
                   title=f"{_low} {_fric} {base_title}", folder=fric_dir, ylim=[0,10])

# -- tap (弾き音) --
tap_dir = con_fig_dir / _tap
tap_dir.mkdir(exist_ok=True)
plot_art_histgram(full_dict=high_c, key_list=tap,
                   title=f"{_hi} {_tap} {base_title}", folder=tap_dir, ylim=[0,20])
plot_art_histgram(full_dict=mid_c, key_list=tap,
                   title=f"{_mid} {_tap} {base_title}", folder=tap_dir, ylim=[0,20])
plot_art_histgram(full_dict=low_c, key_list=tap,
                   title=f"{_low} {_tap} {base_title}", folder=tap_dir, ylim=[0,20])

# -- tap (弾き音) --
app_dir = con_fig_dir / _app
app_dir.mkdir(exist_ok=True)
plot_art_histgram(full_dict=high_c, key_list=approximant,
                   title=f"{_hi} {_app} {base_title}", folder=app_dir, ylim=[0,10])
plot_art_histgram(full_dict=mid_c, key_list=approximant,
                   title=f"{_mid} {_app} {base_title}", folder=app_dir, ylim=[0,10])
plot_art_histgram(full_dict=low_c, key_list=approximant,
                   title=f"{_low} {_app} {base_title}", folder=app_dir, ylim=[0,10])

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# --- special symbols (撥音・促音) ---
SS_fig_dir = fig_folder / _SS
SS_fig_dir.mkdir(exist_ok=True)
plot_count_histogram(high_spec, title=f"{_hi} {_SS} {base_title}", folder=SS_fig_dir, mode=_SS)
plot_count_histogram(mid_spec, title=f"{_mid} {_SS} {base_title}", folder=SS_fig_dir, mode=_SS)
plot_count_histogram(low_spec, title=f"{_low} {_SS} {base_title}", folder=SS_fig_dir, mode=_SS)

