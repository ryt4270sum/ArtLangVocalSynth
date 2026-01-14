"""utils.exp02_plot: 実験2で使う関数とか."""

import sys
from dataclasses import dataclass
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
    CONS_ORDER,
    CONSONANTS,
    SPECIALS,
    VOWEL_ORDER,
    VOWELS,
)

data_folder = Path(__file__).parent.parent.parent / "src" / "analysis" / "exp02_duration_mora_analysis" / "data"
fol_02exp_fig = Path(data_folder) / "02exp_figures"
fol_02exp_fig.mkdir(exist_ok=True)
fol_02exp_csv = Path(data_folder) / "02exp_csv"
fol_02exp_csv.mkdir(exist_ok=True)
fol_02exp_heatmap = fol_02exp_fig / "heatmap"
fol_02exp_heatmap.mkdir(exist_ok=True)

_sl, _l, _s = "SuperLong", "Long", "Short"

def save_table(df_mora_analysis: pd.DataFrame, name: str) -> None:
    """子音-母音の組み合わせ表を保存する.

    与えられたデータを走査し, 子音と母音の組み合わせの出現回数をカウントし,
    その結果を表形式で表示する.

    Args:
        df_mora_analysis: 音素情報を含むデータフレーム.
        name: 保存するファイル名のベース部分.

    Returns:
        なし (表をコンソールに出力し、csvにして保存).
    """
    vowel_order = VOWEL_ORDER
    cons_order = CONS_ORDER
    table = pd.DataFrame(0, index=cons_order, columns=vowel_order)

    for con, vow, spec in df_mora_analysis[["consonant", "vowel", "special"]].to_numpy(object):
        if spec == "cl":
            table.loc["cl", "u"] += 1
        elif spec == "N":
            table.loc["N", "u"] += 1
        else:
            if con not in cons_order:
                con = "#"
            table.loc[con, vow] += 1

    table_count = table.copy()
    total = table_count.to_numpy().sum()
    table_prob = (table_count / total)

    """
    # 確認(合計が1になる)
    # print("Consonant-Vowel Table:")
    # print(table)
    # print(total)
    # print(table_prob.to_numpy().sum())
    """

    cnt_filename = f"{name} consonant_vowel_table.csv"
    prb_filename = f"{name} consonant_vowel_prob_table.csv"
    save_csvfolder = fol_02exp_csv / name
    save_csvfolder.mkdir(exist_ok=True)
    save_count_path = save_csvfolder / cnt_filename
    save_prob_path = save_csvfolder / prb_filename
    table.to_csv(save_count_path, encoding="utf-8-sig")
    table_prob.to_csv(save_prob_path, encoding="utf-8-sig")

    plt.figure(figsize=(8, 10))
    plt.imshow(table_prob, cmap="Blues", aspect="auto")
    plt.colorbar(label="Probability")
    plt.xticks(ticks=np.arange(len(vowel_order)), labels=vowel_order)
    plt.yticks(ticks=np.arange(len(cons_order)), labels=cons_order)
    plt.xlabel("Vowel")
    plt.ylabel("Consonant")
    plt.title(f"{name} Consonant-Vowel Heatmap")
    plt.tight_layout()
    heatmap_fig_path = fol_02exp_heatmap / f"{name} consonant_vowel_heatmap.png"
    plt.savefig(heatmap_fig_path)
    plt.close()

def count_phoneme_durations(data: list) -> tuple[dict, dict, dict]:
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
    c_durs = dict.fromkeys(CONSONANTS, 0)
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
    count_dict: dict[str, int],
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
    slong: dict[str, int],
    long: dict[str, int],
    short: dict[str, int],
    folder: Path,
    mode: str,
    base_title: str,
) -> None:
    """slong/long/short の3種類について count ヒストグラムをまとめて描画."""
    folder.mkdir(exist_ok=True)
    plot_count_histogram(slong, title=f"{_sl} {mode} {base_title}", folder=folder, mode=mode)
    plot_count_histogram(long,  title=f"{_l} {mode} {base_title}", folder=folder, mode=mode)
    plot_count_histogram(short,  title=f"{_s} {mode} {base_title}", folder=folder, mode=mode)

@dataclass(frozen=True)
class ThreeLevelArtConfig:
    """articulation プロットの設定情報をまとめたデータクラス."""

    con_fig_dir: Path
    key_list: list[str]
    label: str
    base_title: str
    ylim: tuple[int, int]


def plot_three_levels_art(
    slong: dict[str, int],
    long: dict[str, int],
    short: dict[str, int],
    cfg: ThreeLevelArtConfig
) -> None:
    """slong/long/short の3種類について articulation ヒストグラムをまとめて描画."""
    sub_dir = cfg.con_fig_dir / cfg.label
    sub_dir.mkdir(exist_ok=True)

    plot_art_histgram(
        full_dict=slong,
        key_list=cfg.key_list,
        title=f"{_sl} {cfg.label} {cfg.base_title}",
        folder=sub_dir,
        ylim=cfg.ylim,
    )
    plot_art_histgram(
        full_dict=long,
        key_list=cfg.key_list,
        title=f"{_l} {cfg.label} {cfg.base_title}",
        folder=sub_dir,
        ylim=cfg.ylim,
    )
    plot_art_histgram(
        full_dict=short,
        key_list=cfg.key_list,
        title=f"{_s} {cfg.label} {cfg.base_title}",
        folder=sub_dir,
        ylim=cfg.ylim,
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
