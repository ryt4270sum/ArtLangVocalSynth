"""MusicXML ファイルから音高統計量を抽出・可視化するスクリプト.

単一曲ごとの平均音高や中央値, 音価重み付き平均などを計算し, 必要に応じて可視化を行う.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from music21 import converter, note


class MusicXmlAnalyzer:
    """MusicXML データから音高・音価の統計量を計算するクラス.

    単一 XML ファイルから音高 (MIDI 値) と音価 (四分音符比) を抽出し,
    平均・中央値・音価重み付き平均の算出やヒストグラム保存を行う.
    """

    def __init__(
        self,
        pic_dir: Path | None = None,
        *,
        strip_ties: bool = True,
        include_grace: bool = False,
    ) -> None:
        """MusicXmlAnalyzer を初期化する.

        Args:
            pic_dir: ヒストグラム画像の保存先ディレクトリ. None の場合は保存しない.
            strip_ties: タイで結ばれた音符を結合して扱うかどうか.
            include_grace: 装飾音 (grace note) を含めるかどうか.
        """
        self.pic_dir = pic_dir
        self.strip_ties = strip_ties
        self.include_grace = include_grace

    def collect_pitches(
        self,
        xml_path: str | Path,
    ) -> tuple[str, list[int], list[float]]:
        """MusicXML ファイルから音高と音価を抽出する.

        Args:
            xml_path: MusicXML ファイルへのパス.

        Returns:
            3要素のタプル.
                - song: 曲名 (拡張子を除いたファイル名)
                - pitches: 音高 (MIDI 値) のリスト.
                - durs: 音価 (4分音符比) のリスト.
        """
        path = Path(xml_path)
        song = path.stem
        score = converter.parse(str(path))

        if self.strip_ties:
            # inPlace=False: 元を残して結合済み Stream を返す
            score = score.stripTies(inPlace=False)

        pitches: list[int] = []
        durs: list[float] = []

        for n in score.recurse().notes:
            # include_grace=False のときは装飾音や長さ 0 を除外
            is_grace = getattr(n.duration, "isGrace", False)
            is_zero_length = (n.duration.quarterLength or 0) == 0

            if not self.include_grace and (is_grace or is_zero_length):
                continue

            if isinstance(n, note.Note):
                pitches.append(int(n.pitch.midi))
                durs.append(float(n.duration.quarterLength))

        return song, pitches, durs

    def get_stat_xml(
        self,
        song: str,
        pitches: list[int],
        durs: list[float],
    ) -> dict[str, Any]:
        """1曲分の音高系列から統計量を計算する.

        Args:
            song: 曲名 (ファイル名など).
            pitches: 音高 (MIDI 値) のリスト.
            durs: 音価 (4分音符比) のリスト.

        Returns:
            曲ごとの統計量をまとめた辞書.
            キーは "song_id", "mean", "median", "wmean".
        """
        if not pitches:
            # 音が無い場合は NaN で返す
            return {"song_id": song, "mean": np.nan, "median": np.nan, "wmean": np.nan}

        mean = float(np.mean(pitches))
        median = float(np.median(pitches))

        if len(durs) == len(pitches) and sum(durs) > 0:
            wmean = float(np.average(pitches, weights=durs))
        else:
            wmean = float("nan")

        return {"song_id": song, "mean": mean, "median": median, "wmean": wmean}

    def save_histogram(
        self,
        song: str,
        pitches: list[int],
    ) -> None:
        """音高分布のヒストグラムを描画し, 画像として保存する.

        Args:
            song: 曲名 (ファイル名など).
            pitches: 音高 (MIDI 値) のリスト.
        """
        if self.pic_dir is None or not pitches:
            return

        note_min, note_max = min(pitches), max(pitches)
        # 右端を含めるよう +1
        bins = np.arange(note_min, note_max + 2, 1)
        pit_median = float(np.median(pitches))
        pit_mean = float(np.mean(pitches))

        plt.figure()
        plt.hist(pitches, bins=bins, range=(note_min, note_max + 1), ec="white")
        plt.title(f"MIDI Note Count [{song}.xml]")
        plt.xlabel("MIDI Note Number")
        plt.xticks(np.arange(note_min, note_max + 1, 1))
        plt.ylabel("Count")
        # 平均・中央値の線
        plt.axvline(
            pit_median,
            linestyle="--",
            linewidth=2,
            label=f"Median = {pit_median:.1f}",
        )
        plt.axvline(
            pit_mean,
            linestyle=":",
            linewidth=2,
            label=f"Mean = {pit_mean:.1f}",
        )
        plt.legend()

        self.pic_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.pic_dir / f"{song}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def main() -> None:
    """MusicXML データの統計量計算を一括で実行する."""
    # フォルダの設定
    musicxml_root = (
        Path(
            r"C:\Users\610ry\OneDrive - MeijiMail\院ゼミ・研究関連\修士論文",
        )
        / "東北きりたん歌唱データベース"
        / "kiritan_singing-master"
        / "musicxml"
    )

    save_dir = (
        Path(r"C:\Users\610ry\MyProgramFiles\ArtLangVocalSynth\ArtLangVocalSynth")
        / "pitch_mora_analysis"
        / "pitch_count"
    )

    collector = MusicXmlAnalyzer(
        pic_dir=save_dir,
        strip_ties=True,
        include_grace=False,
    )

    # 統計の集約
    stats: list[dict[str, Any]] = []

    for i in range(1, 51):
        xml_path = musicxml_root / f"{i:02}.xml"
        if not xml_path.is_file():
            continue

        song, pitches, durs = collector.collect_pitches(xml_path)
        # 必要に応じてヒストグラムを保存する場合は以下を有効化する.
        # collector.save_histogram(song, pitches)  # noqa: ERA001

        stat = collector.get_stat_xml(song, pitches, durs)
        stats.append(stat)

    # DataFrame 化
    df = pd.DataFrame(stats, columns=["song_id", "mean", "median", "wmean"])
    print(df.head())


if __name__ == "__main__":
    main()
