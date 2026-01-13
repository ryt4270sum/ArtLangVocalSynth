"""MusicXML ファイルから音高統計量を抽出・可視化するスクリプト.

単一曲ごとの平均音高や中央値, 音価重み付き平均などを計算し, 必要に応じて可視化を行う.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from music21 import converter, note, tempo


class MusicXmlAnalyzer:
    """MusicXML の音高系列に対する統計計算を行うクラス.

    主に pitch (MIDI ノート番号) と duration (音価) を入力として
    基本統計量(平均・中央値・音価重み付き平均)を返す.
    インスタンス内部状態を持たないため、すべて staticmethod として提供する.
    """

    @staticmethod
    def get_median(series: pd.Series) -> float:
        """Series の中央値を返す(空なら NaN)."""
        if series.empty:
            return float("nan")
        return float(series.median())
    @staticmethod
    def get_mean(series: pd.Series) -> float:
        """Series の平均値を返す(空なら NaN)."""
        if series.empty:
            return float("nan")
        return float(series.mean())

    @staticmethod
    def get_wmean(pitch: pd.Series, dur: pd.Series) -> float:
        """Series の平均値(時間で重みづけ)を返す(空なら NaN)."""
        if pitch.empty or dur.empty or pitch.size != dur.size:
            return float("nan")
        if dur.sum() == 0:
            return float("nan")
        return float(np.average(pitch, weights=dur))


class MusicXmlFile:
    """単一の MusicXML ファイルから音符情報を抽出するクラス.

    音符の pitch, duration, BPM, lyric を読み取り,
    歌詞なしノートの結合や relative_pitch の付与など,
    生データの整形処理のみを担当する.
    統計計算は MusicXmlAnalyzer に委譲する.
    """

    def __init__(self, xml_path: str) -> None:
        """クラスで使用する変数の定義."""
        self.xml_path = Path(xml_path)
        self.song = self.xml_path.stem
        self.score = converter.parse(xml_path)
        self.tied_score = self.score.stripTies(inPlace=False)
        self.data: pd.DataFrame | None = None

    def extract_data(self) -> pd.DataFrame:
        """MusicXML から音符情報を抽出して DataFrame を作成する.

        Returns:
            pandas.DataFrame:
                各ノートについて以下の列を持つデータフレーム.
                (song, note_idx, pitch, dur, BPM, lyric)
        """
        data = []
        note_idx = 0 #その楽曲中で何番目の音符か
        for n in self.tied_score.recurse().notes:
            if not isinstance(n,note.Note):
                continue
            if n.duration.isGrace:
                continue
            ql = n.duration.quarterLength
            if ql is None or ql == 0:
                continue

            mm = n.getContextByClass(tempo.MetronomeMark)
            bpm_value = mm.getQuarterBPM() if mm is not None else None
            bpm = int(bpm_value) if bpm_value is not None else None

            lyric_raw = n.lyric if n.lyric is not None else ""
            lyric = lyric_raw.strip()

            data.append({
                "song": self.song,
                "note_idx": note_idx,
                "pitch": int(n.pitch.midi),
                "dur": float(ql),
                "BPM": bpm,
                "lyric": lyric
            })
            note_idx += 1

        self.data = pd.DataFrame(data)
        return self.data

    def extract_data_with_rests(self) -> pd.DataFrame:
        """MusicXML から音符・休符情報を抽出して DataFrame を作成する.

        Returns:
        pandas.DataFrame:
            各イベント(ノート/休符)について以下の列を持つデータフレーム.
            (song, event_idx, is_rest, pitch, dur, BPM, lyric)
            - pitch: 休符のとき None
            - lyric: 休符のとき ""
        """
        data = []
        evt_idx = 0  # その楽曲中で何番目のイベントか
        for e in self.tied_score.recurse().notesAndRests:
            if not isinstance(e, (note.Note, note.Rest)):
                continue
            if e.duration.isGrace:
                continue
            ql = e.duration.quarterLength
            if ql is None or ql == 0:
                continue

            lyric_raw = e.lyric if e.lyric is not None else ""
            lyric = lyric_raw.strip()

            data.append({
                "song": self.song,
                "event_idx": evt_idx,
                "is_rest": isinstance(e, note.Rest),
                "pitch": int(e.pitch.midi) if isinstance(e, note.Note) else None,
                "dur": float(ql),
                "lyric": lyric  # 休符でも歌詞情報を保持
            })
            evt_idx += 1

        self.data = pd.DataFrame(data)
        return self.data


    def merge_lyricless_notes(self) -> pd.DataFrame:
        """歌詞の無いノートを直前のノートに結合して音価のみ加算する.

        曲頭の歌詞無しノートは破棄する.

        Returns:
            pandas.DataFrame: 歌詞ありノートのみで構成されたデータフレーム.
        """
        if self.data is None:
            msg = "先に extract_data() を呼んで self.data を作ってください。"
            raise  ValueError(msg)

        df = (self.data.sort_values(["song", "note_idx"]).reset_index(drop=True).copy())

        merged_rows: list[dict[str, object]] = []

        for _, row in df.iterrows():
            song = row["song"]
            note_idx = row["note_idx"]
            pitch = row["pitch"]
            dur = row["dur"]
            bpm = row["BPM"]
            lyric = row["lyric"]

            if (pd.isna(lyric) or str(lyric).strip() ==""):
                if merged_rows and merged_rows[-1]["song"] == song:
                    merged_rows[-1]["dur"] += dur
                else:
                    continue
            else:
                merged_rows.append({
                "song": song,
                "note_idx": note_idx,
                "pitch": pitch,
                "dur": dur,
                "BPM": bpm,
                "lyric": str(lyric).strip(),
            })

        self.data = pd.DataFrame(merged_rows)
        return self.data

    def merge_lyricless_notes_with_rests(self) -> pd.DataFrame:
        """歌詞の無いノートを直前のノートに結合して音価のみ加算する.

        休符はそのまま保持する.

        Returns:
            pandas.DataFrame: 歌詞ありノートと休符で構成されたデータフレーム.
        """
        if self.data is None:
            msg = "先に extract_data_with_rests() を呼んで self.data を作ってください。"
            raise  ValueError(msg)

        df = (self.data.sort_values(["song", "event_idx"]).reset_index(drop=True).copy())

        merged_rows: list[dict[str, object]] = []

        for _, row in df.iterrows():
            song = row["song"]
            event_idx = row["event_idx"]
            is_rest = row["is_rest"]
            pitch = row["pitch"]
            dur = row["dur"]
            lyric = row["lyric"]

            # 最初の行の場合
            if merged_rows == []:
                merged_rows.append({
                    "song": song,
                    "event_idx": event_idx,
                    "is_rest": is_rest,
                    "pitch": pitch,
                    "dur": dur,
                    "lyric": str(lyric).strip(),
                })
                continue

            # 休符の場合
            if is_rest == True:
                if merged_rows[-1]["is_rest"] and song == merged_rows[-1]["song"]:
                    merged_rows[-1]["dur"] += dur
                else:
                    merged_rows.append({
                    "song": song,
                    "event_idx": event_idx,
                    "is_rest": is_rest,
                    "pitch": pitch,
                    "dur": dur,
                    "lyric": str(lyric).strip(),
                })
            # ノートの場合
            else:
                if str(lyric).strip() == "":
                    if merged_rows[-1]["is_rest"] == False and song == merged_rows[-1]["song"]:
                        merged_rows[-1]["dur"] += dur
                    else:
                        merged_rows.append({
                        "song": song,
                        "event_idx": event_idx,
                        "is_rest": is_rest,
                        "pitch": pitch,
                        "dur": dur,
                        "lyric": str(lyric).strip(),
                    })
                else:
                    merged_rows.append({
                    "song": song,
                    "event_idx": event_idx,
                    "is_rest": is_rest,
                    "pitch": pitch,
                    "dur": dur,
                    "lyric": str(lyric).strip(),
                })

        self.data = pd.DataFrame(merged_rows)
        return self.data


    def add_relative_pitch(self) -> pd.DataFrame:
        """self.data にrelative_pitch列を追加して返す."""
        if self.data is None:
            msg = "先に extract_data() を呼んで self.data を作ってください。"
            raise  ValueError(msg)

        df = self.data.copy()
        median_pitch = MusicXmlAnalyzer.get_median(df["pitch"])
        df["relative_pitch"] = df["pitch"] - median_pitch

        self.data = df
        return self.data

    def add_interval_pitch(self) -> pd.DataFrame:
        """self.data にinterval_pitch列を追加して返す."""
        if self.data is None:
            msg = "先に extract_data_with_rests() を呼んで self.data を作ってください。"
            raise  ValueError(msg)

        df = self.data.copy()
        interval_pitches: list[int | None] = [0]  # 最初の音符の音高差は0とする
        for i in range(1, len(df)):
            same_song = df.loc[i, "song"] == df.loc[i - 1, "song"]
            curr_is_rest = bool(df.loc[i, "is_rest"])
            prev_is_rest = bool(df.loc[i - 1, "is_rest"])
            if same_song:
                if (not curr_is_rest) and (not prev_is_rest):
                    interval = df.loc[i, "pitch"] - df.loc[i - 1, "pitch"]
                    interval_pitches.append(interval)
                else:
                    interval_pitches.append(None)  # 休符がある場合はNoneとする
            else:
                interval_pitches.append(None)  # 曲が変わる場合や、前後に休符を含む場合はNoneとする
        df["interval_pitch"] = interval_pitches

        self.data = df
        return self.data

class MusicXmlData:
    """MusicXML フォルダ全体を読み込み, 全曲の音符情報を集約するクラス."""
    def __init__(self, folder: str) -> None:
        """分析対象のフォルダパスを受け取り, 参照可能にする."""
        self.folder = Path(folder)
        self.all_df: pd.DataFrame | None = None

    def exp01_load(self) -> pd.DataFrame:
        """exp01: フォルダ内にあるすべてのxmlをまとめたデータセットを作る."""
        all_data: list[pd.DataFrame] = []

        for file in self.folder.iterdir():
            if file.suffix.lower() == ".xml":
                xml_path = Path(self.folder) / file
                xml = MusicXmlFile(str(xml_path))
                xml.extract_data()
                xml.merge_lyricless_notes()
                exp01_df = xml.add_relative_pitch() #relative_pitchの追加

                all_data.append(exp01_df)

        self.all_df = pd.concat(all_data, ignore_index=True)
        return self.all_df

    def exp02_load(self) -> pd.DataFrame:
        """exp02: フォルダ内にあるすべてのxmlをまとめたデータセットを作る."""
        all_data: list[pd.DataFrame] = []

        for file in self.folder.iterdir():
            if file.suffix.lower() == ".xml":
                xml_path = Path(self.folder) / file
                xml = MusicXmlFile(str(xml_path))
                xml.extract_data()
                exp02_df = xml.merge_lyricless_notes()
                all_data.append(exp02_df)

        self.all_df = pd.concat(all_data, ignore_index=True)
        return self.all_df

    def exp03_load(self) -> pd.DataFrame:
        """exp03: フォルダ内にあるすべてのxmlをまとめたデータセットを作る."""
        all_data: list[pd.DataFrame] = []

        for file in self.folder.iterdir():
            if file.suffix.lower() == ".xml":
                xml_path = Path(self.folder) / file
                xml = MusicXmlFile(str(xml_path))
                xml.extract_data_with_rests()
                exp03_df = xml.merge_lyricless_notes_with_rests()
                exp03_df = xml.add_interval_pitch()
                all_data.append(exp03_df)

        self.all_df = pd.concat(all_data, ignore_index=True)
        return self.all_df

class MusicXmlVisualizer:
    """音高分布をヒストグラムとして可視化し, 画像として保存するクラス."""

    def __init__(self, df: pd.DataFrame, pic_dir: str | None) -> None:
        """画像の保存先ディレクトリを指定して初期化する.

        Args:
            df: データフレーム.
            pic_dir: 画像を保存するディレクトリパス. None の場合は保存を行わない.
        """
        self.df = df
        if pic_dir is None:
            self.pic_dir = None
        else:
            self.pic_dir = Path(pic_dir)

    def plot_song_pitch_hist(self) -> None:
        """音高分布をヒストグラムとして可視化し, 画像として保存する."""
        if self.df is None or self.df.empty:
            return
        if self.pic_dir is None:
            return

        song:str = self.df["song"].iloc[0] #曲名 (ファイル名など)
        pitches:list[int] = self.df["pitch"].to_list() #音高 (MIDI 値) のリスト

        note_min, note_max = min(pitches), max(pitches)
        bins = np.arange(note_min, note_max + 2, 1).tolist() # 右端を含めるよう +1

        # Analyzer で統一
        median_pitch: float = MusicXmlAnalyzer.get_median(self.df["pitch"])
        mean_pitch: float = MusicXmlAnalyzer.get_mean(self.df["pitch"])

        plt.figure()
        plt.hist(pitches, bins=bins, ec="white")
        plt.title(f"MIDI Note Count [{song}.xml]")
        plt.xlabel("MIDI Note Number")
        plt.xticks(np.arange(note_min, note_max + 1, 1))
        plt.ylabel("Count")
        # 平均・中央値の線
        plt.axvline(
            median_pitch,
            linestyle="--",
            c="red",
            linewidth=2,
            label=f"Median = {median_pitch:.1f}",
        )
        plt.axvline(
            mean_pitch,
            linestyle=":",
            c="lime",
            linewidth=2,
            label=f"Mean = {mean_pitch:.1f}",
        )
        plt.legend()

        self.pic_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.pic_dir / f"{song}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"{save_path}に保存しました")
