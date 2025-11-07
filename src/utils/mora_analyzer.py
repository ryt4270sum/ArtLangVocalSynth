"""モーラおよび音素の分析を行うためのユーティリティクラス群."""

from collections import Counter
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import pandas as pd
import pyopenjtalk as pjt


class MoraAnalyzer:
    """日本語歌詞のモーラ分割と音素変換を行うクラス."""

    VOWELS: ClassVar[list[str]] = ["a", "i", "u", "e", "o"]
    VOWELS_HIRA: ClassVar[str] = "あいうえお"
    SPECIAL_SYMBOLS: ClassVar[list[str]]= ["cl", "N"]

    SMALL_VOWELS: ClassVar[str] = "ぁぃぅぇぉ"
    SMALL_Y: ClassVar[str] = "ゃゅょ"
    SMALL_Y_MAP: ClassVar[dict[str, str]] = {"ゃ": "あ", "ゅ": "う", "ょ": "お"}

    SPECIAL_MORA_VOWEL: ClassVar[dict[str, str]] = {
        "っ": "",
        "ん": "",
        "ー": "",
        "ゔ": "う",
        "を": "お",
        "ゎ": "あ",
        "ゐ": "い",
        "ゑ": "え",
    }

    ROWS: ClassVar[list[str]] = [
        "かきくけこ",
        "がぎぐげご",
        "さしすせそ",
        "ざじずぜぞ",
        "たちつてと",
        "だぢづでど",
        "なにぬねの",
        "はひふへほ",
        "ばびぶべぼ",
        "ぱぴぷぺぽ",
        "まみむめも",
        "やゆよ",
        "らりるれろ",
        "わを",
    ]

    KATAKANA_START: ClassVar[int] = 0x30A1
    KATAKANA_END: ClassVar[int] = 0x30F6
    HIRAGANA_OFFSET: ClassVar[int] = 0x60

    _INVALID_MODE_MSG: str = "mode must be 'first' | 'all_list' | 'all_joined'"

    def __init__(
        self,
        second_mora_exclude: list[str] | None = None,
        *,
        normalize_long_vowel: bool = True,
    ) -> None:
        """MoraAnalyzer を初期化する.

        Args:
            second_mora_exclude: 2モーラの後部に現れたときに特別扱いする文字のリスト.
                デフォルトでは ['っ', 'ん', 'あ', 'い', 'う', 'え', 'お', 'ー'] を用いる.
            normalize_long_vowel: 長音記号 "ー" を前モーラの母音に正規化するかどうか.
        """
        self.second_mora_exclude = second_mora_exclude or [
            "っ",
            "ん",
            "あ",
            "い",
            "う",
            "え",
            "お",
            "ー",
        ]
        self.normalize_long_vowel = normalize_long_vowel

    # --- 単体ユーティリティ ---

    def split_moras(self, text: str) -> tuple[list[str], int]:
        """文字列をモーラ単位に分割する.

        小書き仮名 (ゃ, ゅ, ょ, ぁ, ぃ, ぅ, ぇ, ぉ) は直前の文字と結合して1モーラとみなす.

        Args:
            text: 分割対象の文字列.

        Returns:
            2要素のタプル.
                - moras: モーラ列.
                - count: モーラ数.
        """
        small_chars = ["ゃ", "ゅ", "ょ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ"]
        moras: list[str] = []

        for char in text:
            if moras and char in small_chars:
                moras[-1] += char
            else:
                moras.append(char)

        return moras, len(moras)

    def _special_vowel_case(self, mora: str) -> str | None:
        """特殊モーラに対する母音を返す.

        撥音, 促音, 長音記号や, ゐ, ゑ などの特殊な文字を扱う.
        対応表に存在しない場合は None を返す.

        Args:
            mora: 1モーラ分の文字列.

        Returns:
            対応する母音1文字, または空文字列, 対応なしの場合は None.
        """
        return self.SPECIAL_MORA_VOWEL.get(mora)

    def _to_hiragana(self, text: str) -> str:
        """全角カタカナをひらがなに変換する.

        Args:
            text: 変換対象の文字列.

        Returns:
            カタカナをひらがなに置き換えた文字列.
        """
        chars: list[str] = []
        for ch in text:
            code = ord(ch)
            if self.KATAKANA_START <= code <= self.KATAKANA_END:
                chars.append(chr(code - self.HIRAGANA_OFFSET))
            else:
                chars.append(ch)
        return "".join(chars)


    def _single_char_vowel(self, ch: str) -> str:
        """1文字モーラの母音を返す."""
        vowels = self.VOWELS_HIRA

        if ch in vowels:
            return ch

        if ch in self.SMALL_VOWELS:
            return vowels[self.SMALL_VOWELS.index(ch)]

        if ch in self.SMALL_Y:
            return self.SMALL_Y_MAP[ch]

        for row in self.ROWS:
            pos = row.find(ch)
            if pos != -1:
                return vowels[pos]

        return ""

    def _two_char_vowel(self, mora: str) -> str:
        """2文字モーラの母音を返す."""
        a, b = mora[0], mora[1]

        if b in self.SMALL_Y:
            return self.SMALL_Y_MAP[b]

        if b in self.SMALL_VOWELS:
            vowels = self.VOWELS_HIRA
            return vowels[self.SMALL_VOWELS.index(b)]

        # 小書きでなければ1文字目に委ねる
        return self.vowel_of_mora(a)

    def vowel_of_mora(self, mora: str) -> str:
        """単一モーラの母音をひらがなで返す.

        Args:
            mora: 1モーラ分の文字列.

        Returns:
            母音に対応するひらがな1文字.
            特定不能な場合や撥音, 促音, 長音などは空文字列を返す.
        """
        if not mora:
            return ""

        # カタカナをひらがなに統一
        mora = self._to_hiragana(mora)

        # 特殊モーラ判定
        special = self._special_vowel_case(mora)
        if special is not None:
            return str(special)

        length = len(mora)
        char_single = 1
        char_double = 2

        if length == char_single:
            return self._single_char_vowel(mora)

        if length == char_double:
            return self._two_char_vowel(mora)

        # 3文字以上は末尾だけ見て再帰
        return self.vowel_of_mora(mora[-1])


    def get_phonemes(self, mora: str) -> list[str]:
        """モーラ文字列を pyopenjtalk で音素列に変換する.

        Args:
            mora: モーラ文字列.

        Returns:
            音素記号のリスト.
        """
        s = str(mora).strip()
        if s == "":
            return []
        return list(pjt.g2p(s, kana=False).split())

    def get_consonant(self, phonemes: list[str]) -> str:
        """音素列から先頭子音を取り出す.

        先頭が母音または特殊音 (N, cl) の場合は子音なしとして "None" を返す.

        Args:
            phonemes: 音素列.

        Returns:
            先頭子音または "None".
        """
        if not phonemes:
            return "None"

        first = phonemes[0]

        if first in self.VOWELS or first in {"N", "cl"}:
            return "None"

        return first

    def get_vowel(self, phonemes: list[str], mode: str) -> str | list[str]:
        """音素列から母音のみを抽出する.

        Args:
            phonemes: 音素列.
            mode: 返り値の形式を指定するモード.
                "first": 最初に出現する母音を1文字の文字列で返す.
                "all_list": すべての母音をリストで返す.
                "all_joined": すべての母音を結合した文字列で返す.

        Returns:
            mode に応じて, 母音1文字, 母音リスト, 母音連結文字列のいずれか.
        """
        if not phonemes:
            return "" if mode == "first" else []

        vowels = [p for p in phonemes if p in self.VOWELS]

        if mode == "first":
            return vowels[0] if vowels else "None"
        if mode == "all_list":
            return vowels
        if mode == "all_joined":
            return "".join(vowels)

        raise ValueError(self._INVALID_MODE_MSG)

    def get_special_symbol(self, phonemes: list[str]) -> str:
        """音素列から特殊記号 (撥音, 促音) を抽出する.

        Args:
            phonemes: 音素列.

        Returns:
            "cl", "N", または "None".
        """
        if "cl" in phonemes:
            return "cl"
        if "N" in phonemes:
            return "N"
        return "None"

    # --- 前処理(歌詞の正規化)---

    def preprocess_lyrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """歌詞列を正規化する.

        - 長音記号 "ー" を前モーラの母音に置き換える (オプション).
        - 表記ゆれをいくつか統一する (例: "を" → "お", "う゛" → "ヴ").

        Args:
            df: 少なくとも "lyric" 列を含む DataFrame.

        Returns:
            正規化後の DataFrame.
        """
        df = df.copy()

        if self.normalize_long_vowel:
            lyrics = df["lyric"].tolist()
            for i in range(1, len(lyrics)):
                if lyrics[i] == "ー":
                    v = self.vowel_of_mora(lyrics[i - 1])
                    lyrics[i] = v if v else "ー"
            df["lyric"] = lyrics

        df["lyric"] = df["lyric"].astype(str)
        df["lyric"] = df["lyric"].str.replace("を", "お")
        df["lyric"] = df["lyric"].str.replace("う゛", "ヴ")

        return df

    # --- 有効行抽出(単モーラ/条件付き2モーラのみを残す)---

    def extract_valid_rows(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[list[Any]], list[list[Any]]]:
        """単モーラおよび条件付き2モーラのみを抽出する.

        Args:
            df: 少なくとも "pitch_normalize", "lyric" を含む DataFrame.

        Returns:
            3要素のタプル.
                - df_valid: 空白行などを除去した DataFrame.
                - valid_data: 有効と判定された [pitch, mora] のリスト.
                - excluded: 除外された [pitch, lyric] のリスト.
        """
        mask = df["lyric"].notna() & df["lyric"].str.strip().ne("")
        df_valid = df.loc[mask, ["pitch_normalize", "lyric"]].copy()

        valid_data: list[list[Any]] = []
        excluded: list[list[Any]] = []

        for pitch, lyric in df_valid.to_numpy(object):
            moras, count = self.split_moras(lyric)

            mora_single = 1
            mora_double = 2

            if count == mora_single:
                valid_data.append([pitch, lyric])
            elif count == mora_double:
                if moras[1] in self.second_mora_exclude:
                    valid_data.append([pitch, moras[0]])
                    valid_data.append([pitch, moras[1]])
                else:
                    excluded.append([pitch, lyric])
            else:
                excluded.append([pitch, lyric])

        return df_valid, valid_data, excluded

    # --- ピッチ群分け ---

    def group_by_pitch(
        self,
        valid_data: list[list[Any]],
    ) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
        """相対ピッチ値に基づいて High, Mid, Low の3群に分割する.

        Args:
            valid_data: [pitch, mora] のリスト.

        Returns:
            3要素のタプル.
                - high: pitch > 0 のデータ.
                - mid: pitch == 0 のデータ.
                - low: pitch < 0 のデータ.
        """
        high: list[list[Any]] = []
        mid: list[list[Any]] = []
        low: list[list[Any]] = []

        for pitch, mora in valid_data:
            if pitch > 0:
                high.append([pitch, mora])
            elif pitch == 0:
                mid.append([pitch, mora])
            else:
                low.append([pitch, mora])

        return high, mid, low

    # --- 可視化(任意)---

    def plot_mora_hist(
        self,
        data: list[list[Any]],
        title: str,
    ) -> None:
        """モーラの出現回数ヒストグラムを描画する.

        Args:
            data: [pitch, mora] 形式のデータ.
            title: グラフタイトル.
        """
        moras = [m for _, m in data]
        cnt = Counter(moras)
        df = pd.DataFrame(
            {
                "mora": list(cnt.keys()),
                "count": list(cnt.values()),
            },
        ).sort_values("count", ascending=False)

        plt.figure(figsize=(12, 5))
        plt.bar(df["mora"], df["count"], edgecolor="white", alpha=0.9)
        plt.title(title)
        plt.xlabel("Mora")
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    # --- 一括実行ラッパ ---

    def analyze(
        self,
        df_rows: pd.DataFrame,
    ) -> dict[str, Any]:
        """モーラ解析の一連の処理をまとめて実行する.

        Args:
            df_rows: 少なくとも "pitch_normalize", "lyric" を含む DataFrame.
                例: MusicXML から抽出した生データを集約したもの.

        Returns:
            解析結果をまとめた辞書.
                - df_valid: 正規化・フィルタ後の DataFrame.
                - valid_data: 有効な [pitch, mora] のリスト.
                - excluded: 除外された [pitch, lyric] のリスト.
                - groups: High/Mid/Low ごとのデータを収めた dict.
        """
        df_prep = self.preprocess_lyrics(df_rows)
        df_valid, valid_data, excluded = self.extract_valid_rows(df_prep)
        high, mid, low = self.group_by_pitch(valid_data)

        return {
            "df_valid": df_valid,
            "valid_data": valid_data,
            "excluded": excluded,
            "groups": {
                "High": high,
                "Mid": mid,
                "Low": low,
            },
        }
