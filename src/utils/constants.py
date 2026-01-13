"""utils.constants - 定数定義モジュール."""

# 母音一覧
VOWELS = ["a", "i", "u", "e", "o"]

# 子音一覧
CONSONANTS = [
    "k","s","t","n","h","m","y","r","w",
    "g","z","d","b","p","ch","j","ts","f","sh",
    "ky","gy","ny","hy","by","my","py","ry","dy","ty","v"
]

# 特殊記号 : 撥音, 促音など
SPECIALS = ["cl", "N"]

# 子音の分類
nasal = ["n", "m", "ny", "my"]
plosive = ["b", "p", "d", "t", "g", "k", "by", "py", "dy", "ty", "gy", "ky"]
fricative = ["s", "sh", "z", "f", "h", "hy", "v", "ch", "j", "ts"]
tap = ["r", "ry"]
approximant = ["y", "w"]

# dataframe用の順序
VOWEL_ORDER = ["a", "i", "u", "e", "o"]
CONS_ORDER = ["#", "b", "by", "ch", "d", "dy", "f", "g", "gy", "h", "hy", "j", "k", "ky",
              "m", "my", "n", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts", "ty",
              "v", "w", "y", "z", "cl", "N"]

# ラベル用文字列
_vow, _con, _ss = "Vowel", "Consonant", "Special Symbol"
_nas, _plo, _fric, _tap, _app = "Nasal", "Plosive", "Fricative", "Tap", "Approximant"


# 解析時に除外する音素リスト
second_mora_exclude = {"っ", "ん", "あ", "い", "う", "え", "お", "ー"}