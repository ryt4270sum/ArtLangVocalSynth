from music21 import converter, environment
import os
import Generate_fromProb as gp
import hashlib
import json
from datetime import datetime

# MuseScore のパス設定（必要に応じて）
env = environment.UserSettings()
env['musicxmlPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"


def write_musicxml(score, xml_path: str):
    """編集後の楽譜をファイルに保存（上書き）"""
    score.write("musicxml", fp=xml_path)

def show_musicxml(xml_path: str):
    """ファイルを再読み込みして常に最新の状態を表示"""
    updated_score = converter.parse(xml_path)
    updated_score.show()

def generate_output_dir(base_dir: str, songname: str, hash_digest: str) -> str:
    folder_name = f"{songname}_artlang_{hash_digest}"
    output_dir = os.path.join(base_dir, "xml_output", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_artlang_lyrics(base_dir: str, songname: str, gen_typeNum: int, mora_count: int = 100, seed: int = 0):
    # n-gram確率モデルを読み込み
    if gen_typeNum == 1:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'monogram_prob.txt')
        gen_type = 'mono'
    elif gen_typeNum == 2:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'bigram_prob.txt')
        gen_type = 'bi'
    elif gen_typeNum == 3:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'trigram_prob.txt')
        gen_type = 'tri'
    else:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'random_prob.txt')
        gen_type = 'rand'

    ngram_probabilities = gp.read_ngram_probabilities(ngram_prob_file)

    # シード設定と歌詞生成
    n = max(1, gen_typeNum) # gen_typeNumが0(rand)の場合に1を返すようにする
    sentence = gp.generate_sentence(ngram_probabilities, mora_count, n, SEED=seed)

    #一意のハッシュを生成
    hash_digest = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:8]

    return sentence, hash_digest, gen_type

def apply_lyrics_to_xml(original_xml_path: str, mora_list: list[str], output_xml_path: str):
    score = converter.parse(original_xml_path) # 元となるmusicxml
    notes = score.parts[0].recurse().notes
    for i, note in enumerate(notes):
        if i < len(mora_list):
            note.lyric = mora_list[i]
        else:
            break
    score.write("musicxml", fp=output_xml_path) #第一引数は書き出しフォーマット

def update_ArtLangLyrics(base_dir: str, config: dict):
    songname = config["songname"]
    gen_typeNum = config["gen_typeNum"]
    seed = config["seed"]
    mora_count = config["mora_count"]
    original_xml_path = config["original_xml_path"]

    
    sentence, hash_digest, gen_type = generate_artlang_lyrics(base_dir, songname, gen_typeNum, mora_count, seed)
    output_dir = generate_output_dir(base_dir, songname, hash_digest)

    #一度sentence（空白無し）をtxtファイルに保存し、空白を後から付け足す
    lyrics_txt_path = os.path.join(output_dir, f"{songname}_artlang_lyrics_{hash_digest}.txt")
    with open(lyrics_txt_path, "w", encoding="utf-8") as f:
        f.write(sentence)
    gp.insert_spaces(lyrics_txt_path, lyrics_txt_path)

    # lyrics.txt から mora_list を作成（3行目を利用）
    with open(lyrics_txt_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split('\n')

    if len(lines) < 3:
        raise ValueError("lyrics.txt にモーラ列（3行目）が存在しません")

    mora_line = lines[2].strip()
    mora_list = mora_line.split()

    # 楽譜に歌詞を埋め込んで保存
    xml_path = os.path.join(output_dir, f"{songname}_artlang_score_{hash_digest}.xml")
    apply_lyrics_to_xml(original_xml_path=original_xml_path, mora_list=mora_list, output_xml_path=xml_path)

    # metadata.json 保存
    metadata_path = os.path.join(output_dir, f"{songname}_artlang_metadata_{hash_digest}.json")
    metadata = {
        "songname": songname,
        "generation_type": gen_type,
        "mora_count": mora_count,
        "seed": seed,
        "hash": hash_digest,
        "lyrics_file": f"{songname}_artlang_lyrics_{hash_digest}.txt",
        "xml_file": f"{songname}_artlang_score_{hash_digest}.xml",
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return xml_path

def main():
    config = {
        "songname": "BadApple",
        "gen_typeNum": 1,
        "seed": 0,
        "mora_count": 10000,
        "original_xml_path": "xml_original/BadApple_main.xml"
    }
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config["original_xml_path"] = os.path.join(base_dir, config["original_xml_path"])

    output_xml_path = update_ArtLangLyrics(base_dir, config)

    show_musicxml(output_xml_path)

if __name__ == '__main__':
    main()