from music21 import converter, note, metadata, tempo
import pandas as pd

def visualize_musicxml(xml_path: str):
    """
    MusicXML を読み込んで、以下の 2 つの pandas.DataFrame を返します：
      1. basic_df: 楽譜の基本情報
         - title:   作品名
         - movement: 楽章名
         - composer: 作曲者
         - part_count: パート数
         - bpm:     テンポ（BPM。なければ NaN）
      2. score_df: 譜面情報
         - measure:   小節番号
         - offset:    小節内オフセット（四分音符単位）
         - duration:  四分音符何拍分か
         - pitch:     音高（休符は空文字）
         - is_rest:   休符かどうか
         - lyric:     歌詞テキスト（なければ空文字）
    """

    # MusicXML 読み込み
    score = converter.parse(xml_path)

    # ─── 1. 基本情報の収集 ──────────────────────────
    md = score.metadata
    # テンポ記号を探す（最初の tempo メタトラックから取得）
    bpm = None
    for el in score.recurse().getElementsByClass(tempo.MetronomeMark):
        bpm = el.number
        break

    basic_info = {
        "title":       md.title or "",
        "movement":    md.movementName or "",
        "composer":    md.composer or "",
        "part_count":  len(score.parts),
        "bpm":         bpm or ""
    }
    basic_df = pd.DataFrame([basic_info])

    # ─── 2. 譜面情報の収集 ──────────────────────────
    records = []
    for part in score.parts:
        part_id   = part.id
        part_name = part.partName or ""
        for elem in part.flat.getElementsByClass([note.Note, note.Rest]):
            is_rest = isinstance(elem, note.Rest)
            rec = {
                "part_id":   part_id,
                "part_name": part_name,
                "measure":   elem.measureNumber,
                "offset":    float(elem.offset),
                "duration":  float(elem.duration.quarterLength),
                "is_rest":   is_rest,
                "pitch":     "" if is_rest else elem.pitch.nameWithOctave,
                "lyric":     elem.lyrics[0].text if isinstance(elem, note.Note) and elem.lyrics else ""
            }
            records.append(rec)

    score_df = (pd.DataFrame(records)
                  .sort_values(["part_id","measure","offset"])
                  .reset_index(drop=True))

    return basic_df, score_df


def save_xml2html(basic_df: pd.DataFrame, score_df: pd.DataFrame, html_path: str):
    # 1. 基本情報を HTML リストとして作成
    info = basic_df.iloc[0].to_dict()
    basic_html = "<h2>楽譜の基本情報</h2>\n<ul>\n"
    for k, v in info.items():
        basic_html += f"  <li><strong>{k}</strong>: {v}</li>\n"
    basic_html += "</ul>\n"

    # 2. 譜面情報の HTML テーブル部分
    #    offset と duration を除外して転置
    df_for_html = score_df.drop(columns=["part_id", "part_name", "offset"])
    df_t = df_for_html.T
    df_t.columns = score_df["measure"].tolist()


    table_html = df_t.to_html(index=True, header=False, escape=False)

    # 3. 全体の HTML を組み立て
    html_full = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>MusicXML Visualize</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    .table-container {{ overflow-x: auto; padding: 10px; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; white-space: nowrap; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>MusicXML Visualize</h1>
  {basic_html}
  <div class="table-container">
    {table_html}
  </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)
    print(f"→ {html_path} に保存しました。")

def xml2html(inp_xml: str, out_html:str):
    xml_file = inp_xml
    html_file = out_html
    basic_df, score_df = visualize_musicxml(xml_file)  # pandas DataFrame 化 
    save_xml2html(basic_df, score_df, html_file)
    #print(df)  # または Jupyter Notebook で df と入力すると表形式で表示されます


if __name__ == "__main__":
    xml_file = r"xml_original\01.xml"
    html_file = "musicxml_visualization.html"
    xml2html(xml_file, html_file)