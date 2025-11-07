import random
import os
import pyopenjtalk #音素変換で必要

# 1. 確率モデルをもとに芸術言語の歌詞を作成する処理

def read_ngram_probabilities(file_path):
    """N-gram確率モデルを読み込む"""
    ngram_probabilities = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            ngram, prob = line.strip().split(': ')
            ngram = eval(ngram)  # 文字列をタプルに変換
            prob = float(prob)
            ngram_probabilities[ngram] = prob
    return ngram_probabilities

def choose_next_mora(ngram_probabilities, context, n):
    """
    次のモーラを確率的に選択する。
    モノグラム: contextはNone。
    バイグラム: contextは現在のモーラ（1つ）。
    トライグラム: contextは現在と直前のモーラ（2つ）。
    """
    if n == 1:
        # モノグラム：全モーラの中から確率的に選択
        candidates = [(ngram[0], prob) for ngram, prob in ngram_probabilities.items()]
    elif n == 2:
        # バイグラム/トライグラム：文脈に基づいて次のモーラを選択
        candidates = [(ngram[1], prob) for ngram, prob in ngram_probabilities.items() if ngram[0] == context]
    elif n == 3:
        # バイグラム/トライグラム：文脈に基づいて次のモーラを選択
        contextlist = list(context)
        candidates = [(ngram[2], prob) for ngram, prob in ngram_probabilities.items() if ngram[0]== contextlist[0] and ngram[1] == contextlist[1]]
    
    if not candidates:
        return None

    moras, probs = zip(*candidates)
    total_prob = sum(probs)
    normalized_probs = [prob / total_prob for prob in probs]
    next_mora = random.choices(moras, weights=normalized_probs)[0]
    return next_mora

def generate_sentence(ngram_probabilities, mora_count, n, SEED):
    """N-gram確率モデルを使って指定されたモーラ数の文章を生成する"""
    print("---今のシード")
    print(f"Seed: {SEED}")
    random.seed(SEED)  # シードを設定

    if not ngram_probabilities:
        return ""

    start_moras = list(set(ngram[0] for ngram in ngram_probabilities.keys()))    
    current_mora = random.choice(start_moras)
    sentence = [current_mora]
    if n==3:
        second_mora = random.choice(start_moras)
        sentence.append(second_mora)

    while len(sentence) < mora_count:
        if n<3:
            current_mora = sentence[len(sentence)-1]
        elif n ==3:
            current_mora = [sentence[len(sentence)-2], sentence[len(sentence)-1]]
        next_mora = choose_next_mora(ngram_probabilities, current_mora, n)

        if not next_mora:
            # 次のモーラがない場合は新しいモーラをランダムに選択して再開
            if n==3:
                print("cannot find next mora--start from random mora")
                sentence[len(sentence)-1] = random.choice(start_moras)
            else:
                print("cannot find next mora--start from random mora")
                current_mora = random.choice(start_moras)
        else:
            sentence.append(next_mora)
            current_mora = next_mora


    return ''.join(sentence)

def save_sentence(sentence, output_file):
    """生成された文章をファイルに保存する"""
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(sentence)

# 2. 出力された文字の間スペースを入れる処理

def insert_spaces(filename_in, filename_out):
    exceptions = {'ゃ', 'ゅ', 'ょ'}
    
    with open(filename_in, 'r', encoding='utf-8') as f:
        text = f.read()

    result = []
    for i, char in enumerate(text):
        # 先頭にはそのまま追加
        if i == 0:
            result.append(char)
            continue

        # 今の文字が例外（ゃゅょ）なら、直前にスペースを入れずに追加
        if char in exceptions:
            result.append(char)
        else:
            # それ以外なら、直前にスペースを入れてから追加
            result.append(' ')
            result.append(char)

    output_text = ''.join(result)

    # 結果を保存
    with open(filename_out, 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write(output_text)


if __name__ == '__main__':
    #初期設定
    ngram_prob_file = ''  # N-gram確率モデルのファイル   
    generate_type = '' #N-gram確率モデルの名前（mono, bi, tri, rand） 、output_fileに使う
    type = ''
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 読み込む確率モデルの設定(nの指定で変わる)

    n = 1  # 使用するN-gramのN（1=モノグラム、2=バイグラム、3=トライグラム，rand=0、生成する時のnは1にする）'
    if n == 1:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'monogram_prob.txt')
        generate_type = 'mono'
    elif n == 2:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'bigram_prob.txt')
        generate_type = 'bi'
    elif n == 3:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'trigram_prob.txt')
        generate_type = 'tri'
    else:
        ngram_prob_file = os.path.join(base_dir, 'probability', 'random_prob.txt')
        generate_type = 'rand'

    # 出力の設定
    songname = 'harugakita'
    SEED = 0
    mora_count = 100  # 生成する文章のモーラ数
    
    ngram_probabilities = read_ngram_probabilities(ngram_prob_file)

    # 芸術言語の歌詞作成処理
    if n == 0:
        sentence = generate_sentence(ngram_probabilities, mora_count, n=1, SEED=SEED)
    else:
        sentence = generate_sentence(ngram_probabilities, mora_count, n, SEED=SEED)

        output_file = os.path.join(base_dir, f'{songname}_{generate_type}_{str(mora_count)}_seed{str(SEED)}.txt') # 生成された文章を保存するファイル名
    
    #save_sentence(sentence, output_file)

    print(f'Generated sentence: \n{sentence}')

    # 歌詞の間にスペースを入れる処理
    inputtxt = output_file
    outputtxt = output_file
    insert_spaces(inputtxt, outputtxt)

    # 音素に変換する処理
    with open(output_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip() # 1行目（歌詞、スペースなし）を取り出す
        f.readline() # 2行目を読み飛ばす
        third_line = f.readline().strip() # 3行目（歌詞、スペースあり）を取り出す 
    #print(first_line)
    text = first_line
    text_gap = third_line
    phones = pyopenjtalk.g2p(text, kana=False) #kana=Trueにするとカナで生成されるらしい？（文字化けするかも）
    phones_gap = pyopenjtalk.g2p(text_gap, kana=False) #kana=Trueにするとカナで生成されるらしい？（文字化けするかも）、   '　'= pau
    
    print(phones)

    with open(output_file, mode='a') as f:
        f.write('\n\n')
        f.write(phones) # 5行目（歌詞の音素、スペースなし）を取り出す 
        f.write('\n\n')
        f.write(phones_gap) # 7行目（歌詞の音素、スペースあり）を取り出す 