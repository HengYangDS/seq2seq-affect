from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import argparse
import regex as re

parser = argparse.ArgumentParser()
parser.add_argument('--vocabulary_path', dest='vocabulary_path', default='../data/embed.txt', type=str, help='词汇表位置')
parser.add_argument('--vad_path', dest='vad_path', default='../data/raw/NRC-VAD-Lexicon.txt', type=str, help='vad字典位置')
parser.add_argument('--symbol_path', dest='symbol_path', default='../data/raw/english_symbols.txt', type=str, help='标点位置')
parser.add_argument('--pad_token', dest='pad_token', default='<pad>', type=str, help='pad的记法')
parser.add_argument('--output_file', dest='output_file', default='../data/vad.txt', type=str, help='结果输入位置')
args = parser.parse_args()


# 载入词汇表
def load_vocabulary(fp):

    vocab = []

    with open(fp, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            word = line[:line.find(' ')]
            vocab.append(word)

    print('词汇表大小:', len(vocab))

    return vocab


def load_vad_lexicon(fp):

    dict_vad = {}

    fr = open(fp, 'r', encoding='utf8')
    fr.readline()  # 第一行的title不要
    line = fr.readline().strip()
    while(line):
        if len(line.split()) > 4:  # 如果是几个单词构成的词汇
            line = fr.readline().strip()
            continue
        word, v, a, d = line.split()
        dict_vad[word] = [v, a, d]
        line = fr.readline().strip()

    print('vad字典大小:', len(dict_vad))

    return dict_vad


def load_symbols(fp):

    symbols = []

    with open(fp, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            symbols.append(line)

    print('标点个数:', len(symbols))

    return symbols


if __name__ == '__main__':

    vocab = load_vocabulary(args.vocabulary_path)  # 载入词汇表
    vad = load_vad_lexicon(args.vad_path)  # 载入vad字典
    symbols = load_symbols(args.symbol_path)  # 载入标点符号

    regex = re.compile(r'^[0-9]+(.[0-9]+)*$')
    porter_stemmer = PorterStemmer()
    word_net_lemmatizer = WordNetLemmatizer()

    vocab_vad_dict = {}
    num_not_in_vad = 0
    num_symbols = 0
    num_numbers = 0
    num_porter = 0
    num_lemma = 0
    num_syn = 0
    num_porter_syn = 0
    num_lemma_syn = 0
    num_init_neutral = 0

    init_words = []

    for word in vocab:

        if word == args.pad_token:
            vocab_vad_dict[word] = ['0.0'] * 3
        elif word in vad:  # 如果单词直接出现在vad字典之中
            vocab_vad_dict[word] = vad[word]
        else:  # 单词并没有出现在vad字典之中
            num_not_in_vad += 1

            # lemma词根查找
            if word_net_lemmatizer.lemmatize(word) in vad:
                num_lemma += 1
                vocab_vad_dict[word] = vad[word_net_lemmatizer.lemmatize(word)]
                continue

            # porter词根查找
            if porter_stemmer.stem(word) in vad:
                num_porter += 1
                vocab_vad_dict[word] = vad[porter_stemmer.stem(word)]
                continue

            # 同义词查找
            synonyms = []
            synsets = wordnet.synsets(word)
            if synsets:
                for sense in synsets:
                    for lemma in sense.lemmas():
                        synonyms.append(lemma.name())
                synonyms = set(synonyms)
                flag = False
                for syn in synonyms:
                    if syn in vad:  # 如果同义词在vad字典中
                        flag = True
                        num_syn += 1
                        vocab_vad_dict[word] = vad[syn]
                        break
                if flag:
                    continue

            # lemma词根的同义词查找
            synonyms = []
            synsets = wordnet.synsets(word_net_lemmatizer.lemmatize(word))
            if synsets:
                for sense in synsets:
                    for lemma in sense.lemmas():
                        synonyms.append(lemma.name())
                synonyms = set(synonyms)
                flag = False
                for syn in synonyms:
                    if syn in vad:  # 如果同义词在vad字典中
                        flag = True
                        num_lemma_syn += 1
                        vocab_vad_dict[word] = vad[syn]
                        break
                if flag:
                    continue

            # porter词根的同义词查找
            synonyms = []
            synsets = wordnet.synsets(porter_stemmer.stem(word))
            if synsets:
                for sense in synsets:
                    for lemma in sense.lemmas():
                        synonyms.append(lemma.name())
                synonyms = set(synonyms)
                flag = False
                for syn in synonyms:
                    if syn in vad:  # 如果同义词在vad字典中
                        flag = True
                        num_porter_syn += 1
                        vocab_vad_dict[word] = vad[syn]
                        break
                if flag:
                    continue

            if word in symbols:  # 如果是标点符号，赋予中性词
                num_symbols += 1
                vocab_vad_dict[word] = ['0.5'] * 3
                continue

            if re.search(regex, word):  # 如果是数字，赋予中性词
                num_numbers += 1
                vocab_vad_dict[word] = ['0.5'] * 3
                continue

            vocab_vad_dict[word] = ['0.5'] * 3
            init_words.append(word)

    num_in_stopwords = 0
    for word in init_words:
        if word in stopwords.words('english'):
            num_in_stopwords += 1

    print('初始化vad字典大小:', len(vocab_vad_dict))
    print('在nrc-vad-lexicon中的单词个数:', len(vocab_vad_dict) - num_not_in_vad)
    print('不在nrc-vad-lexicon中的单词个数:', num_not_in_vad)
    print('通过lemma词根初始化的个数:', num_lemma)
    print('通过portor词根初始化的个数:', num_porter)
    print('通过同义词初始化的个数:', num_syn)
    print('通过lemma词根的同义词初始化的个数:', num_lemma_syn)
    print('通过portor词根的同义词初始化的个数:', num_porter_syn)
    print('通过标点初始化的个数:', num_symbols)
    print('通过数字初始化的个数:', num_numbers)
    print('匹配不到任何近义词而直接初始化的单词个数:', len(init_words))
    print('匹配不到任何近义词而直接初始化的单词:', init_words)
    print('匹配不到任何近义词而直接初始化的单词占词汇表的%.2f%%' % (100.0*len(init_words)/len(vocab)))
    print('匹配不到任何近义词而直接初始化的单词包含停止词的个数:', num_in_stopwords)
    print('匹配不到任何近义词而直接初始化的单词包含%.2f%%的停止词' % (100.0*num_in_stopwords/len(init_words)))

    with open(args.output_file, 'w', encoding='utf8') as fw:
        for key, val in vocab_vad_dict.items():
            fw.write(key + ' ' + ' '.join(val) + '\n')
