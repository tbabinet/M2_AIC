import nltk
import string, pprint, os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

DOC_DIR="TEXTES"
text_list=[]
text_names=[]

def get_tokens(file):
    with open(file, 'r') as d:
        text = d.read()
        tokens = nltk.word_tokenize(clean_text(text))
        return tokens

def get_most_common_tokens(tokens, num):
    count = Counter(tokens)
    return count.most_common(num)

def clean_text(text):
        lowers = text.lower() #lower case for everyone
        #remove the punctuation using the character deletion step of translate
        punct_killer = str.maketrans('', '', string.punctuation)
        no_punctuation = lowers.translate(punct_killer)
        return no_punctuation

def get_text(file):
    with open(file, 'r') as d:
        text = d.read()
        return clean_text(text)
    
def tokenize(text):
    return nltk.word_tokenize(text)
    


def create_tfidf(dir):
    text_list = []
    text_name = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                #print("treating "+file)
                file_path = subdir + os.path.sep + file
                #text_dict[file] = get_text(file_path)
                text_list.append(get_text(file_path))
                text_names.append(file)
    return text_list, text_name

def get_similarity(text1, text2, v):
    t1 = v.transform([text1])
    t2 = v.transform([text2])
    return cosine_similarity(t1,t2)

    
if __name__ == '__main__':
    #tokens = get_tokens('TEXTES/rd_12_psg_8_10.txt')
    create_tfidf("TEXTES")
    v = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidfs = v.fit_transform(text_list)
    print(v.vocabulary_['located'])
    print(tfidfs[1143,2960])                        

    print(text_names.index("rd_16_psg_38_40.txt"))

    str1 = 'this sentence has unseen text such as computer but also king lord juliet'
    str2 = 'i love computer text'
    print(get_similarity(str1,str2,v))
