import nltk
from nltk.corpus import stopwords 
# from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
stop = set(stopwords.words('english')) 
print(stop)

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')

stemmer = nltk.stem.PorterStemmer()


def stem(word):
    return stemmer.stem(word)


def clean_sentence(text, stemming=False):
    text = text.lower()
    for token in punct:
        text = text.replace(token, "")
    words = text.split()
    if stemming:
        stemmed_words = []
        for w in words:
            stemmed_words.append(stem(w))
        words = stemmed_words
    filter_words = [w for w in words if w not in stopwords.words('english')]
    return filter_words


def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)
