import pandas as pd
import nltk
from nltk.corpus import stopwords
import joblib

pt_pos_tagger = joblib.load('pos_tag/POS_tagger_brill.pkl')

import warnings
warnings.filterwarnings('ignore')

EUROPEAN_PORTUGUESE_ONLY_WORDS = ['rotunda','telemóv', 'sanita', 'autocarro', 'passadeira', 
                         'sumo', 'empregado de mesa', 'elétrico', 'eléctrico',
                         'comboio', 'sandes', 'guarda-redes', 'desporto', 'cueca',
                         'rapariga','apelido','bicha','fixe','talho']

BRAZILIANS_PORTUGUESE_ONLY_WORDS = ['rotatória','celular','vaso sanitário','sanitário',
                                    'ônibus', 'faixa de pedestre', 'pedestre', 'suco',
                                    'garçom', 'garçonete', 'bonde', 'sanduíche', 'goleiro',
                                    'esporte', 'calcinha', 'menina', 'sobrenome', 'fila',
                                    'legal', 'açougue']

DEFINED_ARTICLES = ['o', 'a', 'os', 'as']
UNDEFINED_ARTICLES = ['um', 'uma', 'uns', 'umas']
RELATIVE_PERSONAL_PRONOUNS = ['mim','ti','si','nós','vós']
PERSONAL_PRONOUNS = ['ele', 'ela', 'eles', 'elas']
DEMONSTRATIVE_PRONOUNS = ['este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 
                          'essas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'aquilo',
                          'isto', 'isso']
UNDEFINED_PRONOUNS = ['outro', 'outra', 'outros', 'outras', 'outrém', 'outrem', 'outrora',
                      'algum', 'alguma', 'alguns', 'algumas', 'alguém', 'algo', 'algures',
                      'algures', 'alhures']
ADVERBS = ['onde', 'aqui', 'aí', 'ali', 'aquém', 'além', 'entre', 'antes', 'acolá']

def tag_sentence(sentence):
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence, language='portuguese')
    tagged = pt_pos_tagger.tag(sentence)
    sentence = "@@@".join([word + "_" + tag for word, tag in tagged])
    return sentence

def pt_pt_second_person_hints(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for word, tag in sentence:
        if tag == 'PROPESS' and word in ["tu", "te"]:
            count += 1
        if tag == 'PROADJ' and word in ["teu", "tua"]:
            count += 1
        if tag == "V" and word[-1] == "s":
            count += 1
    return count

def pt_br_second_person_hints(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for word, tag in sentence:
        if tag == 'PROPESS' and word in ["você", "lhe"]:
            count += 1
        if tag == 'PROADJ' and word in ["seu", "sua"]:
            count += 1
    return count

def tag_with_split_hyphen(sentence):
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence, language='portuguese')
    sentence = [word.split("-") for word in sentence]
    sentence = [word for sublist in sentence for word in sublist]
    return pt_pos_tagger.tag(sentence)

def pt_pt_pronoun_position_hints(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for word, tag in sentence:
        if '-' in word:
            tag_word = tag_with_split_hyphen(word)
            if tag_word[0][1] == 'V' and tag_word[1][1] == 'PROPESS' and tag_word[1][0] in ['me', 'te', 'se', 'nos', 'vos', 'lhe', 'lhes']:
                count += 1
    return count

def pt_br_pronoun_position_hints(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for i, (word, tag) in enumerate(sentence):
        if i < len(sentence) - 1:
            if tag == "PROPESS" and sentence[i+1][1] == "V":
                count += 1
    return count
    

def gerund_count(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for word, tag in sentence:
        if tag == "V" and word[-3:] == "ndo":
            count += 1
    return count

def a_plus_infinitive_count(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for i, (word, tag) in enumerate(sentence):
        if i < len(sentence) - 1:
            if tag == "PREP" and word == "a" and sentence[i+1][1] == "V" and sentence[i+1][0][-1] == "r":
                count += 1
    return count

def count_acute_accent(sentence):
    sentence = nltk.word_tokenize(sentence, language='portuguese')
    count = 0
    for word in sentence:
        for letter in word:
            if letter in ["á", "é", "í", "ó", "ú"]:
                count += 1
    return count

def count_circumflex_accent(sentence):
    sentence = nltk.word_tokenize(sentence, language='portuguese')
    count = 0
    for word in sentence:
        for letter in word:
            if letter in ["â", "ê", "î", "ô", "û"]:
                count += 1
    return count

def count_article_before_possessive_pronoun(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for i, (word, tag) in enumerate(sentence):
        if i < len(sentence) - 1:
            if tag == "ART" and sentence[i+1][1] == "PROADJ" and sentence[i+1][0] in ["meu", "teu", "seu", "nosso", "vosso", "seus", "suas"]:
                count += 1
    return count

def count_uncontracted_words(sentence):
    sentence = sentence.split("@@@")
    sentence = [word.split("_") for word in sentence]
    count = 0
    for i, (word, tag) in enumerate(sentence):
        if i < len(sentence) - 1:
            if tag == "PREP":
                if word == "a":
                    if sentence[i+1][1] == "ART" and sentence[i+1][0] in DEFINED_ARTICLES:
                        count += 1
                    elif sentence[i+1][1] == "PROADJ" and sentence[i+1][0] in DEMONSTRATIVE_PRONOUNS:
                        count += 1
                    elif sentence[i+1][1] == "ADV-KS-REL" and sentence[i+1][0] == "onde":
                        count += 1
                elif word == "com":
                    if sentence[i+1][1] == "PROPESS" and sentence[i+1][0] in RELATIVE_PERSONAL_PRONOUNS:
                        count += 1
                elif word == "de":
                    if sentence[i+1][1] == "ART" and sentence[i+1][0] in DEFINED_ARTICLES:
                        count += 1
                    elif sentence[i+1][1] == "ART" and sentence[i+1][0] in UNDEFINED_ARTICLES:
                        count += 1
                    elif sentence[i+1][1] == "PROPESS" and sentence[i+1][0] in PERSONAL_PRONOUNS:
                        count += 1
                    elif sentence[i+1][1] == "PROADJ" and sentence[i+1][0] in DEMONSTRATIVE_PRONOUNS:
                        count += 1
                    elif sentence[i+1][1] == "ADV" and sentence[i+1][0] == ADVERBS:
                        count += 1
                elif word == "em":
                    if sentence[i+1][1] == "ART" and sentence[i+1][0] in DEFINED_ARTICLES:
                        count += 1
                    elif sentence[i+1][1] == "ART" and sentence[i+1][0] in UNDEFINED_ARTICLES:
                        count += 1
                    elif sentence[i+1][1] == "PROPESS" and sentence[i+1][0] in PERSONAL_PRONOUNS:
                        count += 1
                    elif sentence[i+1][1] == "PROADJ" and (sentence[i+1][0] in DEMONSTRATIVE_PRONOUNS or sentence[i+1][0] in UNDEFINED_PRONOUNS):
                        count += 1
                    elif sentence[i+1][1] == "ADV" and sentence[i+1][0] == ADVERBS:
                        count += 1
    return count

def count_portuguese_words(sentence):
    sentence = sentence.lower()
    count = 0
    for word in EUROPEAN_PORTUGUESE_ONLY_WORDS:
        if word in sentence:
            count += 1
    return count

def count_brazilian_words(sentence):
    sentence = sentence.lower()
    count = 0
    for word in BRAZILIANS_PORTUGUESE_ONLY_WORDS:
        if word in sentence:
            count += 1
    return count

def get_features(df):
    df["POS-tagged"] = df["text"].apply(tag_sentence)
    df["pt_pt_second_person_hints"] = df["POS-tagged"].apply(pt_pt_second_person_hints)
    df["pt_pt_second_person_hints_bool"] = df["pt_pt_second_person_hints"].apply(lambda x: 1 if x > 0 else 0)
    df["pt_br_second_person_hints"] = df["POS-tagged"].apply(pt_br_second_person_hints)
    df["pt_br_second_person_hints_bool"] = df["pt_br_second_person_hints"].apply(lambda x: 1 if x > 0 else 0)
    
    df["pt_pt_pronoun_position_hints"] = df["POS-tagged"].apply(pt_pt_pronoun_position_hints)
    df["pt_pt_pronoun_position_hints_bool"] = df["pt_pt_pronoun_position_hints"].apply(lambda x: 1 if x > 0 else 0) 
    df["pt_br_pronoun_position_hints"] = df["POS-tagged"].apply(pt_br_pronoun_position_hints)
    df["pt_br_pronoun_position_hints_bool"] = df["pt_br_pronoun_position_hints"].apply(lambda x: 1 if x > 0 else 0)
    
    df["gerund_count"] = df["POS-tagged"].apply(gerund_count)
    df["gerund_count_bool"] = df["gerund_count"].apply(lambda x: 1 if x > 0 else 0)
    df["a_plus_infinitive_count"] = df["POS-tagged"].apply(a_plus_infinitive_count)
    df["a_plus_infinitive_count_bool"] = df["a_plus_infinitive_count"].apply(lambda x: 1 if x > 0 else 0)

    df["count_acute_accent"] = df["text"].apply(count_acute_accent)
    df["count_circumflex_accent"] = df["text"].apply(count_circumflex_accent)

    df["count_article_before_possessive_pronoun"] = df["POS-tagged"].apply(count_article_before_possessive_pronoun)
    df["count_article_before_possessive_pronoun_bool"] = df["count_article_before_possessive_pronoun"].apply(lambda x: 1 if x > 0 else 0)
    
    df["count_portuguese_words"] = df["text"].apply(count_portuguese_words)
    df["count_brazilian_words"] = df["text"].apply(count_brazilian_words)

    df["count_uncontracted_words"] = df["POS-tagged"].apply(count_uncontracted_words)
    df["count_uncontracted_words_bool"] = df["count_uncontracted_words"].apply(lambda x: 1 if x > 0 else 0)
    return df