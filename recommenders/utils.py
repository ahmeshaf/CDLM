import os
import pickle
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import lin_thesaurus as thes
from tqdm import tqdm

eng_stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
             "your", "yours", "yourself", "yourselves", "he", "him", "his",
             "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which",
             "who", "whom", "this", "that", "these", "those", "am", "is", "are",
             "was", "were", "be", "been", "being", "have", "has", "had", "having",
             "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
             "or", "because", "as", "until", "while", "of", "at", "by", "for",
             "with", "about", "against", "between", "into", "through", "during",
             "before", "after", "above", "below", "to", "from", "up", "down", "in",
             "out", "on", "off", "over", "under", "again", "further", "then", "once",
             "here", "there", "when", "where", "why", "how", "all", "any", "both",
             "each", "few", "more", "most", "other", "some", "such", "no", "nor",
             "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
             "can", "will", "just", "don", "should", "now"}

def generate_tf_idf_doc(input_folder, rsd_map):
    tfid_file_path = input_folder + '/tfidf.pkl'
    #     if os.path.exists(tfid_file_path):
    #         return pickle.load(open(tfid_file_path, 'rb'))

    doc_ids, doc_texts = zip(*rsd_map.items())

    tfidf_vectorizer = TfidfVectorizer(stop_words=eng_stopwords)
    vectors = tfidf_vectorizer.fit_transform(doc_texts)

    tfidf_vector_map = {doc: vector for doc, vector in zip(doc_ids, vectors)}

    pickle.dump(tfidf_vector_map, open(tfid_file_path, 'wb'))

    return tfidf_vector_map

def order_mapping(pred_mapping, gold_mapping):
    pred_mapping = dict(pred_mapping)
    new_dict = {}

    for tup in dict(gold_mapping).items():
        new_dict[tup[0]] = pred_mapping[tup[0]]

    return list(new_dict.items())

def read_csv(csv_file, delim=','):

    with open(csv_file) as cf:
        lines = [line.strip('\n').split(delim) for line in cf.readlines()]
        column_names = lines[0]
        rows_map = [{c: r for c, r in zip(column_names, row)} for row in lines[1:]]
    return rows_map

def get_sentence(text, start, end):
    while start != 0 and text[start] != '\n':
        start-=1
    while end != len(text) and text[end] != '\n':
        end+=1

    sentence_string = text[start+1: end]
    return sentence_string

def run_scorer(params):
    perl_script = subprocess.check_output(params)
    #     perl_script = subprocess.check_output()
    perl_script = perl_script.decode("utf-8")
    index = perl_script.find("====== TOTALS =======")
    return perl_script[index:]


def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


def generate_results(key_file, response_file):
    params = ["perl", "./scorer/scorer.pl", "bcub", key_file, response_file]
    # print("BCUB SCORE")
    # print(run_scorer(params))

    params[2] = "muc"
    print("MUC SCORE")
    print(run_scorer(params))


def get_muc_recall(key_file, response_file, script_path = "./scorer/scorer.pl"):
    params = ["perl", script_path, "muc", key_file, response_file]
    score_desc = run_scorer(params)
    print(score_desc)

    tag1 = 'Coreference:'
    tag2 = 'Recall:'
    index1 = score_desc.index(tag1) + len(tag1)
    recall_line = score_desc[index1:]
    index2 = recall_line.index(tag2) + len(tag2)
    recall_line = recall_line[index2:]
    recall_score = float(recall_line[recall_line.index(')') + 1: recall_line.index('%')].strip())

    links = int(recall_line[:recall_line.index('/')].strip().strip('('))

    return recall_score/100, links





def generate_key_file(coref_map_tuples, name, out_dir, out_file_path):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    clus_to_int = {}
    clus_number = 0
    with open(out_file_path, 'w') as of:
        of.write("#begin document (%s);\n" % name)
        for i, map_ in enumerate(coref_map_tuples):
            en_id = map_[0]
            clus_id = map_[1]
            if clus_id in clus_to_int:
                clus_int = clus_to_int[clus_id]
            else:
                clus_to_int[clus_id] = clus_number
                clus_number += 1
                clus_int = clus_to_int[clus_id]
            of.write("%s\t0\t%d\t%s\t(%d)\n" % (name, i, en_id, clus_int))
        of.write("#end document\n")

def get_eve_mention_maps(input_folder):
    eve_maps = []
    corpuses = ['Train', 'Dev', 'Test']
    for corpus in corpuses:
        p_file = input_folder + '/eve_mention_map_%s.pkl' %corpus
        if os.path.exists(p_file):
            eve_maps.append(pickle.load(open(p_file, 'rb')))
    return eve_maps

def ecb_filter(event, cluses):
    eve_topic = event.split('_')[0]
    return [clus for clus in cluses if clus.split('_')[0] == eve_topic]

def get_unique_arg_tokens(arguments):
    start_end_set = set()
    for arg in arguments:
        tn = arg['tokens_number']
        start_end_set.add('_'.join([str(tn[0]), str(tn[-1])]))
    return [[int(i) for i in ss.split('_')] for ss in start_end_set]

def add_formatted_sentence(mentions):
    for mention in mentions:
        sentence_toks = mention['sentence_string'].split(' ')
        eve_men_token_numbers = mention['tokens_number']
        sentence_toks[eve_men_token_numbers[0]] = '{' + sentence_toks[int(eve_men_token_numbers[0])]
        sentence_toks[eve_men_token_numbers[-1]] = sentence_toks[int(eve_men_token_numbers[-1])] + '}'

        for arg in mention['arguments']:
            if arg['sent_id'] == mention['sent_id']:
                arg_token_numbers = arg['tokens_number']
                sentence_toks[arg_token_numbers[0]] = '[' + sentence_toks[arg_token_numbers[0]]
                sentence_toks[arg_token_numbers[-1]] = sentence_toks[arg_token_numbers[-1]] + ']'

        # arg_start_end_tokens = get_unique_arg_tokens(mention['arguments'])
        # for arg_token_numbers in arg_start_end_tokens:
        #     sentence_toks[arg_token_numbers[0]] = '[' + sentence_toks[arg_token_numbers[0]]
        #     sentence_toks[arg_token_numbers[-1]] = sentence_toks[arg_token_numbers[-1]] + ']'

        mention['formatted_sentence'] = ' '.join(sentence_toks)


def get_lin_related_words(lemma):
    n_key = 'simN.lsp'
    v_key = 'simV.lsp'

    # noun_rw = set()
    # verb_rw = set()

    l_related_dict = dict(thes.scored_synonyms(lemma))
    noun_rw = list(l_related_dict[n_key])
    verb_rw = list(l_related_dict[v_key])

    return {'verbs': verb_rw, 'nouns': noun_rw}

def get_derivationally_related_verbs(nlp, word):
    if word.strip() != '':
        w = nlp(word)[0]
        lemma_set = set([w.text])
        if w.pos_ != 'VERB':
            non_verb_synsets = wn.synsets(word)
            if len(non_verb_synsets) > 0:
                syn = non_verb_synsets[0]
                lemmas = syn.lemmas()
                for lemma in lemmas:
                    deriv_related_forms = lemma.derivationally_related_forms()
                    for form in deriv_related_forms:
                        if form.synset().pos() == 'v':
                            lemma_set.add(form._name)
        return list(lemma_set)
    return set()

def add_spacy_features(nlp, eve_mention_map):
    for mention in eve_mention_map.values():
        sentence_nlp = nlp(mention['sentence_string'])
        mention_nlp = nlp(mention['tokens_str'])

        lemma = ' '.join([t.lemma_ for t in mention_nlp if not t.is_punct and not t.is_punct])
        lemma_vector = mention_nlp.vector

        for tok in mention_nlp:
            if not tok.is_stop and not tok.is_punct:
                lemma = tok.head.lemma_
                lemma_vector = tok.vector

        tokens = [tok.lemma_ for tok in sentence_nlp if not tok.is_stop and not tok.is_punct]
        mention['lemma'] = lemma
        mention['lemma_set'] = get_derivationally_related_verbs(nlp, lemma)
        mention['lemma_vector'] = lemma_vector
        mention['sentence_toks'] = tokens
        mention['sentence_vector'] = sentence_nlp.vector

def generate_related_words_map(mention_dicts):
    related_words_map = {}
    for i, mention in tqdm(enumerate(mention_dicts)):
        lemma_set = mention['lemma_set']
        for lemma in lemma_set:
            if lemma not in related_words_map:
                related_words_map[lemma] = get_lin_related_words(lemma)
    return related_words_map