import pickle
import torch
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk.corpus import lin_thesaurus as thes
from scipy.spatial.distance import cosine

class Featurizer():
    def __init__(self, tfidf_path, related_words_path, context_vec_path, sent2ent_path):
        self.related_word_map = pickle.load(open(related_words_path, 'rb'))
        self.context_vector_map = pickle.load(open(context_vec_path, 'rb'))
        self.tfidf_map = pickle.load(open(tfidf_path, 'rb'))
        self.sent2ent_map = pickle.load(open(sent2ent_path, 'rb'))

        self.doc2ent_map = defaultdict(list)
        for ents in self.sent2ent_map.values():
            for ent in ents:
                doc_id = ent['doc_id']
                self.doc2ent_map[doc_id].append(ent)

        self.feature_functions = [self.lemma,
                                  self.lemma_ngrams,
                                  self.related,
                                  self.cdlm_context,
                                  self.coref_args,
                                  self.tfidf]

    def get_feature_len(self):
        return sum([func(None, None, num_feature=True) for func in self.feature_functions])

    def get_men_id(self, men):
        return "_".join([men['doc_id'].split('.')[0], men['m_id']])

    def js(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        if len(set1) == 0 or len(set2) == 0:
            return 0.
        return len(set1.intersection(set2))/len(set1.union(set2))

    def featurize(self, batch):
        batch_features = []
        for item in batch:
            men1, men2 = item
            item_features = []
            for func in self.feature_functions:
                features = func(men1, men2)
                if not isinstance(features, list):
                    features = [features]
                item_features.extend(features)

            batch_features.append(item_features)
        return torch.tensor(batch_features)

    def lemma(self, mentions1, mentions2, num_feature=False):
        if num_feature:
            return 1
        lemma1 = mentions1['lemmas']
        lemma2 = mentions2['lemmas']
        return lemma1 == lemma2

    def lemma_ngrams(self, men1, men2, num_feature=False):
        if num_feature:
            return 1
        lemma1 = men1['tokens']
        lemma1_ngrams = [lemma1[i:i + 3] for i in range(len(lemma1) - 2)]
        lemma2 = men2['tokens']
        lemma2_ngrams = [lemma2[i:i + 3] for i in range(len(lemma1) - 2)]
        return self.js(lemma1_ngrams, lemma2_ngrams)

    def related(self, men1, men2, top_n=50, num_feature=False):
        if num_feature:
            return 3
        lemma_set1 = [men1['lemmas']]
        lemma_set2 = [men2['lemmas']]

        rw1_v = set([w[0] for lemma in lemma_set1 \
                     for w in self.related_word_map[lemma]['verbs'][:top_n]])
        rw2_v = set([w[1] for lemma in lemma_set2 \
                     for w in self.related_word_map[lemma]['verbs'][:top_n]])

        rw1_n = set([w[0] for lemma in lemma_set1 \
                     for w in self.related_word_map[lemma]['nouns'][:top_n]])
        rw2_n = set([w[0] for lemma in lemma_set2 \
                     for w in self.related_word_map[lemma]['nouns'][:top_n]])

        overlap_feature = 0.
        w2_contains_w1_feature = 0.
        w1_contains_w2_feature = 0
        if len(rw1_v) > 0 and len(rw2_v) > 0:
            overlap_feature = len(rw1_v.intersection(rw2_v)) / len(rw1_v)
            w2_contains_w1_feature = int(len(rw2_v.intersection(lemma_set1)) > 0)
            w1_contains_w2_feature = int(len(rw1_v.intersection(lemma_set2)) > 0)
        else:
            all_w1 = set(list(rw1_v) + list(rw1_n))
            all_w2 = set(list(rw2_v) + list(rw2_n))

            if len(all_w1) > 0 and len(all_w2) > 0:
                overlap_feature = len(all_w1.intersection(all_w2)) / len(all_w1)
                w2_contains_w1_feature = int(len(all_w2.intersection(lemma_set1)) > 0)
                w1_contains_w2_feature = int(len(all_w1.intersection(lemma_set2)) > 0)

        return [overlap_feature, w2_contains_w1_feature, w1_contains_w2_feature]

    def cdlm_context(self, men1, men2, num_feature=False, turnoff=True):
        if num_feature:
            return 1

        if turnoff:
            return 0.

        men1_id = self.get_men_id(men1)
        men2_id = self.get_men_id(men2)

        return 1 - cosine(self.context_vector_map[men1_id],
                      self.context_vector_map[men2_id])

    def coref_args(self, men1, men2, num_feature=False):
        if num_feature:
            return 2
        sent_id1 = men1['doc_id'] + '_' + men1['sentence_id']
        sent_id2 = men2['doc_id'] + '_' + men2['sentence_id']

        args1 = self.sent2ent_map[sent_id1]
        clusids1 = [arg['cluster_id'] for arg in args1]
        args2 = self.sent2ent_map[sent_id2]
        clusids2 = [arg['cluster_id'] for arg in args2]
        coref_sent_feature = self.js(clusids1, clusids2)

        doc1 = men1['doc_id']
        doc2 = men2['doc_id']

        args1 = self.doc2ent_map[doc1]
        args2 = self.doc2ent_map[doc2]

        clusids1 = [arg['cluster_id'] for arg in args1]
        clusids2 = [arg['cluster_id'] for arg in args2]

        coref_doc_feature = self.js(clusids1, clusids2)

        return [coref_sent_feature, coref_doc_feature]

    def tfidf(self, men1, men2, num_feature=False):
        if num_feature:
            return 1
        doc_id1 = men1['doc_id']
        doc_id2 = men2['doc_id']
        return 1 - cosine(self.tfidf_map[doc_id1].toarray(),
                      self.tfidf_map[doc_id2].toarray())


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


def generate_related_words_map(mention_dicts):
    related_words_map = {}
    for i, mention in tqdm(enumerate(mention_dicts)):
        lemma_set = mention['lemma_set']
        for lemma in lemma_set:
            if lemma not in related_words_map:
                related_words_map[lemma] = get_lin_related_words(lemma)
    return related_words_map


def generate_related_words(all_eve_mentions, out_file_path):
    related_words_map = {}
    for mention in tqdm(all_eve_mentions, total=len(all_eve_mentions)):
        # mention = all_eve_mentions[i]
        lemma_set = mention['lemmas']

        if not isinstance(lemma_set, list):
            lemma_set = [lemma_set]

        for lemma in lemma_set:
            if lemma not in related_words_map:
                related_words_map[lemma] = get_lin_related_words(lemma)
    pickle.dump(related_words_map, open(out_file_path, 'wb'))


def generate_tfidf_map(all_data, out_file_path):
    stemmer = SnowballStemmer('english')
    doc2words = defaultdict(list)
    for doc, tokens in all_data.items():
        for tok in tokens:
            tok=tok[2]
            if tok.lower() != 'http' and len(tok) != 1:
                doc2words[doc].append(stemmer.stem(tok.lower()))

    docs, doc_words  = zip(*doc2words.items())
    doc_texts = [' '.join(words) for words in doc_words]
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    vectors = tfidf_vectorizer.fit_transform(doc_texts)

    tfidf_vector_map = {doc: vector for doc, vector in zip(docs, vectors)}

    pickle.dump(tfidf_vector_map, open(out_file_path, 'wb'))


def generate_sent_2_ent(ent_data, out_file_path):
    sent2ent = defaultdict(list)
    for ent in ent_data:
        doc = ent["doc_id"]
        sent_id = ent["sentence_id"]
        sent_id = doc + '_' + sent_id
        sent2ent[sent_id].append(ent)
    pickle.dump(sent2ent, open(out_file_path, 'wb'))

if __name__=='__main__':
    data_path = './data/ecb/mentions/'
    out_folder = './data/ecb/cdlm/'
    splits = ['dev', 'train', 'test']

    all_data = {}
    all_events = []
    all_entities = []
    sent_to_ent = {}
    for split in splits:
        split_path = data_path + '/%s.json'%split
        split_data = json.load(open(split_path))
        all_data = {**all_data, **split_data}

        eve_split_path = data_path + '/%s_events.json'%split
        eve_data = json.load(open(eve_split_path))
        all_events.extend(eve_data)

        ent_data = json.load(open(data_path + '/%s_entities.json'%split))
        all_entities.extend(ent_data)

    generate_sent_2_ent(all_entities, out_folder + '/sent2ent.pkl')

    # generate_tfidf_map(all_data, out_folder + '/tfidf.pkl')
    # generate_related_words(all_events, out_folder + '/related_words_map.pkl')

