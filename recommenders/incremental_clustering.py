from collections import defaultdict, Counter
import random
random.seed(42)
import pickle
import numpy as np
import json
from tqdm import tqdm
from recommenders.utils import order_mapping, generate_key_file, get_muc_recall
import os

class IncrementalClusterer():
    """
    The class to handle, Querying and Merging of Event clusters
    """
    def __init__(self, score_data_path):
        score_data = pickle.load(open(score_data_path, 'rb'))
        self.pair_score_map = {}
        self.topic_mention_map = score_data[1]
        for item in score_data[0]:
            p = item[0]
            s = item[1]
            self.pair_score_map["%s_%s"%(str(p[0]), str(p[1]))] = s
        self.event_clusters = defaultdict(set)
        self.topic_to_clus = defaultdict(set)
        self.lemma_clus_map = defaultdict(set)
        self.tok_clus_map = defaultdict(set)

    def avg_mention_score(self, eve_mentions, clus_mentions):
        all_scores = []
        for m1 in eve_mentions:
            m_id1 = self.topic_mention_map['_'.join([m1['doc_id'].split('.')[0], m1['m_id']])]
            for m2 in clus_mentions:
                m_id2 = self.topic_mention_map['_'.join([m2['doc_id'].split('.')[0], m2['m_id']])]
                p_id = "%s_%s" % (str(m_id1), str(m_id2))
                all_scores.append(self.pair_score_map[p_id])
        return np.max(all_scores)

    def clear_globals(self):
        self.event_clusters.clear()
        self.topic_to_clus.clear()
        self.lemma_clus_map.clear()
        self.tok_clus_map.clear()

    def get_possible_clus_lemma(self, mentions_dicts):
        lemmas = set([m['lemmas'] for m in mentions_dicts])
        possible_clus = set()
        for lemma in lemmas:
            possible_clus.update(self.lemma_clus_map[lemma])
        return possible_clus

    def get_possible_clus_from_toks(self, mentions_dicts):
        all_tokens = [tok.lower() for mention in mentions_dicts for tok in mention['sentence_toks']]
        possible_clus = []
        for tok in all_tokens:
            cluses = self.tok_clus_map[tok]
            possible_clus.extend(cluses)
        return possible_clus

    def get_possible_clus_topic(self, mentions_dicts):
        topics = set([m['topic'] for m in mentions_dicts])
        possible_clus = set()
        for topic in topics:
            possible_clus.update(self.topic_to_clus[topic])
        return possible_clus

    def merge_cluster(self, clus, event, mentions_dicts):
        lemmas = set([m['lemmas'] for m in mentions_dicts])
        for lemma in lemmas:
            self.lemma_clus_map[lemma].add(clus)
            if event in self.lemma_clus_map[lemma]:
                self.lemma_clus_map[lemma].remove(event)
        all_tokens = [tok.lower() for mention in mentions_dicts for tok in mention['sentence_toks']]
        for tok in all_tokens:
            self.tok_clus_map[tok].add(clus)
            if event in self.tok_clus_map[tok]:
                self.tok_clus_map[tok].remove(event)
        topics = set([m['topic'] for m in mentions_dicts])
        for topic in topics:
            self.topic_to_clus[topic].add(clus)
            if event in self.topic_to_clus[topic]:
                self.topic_to_clus[topic].remove(event)
        self.event_clusters[clus].add(event)

    def merge_clusters(self, main_clus, cluses, clus_mentions):
        for clus, mentions_dicts in zip(cluses, clus_mentions):
            self.merge_cluster(main_clus, clus, mentions_dicts)

    def add_cluster(self, eve, mentions_dicts):
        topics = set([m['topic'] for m in mentions_dicts])
        for topic in topics:
            self.topic_to_clus[topic].add(eve)
        lemmas = set([m['lemmas'] for m in mentions_dicts])
        for lemma in lemmas:
            self.lemma_clus_map[lemma].add(eve)
        all_tokens = [tok.lower() for mention in mentions_dicts for tok in mention['sentence_toks']]
        for tok in all_tokens:
            self.tok_clus_map[tok].add(eve)
        self.event_clusters[eve].add(eve)

    def from_same_doc(self, clus, event):
        events_in_cluster = self.event_clusters[clus]
        docs = set([e.split(':')[0] for e in events_in_cluster])
        eve_doc = event.split(':')[0]
        return eve_doc in docs

    def sample_candidates(self, event, mentions_dicts):
        possible_clus_lemma = list(self.get_possible_clus_lemma(mentions_dicts))
        possible_clus_tok = list(self.get_possible_clus_from_toks(mentions_dicts))
        tok_clus_counter = Counter(possible_clus_tok).most_common()
        tok_clus_thres = set([p[0] for p in tok_clus_counter if p[1] > 0])
        possible_clus_tok = [p for p in possible_clus_tok if p in tok_clus_thres]
        possible_clus = [c for c in possible_clus_lemma + possible_clus_tok if not self.from_same_doc(c, event)]
        counter = Counter(possible_clus)
        possible_clus = [p[0] for p in counter.most_common() if event != p[0]]
        possible_clus = [p for p in possible_clus if event.split('_')[0] == p.split('_')[0]]
        return list(possible_clus)

    def clustering_phase(self, eve_mention_map, cutoff=100, threshold=0., use_scorer=True):
        eve_within_doc_map = {eve: val['doc_id'] + ':' + str(val['cluster_id']) for eve, val in eve_mention_map.items()}
        eve_link_map_arr_map = defaultdict(list)
        print('mention: %d' %len(eve_mention_map))
        for men, eve in eve_within_doc_map.items():
            eve_link_map_arr_map[eve].append(men)
        events = sorted(list(eve_link_map_arr_map.keys()))
        self.clear_globals()
        ranks = []
        recommendations = []
        topic_event = defaultdict(list)
        for event in events:
            topic_event[event.split('_')[0][:4]].append(event)
        self.comparisons = 0
        for topic, events in tqdm(topic_event.items(), desc='clustering by topic'):
            events = sorted(events)
            for i, event in enumerate(events):
                threshold_ = threshold
                mentions = eve_link_map_arr_map[event]
                mentions_dicts = [eve_mention_map[m] for m in mentions]
                clus_candidates = self.sample_candidates(event, mentions_dicts)
                found_coreference = False
                if len(clus_candidates) > 0:
                    cluses_mentions_dicts = [
                        [
                            eve_mention_map[e] \
                            for e in eve_link_map_arr_map[clus]
                        ] \
                        for clus in clus_candidates
                    ]

                    clus_scores = [self.avg_mention_score(mentions_dicts, clus_mentions_dicts)
                                     for clus_mentions_dicts in cluses_mentions_dicts]
                    possible_clus_f = list(zip(clus_candidates,
                                               cluses_mentions_dicts,
                                               clus_scores))
                    if use_scorer:
                        possible_clus_f = sorted(possible_clus_f, key=lambda x: -x[-1])
                        possible_clus_f = [c for c in possible_clus_f if c[-1] > threshold_]
                    if len(possible_clus_f) > 0:
                        my_cutoff = int(cutoff)
                        if isinstance(cutoff, float):
                            possible_clus_f_cutoff = possible_clus_f[:my_cutoff]
                            s_cluses, s_clus_dicts, _ = zip(*possible_clus_f_cutoff)
                            cut_clus_found = False
                            for clus in s_cluses:
                                if clus.split(':')[-1] == event.split(':')[-1] and clus != event:
                                    cut_clus_found = True
                            if not cut_clus_found:
                                my_cutoff = my_cutoff + random.randint(0, 1)
                        possible_clus_f = possible_clus_f[:my_cutoff]
                        s_cluses, s_clus_dicts, _ = zip(*possible_clus_f)
                        coref_clus = []
                        clus_ranks = []
                        clus_mentions = []
                        self.comparisons += len(possible_clus_f)
                        for rank, clus in enumerate(s_cluses):
                            if clus.split(':')[-1] == event.split(':')[-1] and clus != event:
                                found_coreference = True
                                coref_clus.append(clus)
                                clus_mentions.append(s_clus_dicts[rank])
                                clus_ranks.append(rank+1)
                        recommendations.append(int(found_coreference))
                        if found_coreference:
                            main_clus = coref_clus[0]
                            if len(coref_clus) > 1:
                                self.merge_clusters(main_clus, coref_clus[1:], clus_mentions[1:])
                                for clus in coref_clus[1:]:
                                    eve_link_map_arr_map[main_clus].extend(eve_link_map_arr_map[clus])
                                    eve_link_map_arr_map.pop(clus)
                            ranks.append(clus_ranks[0])
                            self.merge_cluster(main_clus, event, mentions_dicts)
                            eve_link_map_arr_map[main_clus].extend(eve_link_map_arr_map[event])
                if not found_coreference:
                    self.add_cluster(event, mentions_dicts)
        print(sum(recommendations)/len(recommendations))
        return ranks, recommendations


def comparison_anaylis(input_folder, split, sent2tok_map, score_path, ks, threshold=0.):
    eve_mentions_path = input_folder + '/mentions/%s_events.json'%split
    events = json.load(open(eve_mentions_path))
    eve_mention_map = {}
    for eve in events:
        doc_id = eve['doc_id'].split('.')[0]
        m_id = eve['m_id']
        eve['cluster_id'] = str(eve['cluster_id'])
        eve['sentence_toks'] = sent2tok_map[eve['doc_id'] + '_' + str(eve['sentence_id'])]
        mention_id = '_'.join([doc_id, str(eve['sentence_id']), str(eve['sentence_numbers'][0]), str(eve['sentence_numbers'][-1])])
        eve_mention_map[mention_id] = eve

    key_file_dir = input_folder + '/key_files/'
    if not os.path.exists(key_file_dir):
        os.makedirs(key_file_dir)
    key_file = key_file_dir + 'eve_%s.txt' % split
    eve_within_doc_map = {eve: val['doc_id'] + ':' + val['cluster_id'] for eve, val in eve_mention_map.items()}
    eve_link_map_cross = {val: val.strip().split(':')[-1] for val in eve_within_doc_map.values()}
    print('clusters %d' % len(eve_link_map_cross.values()))
    generate_key_file(eve_link_map_cross.items(), 'ECB_%s' % split, key_file_dir, key_file)
    clusterer = IncrementalClusterer(score_path)
    comps = []
    recalls = []
    Ps = []
    for k in ks:
        ranks, recomms = clusterer.clustering_phase(eve_mention_map, k, threshold=threshold, use_scorer=True)
        total_comparisons = clusterer.comparisons
        clusterer.comparisons = 0
        event_clusters = clusterer.event_clusters
        cluster_map = {}
        for cl, eves in event_clusters.items():
            for eve in list(eves):
                cluster_map[eve] = cl
        predicted_eve_cross_tups = order_mapping(cluster_map, eve_link_map_cross)
        key_file_pred = key_file_dir + 'eve_scorer_%s.txt' % split
        generate_key_file(predicted_eve_cross_tups, 'ECB_%s' % split, key_file_dir, key_file_pred)
        muc_recall, muc_links = get_muc_recall(key_file, key_file_pred, script_path='../scorer/scorer.pl')
        P = sum(recomms) / len(recomms)
        comps.append(total_comparisons)
        recalls.append(muc_recall)
        Ps.append(P)
    results = list(zip(ks, recalls, Ps, comps))
    print(results)
    return results


if __name__ == '__main__':
    # dev_eve_path = './data/ecb/mentions/dev_events.json'
    # dev_events = json.load(open(dev_eve_path))
    #
    cdlm_score_path = './data/ecb/cdlm/cdlm/cdlm_score_data_test.pkl'
    regressor_score_path = './data/ecb/cdlm/regressor/regressor_score_data_test.pkl'
    # regressor_score_path_Dev = './data/ecb/cdlm/regressor_score_data_dev.pkl'
    #
    # mention_map_dev = {}
    # for eve in dev_events:
    #     doc_id = eve['doc_id'].split('.')[0]
    #     m_id = eve['m_id']
    #     eve['cluster_id'] = str(eve['cluster_id'])
    #     mention_id = doc_id + '_' + m_id
    #     mention_map_dev[mention_id] = eve
    #
    # clustering = IncrementalClusterer(cdlm_score_path)
    # ranks, recommendations = clustering.clustering_phase(mention_map_dev, 10, 0.0, use_scorer=True)
    sent2tok_map = pickle.load(open('./data/ecb/cdlm/sent2tok_map.pkl', 'rb'))
    # comparison_anaylis('./data/ecb/', 'test', sent2tok_map, r2tok_map.pkl', 'rb'))
    # comparison_anaylis('./data/ecb/', 'dev', sent2tok_map, regressor_score_path_Dev, [10], 0.371)
    comparison_anaylis('./data/ecb/', 'test', sent2tok_map, regressor_score_path, [5], 0.369)