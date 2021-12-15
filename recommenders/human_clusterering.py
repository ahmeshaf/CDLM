from collections import defaultdict
from recommenders.incremental_clustering import IncrementalClusterer
from cross_encoder.models import Regressor
import random
import torch
import pickle
from recommenders.featurization import Featurizer

import textwrap
from collections import Counter
from copy import deepcopy


def has_upper(word):
    for w in word:
        if w.isupper():
            return True
    return False


def get_args_and_triggers(mentions_dicts):
    args = set()
    triggers = set()
    for men in mentions_dicts:
        args.update([m['tokens_str'] for m in men['arguments'] if has_upper(m['tokens_str'])])
        triggers.add(men['tokens_str'])

    return triggers, args


def print_event(event, mentions_dicts):
    print_lines = []
    print_lines.append('### Event Cluster: %s'%event)
    triggers, args = get_args_and_triggers(mentions_dicts)
    print_string = '<trigger>: ' + '|'.join(triggers) + '\n' + \
                '<args>: ' + '|'.join(args) + '\n' + '<description>:'

    print_lines.append(print_string)

    # print_description(mentions_dicts)

    for men in mentions_dicts:
        formatted_sentence = men['formatted_sentence']
        wrapper = textwrap.TextWrapper(width=50)
        word_list = wrapper.wrap(text=formatted_sentence)
        for element in word_list:
            print_lines.append(element)
        # print_lines.append('\n')

    print('\n'.join(print_lines))


def print_description(clus_mentions_dicts):
    for i, mentions_dicts  in enumerate(clus_mentions_dicts):
        triggers, args = get_args_and_triggers(mentions_dicts)
        print_string = '<trigger>: ' + '|'.join(triggers) + '\n'+ \
                       '   <args>: ' + '|'.join(args)

        print('%d: '%i + print_string)

def get_desctip_string(mentions_dicts):
    descrips = []
    for men in mentions_dicts:
        sentence_tokens = men['sentence_string'].split(' ')

        m_token_nums = men['tokens_number']

        m_tokens = [sentence_tokens[t] for t in m_token_nums]

        start = m_token_nums[0]
        end = m_token_nums[-1]

        if end < len(sentence_tokens) - 1:
            desc_tokens = sentence_tokens[start - 5: start] + m_tokens + \
                sentence_tokens[end+1: end + 6]
        else:
            desc_tokens = sentence_tokens[start - 12: start] + m_tokens

        desc_tokens = ['...'] + desc_tokens + ['...']

        descrips.append([desc_tokens, men['tokens_str']])

    return ' '.join(max(descrips, key=lambda x: len(x[0]))[0]), max(descrips, key=lambda x: len(x[0]))[1]

def print_description2(clus_mentions_dicts):
    for i, mentions_dicts  in enumerate(clus_mentions_dicts):
        descrip_string, men = get_desctip_string(mentions_dicts)
        triggers, args = get_args_and_triggers(mentions_dicts)

        print_string = '<trigger>: ' + '|'.join(triggers) + '\n' + \
                       '   <args>: ' + '|'.join(args) + '\n' + \
                       '   <description>: ' + descrip_string

        print('%d: ' % i + print_string)

state_stack = []

class HumanClusterer(IncrementalBatchTrainerLDC):
    def __init__(self, input_folder, contine_file_path, filter=None):
        # self.featurizer = featurizer
        self.comparisons = 0
        self.annotations = []
        # self.events = None
        self.event_counter = 0
        self.events_count = 0
        self.continue_file_path = contine_file_path
        self.input_folder = input_folder
        super(HumanClusterer, self).__init__(filter)

    def copy(self, inc_class):
        self.comparisons = inc_class.comparisons
        self.annotations = inc_class.annotations[:]
        # self.events = inc_class.events[:]
        self.event_counter = inc_class.event_counter
        self.events_count = inc_class.events_count
        self.continue_file_path = inc_class.continue_file_path
        self.input_folder = inc_class.input_folder
        super().copy(inc_class)

    def sample_candidates(self, event, mentions_dicts, token_threshold=1):
        possible_clus_type = list(self.get_possible_clus_from_type(mentions_dicts))
        possible_clus_lemma = list(self.get_possible_clus_lemma(mentions_dicts))

        possible_clus_tok = list(self.get_possible_clus_from_toks(mentions_dicts))

        tok_clus_counter = Counter(possible_clus_tok).most_common()
        tok_clus_thres = set([p[0] for p in tok_clus_counter if p[1] > token_threshold])

        possible_clus_tok = [p for p in possible_clus_tok if p in tok_clus_thres]
        possible_clus = [c for c in possible_clus_type + possible_clus_lemma + possible_clus_tok ]
        counter = Counter(possible_clus)
        possible_clus = [p[0] for p in counter.most_common() if event != p[0]]

        if self.filter != None:
            possible_clus = self.filter(event, possible_clus)

        return possible_clus

    def save_annotations(self, eve_link_map_arr_map):
        annotaions_path = self.input_folder + '/kb_linking.tab'
        with open(annotaions_path, 'w') as kf:
            kf.write('\t'.join(['kb_id', 'mention_id']))
            kf.write('\n')
            for i, cluster in enumerate(self.event_clusters.values()):
                kb_id = str(i)
                mentions = []
                for eve in cluster:
                    mentions.extend(eve_link_map_arr_map[eve])

                rows = [[kb_id, m] for m in mentions]
                str_rows = ['\t'.join(row) for row in rows]

                kf.write('\n'.join(str_rows))
                kf.write('\n')
        print('saved annotations at %s' % annotaions_path)

    def clustering_phase(self, eve_mention_map,  model, featurizer, cutoff=5, threshold=0., use_scorer=True):

        eve_within_doc_map = {eve: val['within_coref_chain'] for eve, val in eve_mention_map_test.items()}
        eve_link_map_arr_map = defaultdict(list)
        # print('mention: %d' %len(eve_mention_map))

        for men, eve in eve_within_doc_map.items():
            eve_link_map_arr_map[eve].append(men)

        all_events = sorted(list(eve_link_map_arr_map.keys()))
        # events_count = len(self.events)
        # event_counter = 0
        # random.shuffle(events)
        # self.clear_globals()
        training_tuples = []
        ranks = []
        topic_event = defaultdict(list)

        # self.events_count = 5

        for i in range(self.event_counter, self.events_count):
            event = all_events[i]
            topic_event[event.split('_')[0]].append(event)

        # for event in self.events:
        #     topic_event[event.split('_')[0]].append(event)

        topic_events = list(topic_event.values())

        for events in topic_events:
            random.shuffle(events)
            for i, event in enumerate(events):

                pickle.dump(self, open(self.continue_file_path, 'wb'))

                threshold_ = threshold*i/len(events)
                mentions = eve_link_map_arr_map[event]
                mentions_dicts = [eve_mention_map[m] for m in mentions]

                clus_candidates = self.sample_candidates(event, mentions_dicts)
                found_coreference = False
                if len(clus_candidates) > 0:
                    cluses_mentions_dicts = [
                        [
                            eve_mention_map[e] \
                            for eve in self.event_clusters[clus] for e in eve_link_map_arr_map[eve]
                        ] \
                        for clus in clus_candidates
                    ]

                    clus_features = [featurizer.featurize(mentions_dicts, clus_mentions_dicts)
                                     for clus_mentions_dicts in cluses_mentions_dicts]

                    # clus_features_standard = featurizer.standardize(clus_features)
                    with torch.no_grad():
                        clus_scores = model(torch.FloatTensor(clus_features))

                    possible_clus_f = list(zip(clus_candidates,
                                               clus_features,
                                               cluses_mentions_dicts,
                                               clus_scores))
                    if use_scorer:
                        possible_clus_f = sorted(possible_clus_f, key=lambda x: -x[-1])
                        possible_clus_f = [c for c in possible_clus_f if c[-1] > threshold_ and c[1][0] > 0]
                    else:
                        # lemma feature is not zero
                        # possible_clus_f = [c for c in possible_clus_f if c[1][0] > 0]
                        possible_clus_f = sorted(possible_clus_f, key=lambda x: -x[1][1])
                        pass

                    possible_clus_f = possible_clus_f[:cutoff]

                    if len(possible_clus_f) > 0:
                        # s_cluses, s_clus_features, s_clus_features_stand, _ = zip(*possible_clus_f)
                        s_cluses, s_clus_features, cluses_mentions_dicts, scores = zip(*possible_clus_f)

                        print('\n\nEvent %d/%d' %(self.event_counter + 1, self.events_count))
                        print_event(event, mentions_dicts)
                        print('\n### Candidates: ')
                        print_description2(cluses_mentions_dicts)
                        pass
                        clus = ''
                        choices = [str(i) for i in range(len(possible_clus_f))]
                        while True:
                            prompt = input("\n(e)xpand, s(k)ip, (m)erge, (g)o-back, (s)ave, e(x)it \n")
                            if prompt.lower() == 'e':
                                prompt = input("\nchoices: [" + ', '.join(choices) + ']\n')
                                if prompt in choices:
                                    print('\nCandidate: %s'%prompt)
                                    print_event(s_cluses[int(prompt)], cluses_mentions_dicts[int(prompt)])
                                else:
                                    print('Invalid choice. Try again')
                            elif prompt.lower() == 'm':
                                prompt = input("\nchoices: [" + ', '.join(choices) + ']\n')
                                if prompt in choices:
                                    clus = s_cluses[int(prompt)]
                                    break
                                else:
                                    print('Invalid choice. Try again')
                            elif prompt.lower() == 'g':
                                if len(state_stack) == 0:
                                    print("Cannot go back any further")
                                else:
                                    prev_state = state_stack.pop(-1)
                                    self.copy(prev_state)
                                    return self.clustering_phase(eve_mention_map,
                                                                 model,
                                                                 featurizer,
                                                                 cutoff=5,
                                                                 threshold=0,
                                                                 use_scorer=True)
                                # print('sorry, feature not implemented yet')
                            elif prompt.lower() == 's':
                                self.save_annotations(eve_link_map_arr_map)

                            elif prompt.lower() == 'x':
                                self.save_annotations(eve_link_map_arr_map)
                                exit()

                            else:
                                break

                        found_coreference = clus != ''




                        # check_input = input("wait")
                        # for rank, clus in enumerate(s_cluses):
                        #     self.comparisons+=1
                        #     if clus.split(':')[-1] == event.split(':')[-1] and clus != event:
                        #         found_coreference = True
                        #         ranks.append(rank+1)
                        #         break

                        if found_coreference:
                            if len(state_stack) > 5:
                                state_stack.pop(0)
                            state_stack.append(deepcopy(self))
                            # # if rank != 0:
                            # if len(s_cluses) > rank + 1:
                            #     neg_samples_features = s_clus_features[:rank] + s_clus_features[rank+1: rank+2]
                            #     pos_sample_feature = s_clus_features[rank]
                            #     training_tuples.append((pos_sample_feature, neg_samples_features))
                            # else:
                            #     neg_samples_features = s_clus_features[:rank]
                            #     pos_sample_feature = s_clus_features[rank]
                            #     training_tuples.append((pos_sample_feature, neg_samples_features))
                            #
                            # clus = s_cluses[rank]
                            self.merge_cluster(clus, event, mentions_dicts)
                            # eve_link_map_arr_map[clus].extend(eve_link_map_arr_map[event])

                if not found_coreference:
                    self.add_cluster(event, mentions_dicts)
                self.event_counter += 1
        # print_rank_estimates(ranks)

        self.save_annotations(eve_link_map_arr_map)
        if os.path.exists(self.continue_file_path):
            os.remove(self.continue_file_path)
        return ranks, training_tuples



from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer('english')

def add_stems(mentions):
    for mention in mentions:
        mention['stem'] = s_stemmer.stem(mention['lemma'].lower())
        sent_stems = []
        for w in mention['sentence_toks']:
            sent_stems.append(s_stemmer.stem(w.lower()))
        mention['sentence_stems'] = sent_stems


def run_experiment(eve_mention_map_test, input_folder):

    eve_within_doc_map = {eve: val['within_coref_chain'] for eve, val in eve_mention_map_test.items()}
    eve_link_map_arr_map = defaultdict(list)
    # print('mention: %d' %len(eve_mention_map))

    for men, eve in eve_within_doc_map.items():
        eve_link_map_arr_map[eve].append(men)

    # print('within events %d' %len(eve_link_map_arr_map))



    tfid_file_path = input_folder + 'tfidf.pkl'
    tfidf_vector_map = pickle.load(open(tfid_file_path, 'rb'))

    bert_vector_map = pickle.load(open(input_folder + 'context_vector_map.pkl', 'rb'))
    related_words_map = pickle.load(open(input_folder + 'related_words_map.pkl', 'rb'))

    featurizer = Featurizer(tfidf_vector_map, bert_vector_map, related_words_map)

    continue_file_path = './data/LDC_Eng/continue.pkl'
    continue_prompt = 'n'
    if os.path.exists(continue_file_path):
        continue_prompt = input("Continue? y/n")
        if continue_prompt == 'y':
            clusterer = pickle.load(open(continue_file_path, 'rb'))
    if continue_prompt != 'y':
        events = sorted(list(eve_link_map_arr_map.keys()))
        clusterer = HumanClusterer(input_folder, continue_file_path, ecb_filter)
        clusterer.events = events
        clusterer.events_count = len(events)

    clusterer.input_folder = input_folder
    clusterer.continue_file_path = continue_file_path
    # clusterer = cluster_class(ecb_filter)
    model = Regressor(featurizer.features_len())
    model.load_state_dict(torch.load('./model_chk_ldc/ranker_mrr.chk'))
    print(model.linear1.weight.detach().numpy())
    ranks, _ = clusterer.clustering_phase(eve_mention_map_test, model, featurizer, use_scorer=True , threshold=0.0)

    # print_rank_estimates(ranks)

if __name__=='__main__':
    input_folder = './data/LDC_Eng/'
    eve_mention_map_train, eve_mention_map_dev, eve_mention_map_test = get_eve_mention_maps(input_folder)
    all_mentions = {**eve_mention_map_train, **eve_mention_map_dev, **eve_mention_map_test}
    add_stems(all_mentions.values())

    run_experiment(eve_mention_map_dev, input_folder)
