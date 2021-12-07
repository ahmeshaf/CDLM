import glob
from bs4 import BeautifulSoup as bs
import os
from collections import defaultdict
import json
import pickle
import spacy
from recommenders.utils import generate_tf_idf_doc
from recommenders.featurization import generate_related_words

def read_csv(file_path, delim='\t', columns=True):
    with open(file_path) as fp:
        rows = [line.strip().split(delim) for line in fp.readlines()]
    if columns:
        dict_rows = []
        column_names = rows[0]
        for row in rows[1:]:
            row_dict = {c:r for r, c in zip(row, column_names)}
            dict_rows.append(row_dict)
        return dict_rows
    else:
        return rows


def aida_ltf2json(ltfxml_dir, output_folder, split_name):
    doc_dict = defaultdict(list)
    for file_path in glob.glob(ltfxml_dir + '/*.xml'):
        with open(file_path) as fp:
            file_bs = bs(fp.read())
            doc = file_bs.doc
            doc_id = doc['id']
            sentences = doc.find_all('seg')
            tok_counter = 1
            for sent_id, sentence in enumerate(sentences):
                tokens = sentence.find_all('token')
                for tok in tokens:
                    tok_text = tok.text
                    doc_dict[doc_id].append([sent_id, tok_counter, tok_text, False])
                    tok_counter+=1
    json.dump(doc_dict, open(output_folder + '/%s.json'%split_name, 'w'), indent=2)


def get_doc_sent_map(ltfxml_dir):
    doc_sent_dict = {}
    for file_path in glob.glob(ltfxml_dir + '/*.xml'):
        with open(file_path) as fp:
            file_bs = bs(fp.read())
            doc = file_bs.doc
            doc_id = doc['id']
            sentences = doc.find_all('seg')
            tok_counter = 1
            for sent_id, sentence in enumerate(sentences):
                sent_start_char = sentence['start_char']
                doc_sent_id = '_'.join([doc_id, sent_start_char])
                tokens = []
                for tok in  sentence.find_all('token'):
                    tok_dict = tok.attrs
                    tok_dict['text'] = tok.text
                    tok_dict['token_id'] = tok_counter
                    tokens.append(tok_dict)
                    tok_counter += 1
                doc_sent_dict[doc_sent_id] = tokens
    return doc_sent_dict


def  save_doc_info(topics_file_path, output_folder):
    topic_dict_rows = read_csv(topics_file_path)
    topic_map = {}
    for row in topic_dict_rows:
        topic_map[row['child_uid']] = {'topic':row['topic'], 'language':row['lang_id']}
    json.dump(topic_map, open(output_folder + '/topic_map.json', 'w'), indent=2, sort_keys=True)


def generate_mentions(mention_file, doc_sent_map):
    nlp = spacy.load('en_core_web_sm', disable=['textcat'])
    mention_rows = read_csv(mention_file)
    mentions = []
    cluter_counter = 10000000
    for row in mention_rows:
        if not row['language'] == 'eng':
            continue
        doc_id = row['doc_id']
        topic = row['topic']
        # tokens = row['canonical_mention.actual']
        # if tokens == '':
        #     tokens = row['mention.actual']
        subsubtype = row['type']
        subtype = row['subtype']
        start_offset = int(row['start_offset'])
        end_offset = int(row['end_offset'])
        sent_start_offset = int(row['sent_start_offset'])

        doc_sent_id = '%s_%s'%(doc_id, sent_start_offset)
        doc_sent_tokens = doc_sent_map[doc_sent_id]

        token_ids = []
        sent_tok_numbers = []
        tokens = []
        for i, tok in enumerate(doc_sent_tokens):
            id_ = tok['id'].split('-')[1]
            token_id = tok['token_id']
            if end_offset < int(tok['end_char']):
                break
            if start_offset <= int(tok['start_char']) and \
                    int(tok['end_char']) <= end_offset:
                token_ids.append(token_id)
                sent_tok_numbers.append(i)
                tokens.append(tok['text'])

        tokens = ' '.join(tokens)
        lemmas, tags = [], []
        for tok in nlp(tokens):
            lemmas.append(tok.lemma_)
            tags.append(tok.tag_)

        mention = {
            'doc_id': doc_id,
            'm_id': row['mention_id'],
            'sentence_id': id_,
            'tokens_ids': token_ids,
            'sentence_tok_numbers': sent_tok_numbers,
            'tokens': tokens,
            'cluster_id': cluter_counter,
            'lemmas': ' '.join(lemmas),
            'tags': ' '.join(tags),
            'topic': topic,
            'singleton': False,
            'subsubtype': subsubtype,
            'subtype': subtype
        }
        if mention['tokens'].strip() != '':
            mentions.append(mention)
            cluter_counter+=1
    return mentions

def generate_tfidf(doc_sent_map, topic_map, output_folder):
    english_docs = set([key for key, val in topic_map.items() if val['language'] == 'eng'])
    doc_map = defaultdict(str)
    for doc_sent, tok_list in doc_sent_map.items():
        sentence = ' '.join([tok['text'].lower() for tok in tok_list])
        doc = doc_sent.split('_')[0]
        if doc in english_docs:
            doc_map[doc] += ' ' + sentence

    generate_tf_idf_doc(output_folder, doc_map)


if __name__=='__main__':
    ltf_dir = '/media/rehan/work/aida/LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0/ltf/'
    topics_file_path = '/media/rehan/work/aida/LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0/docs/parent_children.tab'
    output_folder = './mentions/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # aida_ltf2json(ltf_dir, output_folder, 'test')
    # save_doc_info(topics_file_path, './')
    # doc_sent_map = get_doc_sent_map(ltf_dir)
    # pickle.dump(doc_sent_map, open('doc_sent_map.pkl', 'wb'))
    doc_sent_map = pickle.load(open('doc_sent_map.pkl', 'rb'))
    topic_map = json.load(open('topic_map.json'))

    regressor_folder = 'cdlm/regressor/'
    # generate_tfidf(doc_sent_map, topic_map, regressor_folder)
    event_mentions = json.load(open('mentions/test_events.json'))
    generate_related_words(event_mentions, regressor_folder + '/related_words_map.pkl')
    # mentions = generate_mentions('./mentions/event.csv.tsv', doc_sent_map)
    # mentions = generate_mentions('./mentions/entity.csv.tsv', doc_sent_map)
    # json.dump(mentions, open(output_folder + '/test_events.json', 'w'), indent=2, )
    # json.dump(mentions, open(output_folder + '/test_entities.json', 'w'), indent=2, )
    pass
