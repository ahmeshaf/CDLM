import pyhocon
import os
import pickle
from cross_encoder.dataset import CrossEncoderDatasetFull
from cross_encoder.models import FullCrossEncoder, FullCrossEncoderSingle, Regressor
from recommenders.featurization import Featurizer
from transformers import AutoModel
import torch
from itertools import product
from tqdm import tqdm

def generate_mention_representations(config_file_path, model_dir, splits, out_folder, num_gpus=4, batch_size = 150, cpu=False):
    long = True
    cdmlm = True
    seman = True
    config = pyhocon.ConfigFactory.parse_file(config_file_path)

    all_split_vector_maps = []

    for split in splits:
        config.split = split
        inter_folder = out_folder + '/intermediate/%s/' % config.split
        vector_map_path = inter_folder + '/context_vector_map_%s.pkl'%config.split
        if not os.path.exists(inter_folder):
            os.makedirs(inter_folder)

        if os.path.exists(vector_map_path):
            vector_map = pickle.load(open(vector_map_path, 'rb'))
            all_split_vector_maps.append(vector_map)
            continue

        data_path = './%s/%s.pkl' % (inter_folder, config.split)
        if not os.path.exists(data_path):
            data = CrossEncoderDatasetFull(config, config.split)
            pickle.dump(data, open(data_path, 'wb'))
        data = pickle.load(open(data_path, 'rb'))

        device_ids = config.gpu_num[:num_gpus]
        device = torch.device("cuda:{}".format(device_ids[0]))
        
        if cpu:
            device = torch.device("cpu")
        
        cross_encoder = FullCrossEncoderSingle(config, long=seman)
        cross_encoder = cross_encoder.to(device)
        cross_encoder.model = AutoModel.from_pretrained(os.path.join(model_dir, 'bert')).to(device)
        if cpu:
            cross_encoder.linear.to(device)
        cross_encoder.linear.load_state_dict(torch.load(os.path.join(model_dir, 'linear')))
        model = torch.nn.DataParallel(cross_encoder, device_ids=device_ids)
        if cpu:
            model.module.to(device)
        model.eval()

        vector_map = {}

        for topic_num, topic in enumerate(data.topics):

            topic_mentions_ids = data.mentions_by_topics[topic]
            topic_mentions = [data.mentions[x] for x in topic_mentions_ids]
            mentions_repr = data.prepare_mention_representation(topic_mentions)

            m_ids = ["_".join([m['doc_id'].split('.')[0], m['m_id']]) for m in topic_mentions]

            instances = [' '.join(['<g>', '<doc-s>', men, '</doc-s>']) for men in mentions_repr]

            for i in tqdm(range(0, len(instances), batch_size), desc="topic: " + topic):
                batch = [instances[x] for x in range(i, min(i + batch_size, len(instances)))]
                batch_m_ids = [m_ids[x] for x in range(i, min(i + batch_size, len(instances)))]
                if not cdmlm:
                    bert_tokens = model.module.tokenizer(batch, pad_to_max_length=True)
                else:
                    bert_tokens = model.module.tokenizer(batch, pad_to_max_length=True,
                                                         add_special_tokens=False)

                input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
                attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)
                if long or seman:
                    m = input_ids.cpu()
                    k = m == cross_encoder.vals[0]
                    p = m == cross_encoder.vals[1]

                    v = (k.int() + p.int()).bool()
                    nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 2)

                    q = torch.arange(m.shape[1])
                    q = q.repeat(m.shape[0], 1)

                    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
                    # msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                    # msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

                    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
                    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
                    # msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
                    # msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

                    # attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
                    attention_mask_g = msk_0.int() * msk_1.int()
                if long:
                    input_ids = input_ids[:, :4096]
                    attention_mask = attention_mask[:, :4096]
                    attention_mask[:, 0] = 2
                    attention_mask[attention_mask_g == 1] = 2
                if seman:
                    arg1 = msk_0_ar.int() * msk_1_ar.int()
                    # arg2 = msk_2_ar.int() * msk_3_ar.int()
                    arg1 = arg1[:, :4096]
                    # arg2 = arg2[:, :4096]
                    arg1 = arg1.to(device)
                    # arg2 = arg2.to(device)
                else:
                    arg1 = None
                    arg2 = None

                with torch.no_grad():
                    men_vectors = model(input_ids, attention_mask, arg1).detach().cpu().numpy()

                for id, vec in zip(batch_m_ids, men_vectors):
                    vector_map[id] = vec

        pickle.dump(vector_map, open(vector_map_path, 'wb'))
        all_split_vector_maps.append(vector_map)

    all_vector_map = {}
    for vec_map in all_split_vector_maps:
        all_vector_map = {**all_vector_map, **vec_map}

    pickle.dump(all_vector_map, open(out_folder + '/context_vector_map.pkl', 'wb'))


def generate_cdlm_score(config_file_path, model_dir, split, out_folder, num_gpus=4, batch_size = 150):
    long = True
    cdmlm = True
    seman = True
    config = pyhocon.ConfigFactory.parse_file(config_file_path)
    config.split = split
    inter_folder = out_folder + '/intermediate/%s/'%config.split
    if not os.path.exists(inter_folder):
        os.makedirs(inter_folder)

    data_path = './%s/%s.pkl' % (inter_folder, config.split)
    if not os.path.exists(data_path):
        data = CrossEncoderDatasetFull(config, config.split)
        pickle.dump(data, open(data_path, 'wb'))
    data = pickle.load(open(data_path, 'rb'))

    device_ids = config.gpu_num[:num_gpus]
    device = torch.device("cuda:{}".format(device_ids[0]))

    cross_encoder = FullCrossEncoder(config, long=seman)
    cross_encoder = cross_encoder.to(device)
    cross_encoder.model = AutoModel.from_pretrained(os.path.join(model_dir, 'bert')).to(device)
    cross_encoder.linear.load_state_dict(torch.load(os.path.join(model_dir, 'linear')))
    model = torch.nn.DataParallel(cross_encoder, device_ids=device_ids)
    model.eval()

    for topic_num, topic in enumerate(data.topics):
        topic_scores_path = inter_folder + '/topic_%s_scores.pkl' % topic
        topic_id_pairs_path = inter_folder + '/topic_%s_id_pairs.pkl' % topic

        if os.path.exists(topic_scores_path):
            print("topic: " + topic + ' already done')
            continue

        topic_scores = []
        topic_mentions_ids = data.mentions_by_topics[topic]
        topic_mentions = [data.mentions[x] for x in topic_mentions_ids]
        mentions_repr = data.prepare_mention_representation(topic_mentions)

        first, second = zip(*list(product(range(len(topic_mentions)), repeat=2)))
        first, second = torch.tensor(first), torch.tensor(second)
        instances = [' '.join(['<g>', "<doc-s>", mentions_repr[first[i]], "</doc-s>", "<doc-s>",
                               mentions_repr[second[i]], "</doc-s>"]) for i in range(len(first))]

        for i in tqdm(range(0, len(instances), batch_size), desc="topic: " + topic):
            batch = [instances[x] for x in range(i, min(i + batch_size, len(instances)))]
            if not cdmlm:
                bert_tokens = model.module.tokenizer(batch, pad_to_max_length=True)
            else:
                bert_tokens = model.module.tokenizer(batch, pad_to_max_length=True,
                                                     add_special_tokens=False)

            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)
            if long or seman:
                m = input_ids.cpu()
                k = m == cross_encoder.vals[0]
                p = m == cross_encoder.vals[1]

                v = (k.int() + p.int()).bool()
                nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 4)

                q = torch.arange(m.shape[1])
                q = q.repeat(m.shape[0], 1)

                msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
                msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

                msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
                msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
                msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
                msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

                attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
            if long:
                input_ids = input_ids[:, :4096]
                attention_mask = attention_mask[:, :4096]
                attention_mask[:, 0] = 2
                attention_mask[attention_mask_g == 1] = 2
            if seman:
                arg1 = msk_0_ar.int() * msk_1_ar.int()
                arg2 = msk_2_ar.int() * msk_3_ar.int()
                arg1 = arg1[:, :4096]
                arg2 = arg2[:, :4096]
                arg1 = arg1.to(device)
                arg2 = arg2.to(device)
            else:
                arg1 = None
                arg2 = None

            with torch.no_grad():
                scores = model(input_ids, attention_mask, arg1, arg2)
                scores = torch.sigmoid(scores)
                topic_scores.extend(scores.detach().cpu().squeeze(1).numpy())

        topic_scores = torch.stack(topic_scores)

        topic_id_pairs = list(zip([topic_mentions_ids[f] for f in first],
                            [topic_mentions_ids[s] for s in second]))

        pickle.dump(topic_scores, open(topic_scores_path, 'wb'))
        pickle.dump(topic_id_pairs, open(topic_id_pairs_path, 'wb'))

    merge(data_path, inter_folder, out_folder, split)


def merge(data_path, inter_folder, out_folder, split):
    data = pickle.load(open(data_path, 'rb'))
    topic_mention_map = {}

    all_merged_scores = []

    for topic_num, topic in enumerate(data.topics):
        topic_scores_path = inter_folder + '/topic_%s_scores.pkl' % topic
        topic_id_pairs_path = inter_folder + '/topic_%s_id_pairs.pkl' % topic

        topic_scores = pickle.load(open(topic_scores_path, 'rb'))
        topic_id_pairs = pickle.load(open(topic_id_pairs_path, 'rb'))

        merged_scores = zip(topic_id_pairs, topic_scores)

        all_merged_scores.extend(merged_scores)

        # topic_mentions_ids = data.mentions_by_topics[topic_num]
        topic_mentions_ids = data.mentions_by_topics[topic]
        topic_mentions = [data.mentions[x] for x in topic_mentions_ids]

        m_ids = ["_".join([m['doc_id'].split('.')[0], m['m_id']]) for m in topic_mentions]
        for item in zip(topic_mentions_ids, m_ids):
            topic_mention_map[item[1]] = item[0]


    save_data = [all_merged_scores, topic_mention_map]

    pickle.dump(save_data, open(out_folder + '/cdlm_score_data_%s.pkl'%split, 'wb'))


def generate_regressor_score(config_file_path, model_path,
                             split, out_folder, num_gpus=2,
                             batch_size=3000, run_name='regressor'):
    config = pyhocon.ConfigFactory.parse_file(config_file_path)
    config.split = split
    inter_folder = out_folder + '/intermediate/%s/' % config.split
    score_folder = out_folder + '/regressor/'
    if not os.path.exists(score_folder):
        os.makedirs(score_folder)
    if not os.path.exists(inter_folder):
        os.makedirs(inter_folder)

    data_path = './%s/%s.pkl' % (inter_folder, config.split)
    if not os.path.exists(data_path):
        data = CrossEncoderDatasetFull(config, config.split)
        pickle.dump(data, open(data_path, 'wb'))
    data = pickle.load(open(data_path, 'rb'))

    device_ids = config.gpu_num[:num_gpus]
    device = torch.device("cuda:{}".format(device_ids[0]))

    vec_path = out_folder + '/context_vector_map.pkl'
    rel_path = out_folder + '/related_words_map.pkl'
    tfidf_path = out_folder + '/tfidf.pkl'

    sent2ent_path = out_folder + '/sent2ent.pkl'

    featurizer = Featurizer(tfidf_path, rel_path, vec_path, sent2ent_path)
    model = Regressor(featurizer.get_feature_len())
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    parallel_model = torch.nn.DataParallel(model, device_ids=device_ids)
    parallel_model.eval()

    all_scores = []
    all_id_pairs = []
    topic_mention_map = {}

    for topic_num, topic in enumerate(data.topics):

        topic_scores = []
        topic_mentions_ids = data.mentions_by_topics[topic]
        topic_mentions = [data.mentions[x] for x in topic_mentions_ids]

        m_ids = ["_".join([m['doc_id'].split('.')[0], m['m_id']]) for m in topic_mentions]
        for item in zip(topic_mentions_ids, m_ids):
            topic_mention_map[item[1]] = item[0]



        first, second = zip(*list(product(range(len(topic_mentions)), repeat=2)))

        topic_id_pairs = list(zip([topic_mentions_ids[f] for f in first],
                                  [topic_mentions_ids[s] for s in second]))
        all_id_pairs.extend(topic_id_pairs)

        instances = [(topic_mentions[first[i]], topic_mentions[second[i]]) for i in range(len(first))]

        for i in tqdm(range(0, len(instances), batch_size), desc="topic: " + topic):
            batch = [instances[x] for x in range(i, min(i + batch_size, len(instances)))]

            batch_x = torch.tensor(featurizer.featurize(batch)).float().to(device)

            with torch.no_grad():
                scores = parallel_model(batch_x)
                all_scores.extend(scores.detach().cpu().squeeze(1).numpy())

    all_merged_scores = list(zip(all_id_pairs, all_scores))
    save_data = [all_merged_scores, topic_mention_map]

    pickle.dump(save_data, open(out_folder + '/regressor/%s_score_data_%s.pkl' % (run_name, split), 'wb'))


if __name__=='__main__':
    config_file_path = './config_pairwise_long_reg_span.json'
    model_dir = './models/cdlm2/checkpoint_8/'
    splits = ['test']
    out_folder = './data/LDC2021E11_Eng/cdlm/'

    generate_mention_representations(config_file_path, model_dir, splits, out_folder, num_gpus=2, batch_size=12)
    # generate_regressor_score(config_file_path,
    #                          out_folder + '/regressor/regressor.chk',
    #                          'test', out_folder, run_name='regressor_no_cdlm')
