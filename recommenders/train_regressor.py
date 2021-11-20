import argparse
import pyhocon
from tqdm import tqdm
from itertools import combinations
from sklearn.utils import shuffle
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from cross_encoder.models import Regressor
from cross_encoder.evaluator import Evaluation
from cross_encoder.dataset import CrossEncoderDatasetFull
from cross_encoder.utils import *
from recommenders.featurization import Featurizer


def collate_fn(data):
    return data


def neg_loss(pos, neg):
    return -(torch.sum(torch.log(pos)) + torch.sum(torch.log(1 - neg)))


def train_regressor(config_path, out_folder, num_gpus=2):
    config = pyhocon.ConfigFactory.parse_file(config_path)

    inter_folder = out_folder + '/intermediate/'
    if not os.path.exists(inter_folder):
        os.makedirs(inter_folder)

    train_path = './%s/%s/%s.pkl' % (inter_folder, 'train', 'train')
    if not os.path.exists(train_path):
        train = CrossEncoderDatasetFull(config, 'train', mode='regressor')
        pickle.dump(train, open(train_path, 'wb'))
    train = pickle.load(open(train_path, 'rb'))
    config.batch_size = 6000
    train_loader = data.DataLoader(train, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=collate_fn)

    dev_path = './%s/%s/%s.pkl' % (inter_folder, 'dev', 'dev')
    if not os.path.exists(dev_path):
        dev = CrossEncoderDatasetFull(config, 'dev', mode='regressor')
        pickle.dump(dev, open(dev_path, 'wb'))
    dev = pickle.load(open(dev_path, 'rb'))
    dev_loader = data.DataLoader(dev, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=0, collate_fn=collate_fn)

    device_ids = config.gpu_num[:num_gpus]
    device = torch.device("cuda:{}".format(device_ids[0]))

    if device.type == "cuda":
        torch.cuda.set_device(device)

    vec_path = out_folder + '/context_vector_map.pkl'
    rel_path = out_folder + '/related_words_map.pkl'
    tfidf_path = out_folder + '/tfidf.pkl'
    sent2ent_path = out_folder + '/sent2ent.pkl'

    featurizer = Featurizer(tfidf_path, rel_path, vec_path, sent2ent_path)

    model = Regressor(featurizer.get_feature_len()).to(device)
    parallel_model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = get_loss_function(config)
    optimizer = get_optimizer(config, [parallel_model])
    scheduler = get_scheduler(optimizer, total_steps=config.epochs * len(train_loader))

    for epoch in range(config.epochs):
        accumulate_loss = 0
        number_of_positive_pairs, number_of_pairs = 0, 0

        parallel_model.train()

        running_loss = 0.0
        tk = tqdm(train_loader)
        # i = 0
        loss_mini_batch = 0
        optimizer.zero_grad()
        run_f1 = []
        run_prec = []
        run_rec = []
        run_acc = []
        run_loss = []
        run_f1 = 0
        run_prec = 0
        run_rec = 0
        run_acc = 0
        run_loss = 0

        for i, batch in enumerate(tk):
            if i == 1000:
                break
            batch_x, batch_y = zip(*batch)

            batch_y = np.array(batch_y).reshape((-1,1))

            pos_indices = np.where(batch_y==1)[0]
            neg_indices = np.where(batch_y!=1)[0]

            pos_x = [batch_x[i] for i in pos_indices]
            neg_x = [batch_x[i] for i in neg_indices][:len(pos_x)*3]

            batch_x = pos_x + neg_x

            batch_y = [1.]*len(pos_x) + [0.]*len(neg_x)
            #
            # featurized_x = featurizer.featurize(batch_x)
            # featurized_x = torch.tensor(featurized_x).to(device)
            #
            batch_y = torch.tensor(batch_y).reshape((-1,1))
            # scores = parallel_model(featurized_x.float())

            pos_x_f = torch.tensor(featurizer.featurize(pos_x)).float().to(device)
            pos = parallel_model(pos_x_f)

            neg_x_f = torch.tensor(featurizer.featurize(neg_x)).float().to(device)
            neg = parallel_model(neg_x_f)

            loss = neg_loss(pos, neg)
            loss.backward()
            # loss = criterion(scores, batch_y.to(device))
            # loss.mean().backward()

            loss_mini_batch += loss.mean().item()
            # torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            parallel_model.zero_grad()

            scores = torch.cat((pos, neg))

            accumulate_loss += loss.item() / config.batch_size
            number_of_positive_pairs += len((batch_y == 1).nonzero())
            number_of_pairs += len(batch_y)
            strict_preds = (scores > 0.75).to(torch.int)
            eval = Evaluation(strict_preds, batch_y.to(device))
            rec, prec, f1s, acc = eval.get_recall(), eval.get_precision(), eval.get_f1(), eval.get_accuracy()
            s = i + 1
            run_f1 = run_f1 * i / s + f1s / s
            run_prec = run_prec * i / s + prec / s
            run_rec = run_rec * i / s + rec / s
            run_acc = run_acc * i / s + acc / s
            run_loss = run_loss * i / s + loss.mean().item() / s

        print(run_f1)
        print(run_prec)
        print(run_rec)
        print()
        torch.save(model.state_dict(), os.path.join(out_folder, 'regressor.chk'))


if __name__=='__main__':
    config_file_path = './config_pairwise_long_reg_span.json'
    out_folder = './data/ecb/cdlm/'
    train_regressor(config_file_path, out_folder)
