import numpy as np
from sklearn.neighbors import NearestNeighbors


def split_data(input_data, batch_size, input_bills, input_labels=None, shuffle=False):
    n = np.int64(input_data.shape[0])

    if shuffle:
        np.random.shuffle(input_data)
        
    n_batches = np.int64(np.floor(n / float(batch_size)))
        
    for i in np.arange(n_batches):
        d = np.copy(input_data[i * batch_size:(i + 1) * batch_size, :])
        b = np.copy(input_bills[i * batch_size:(i + 1) * batch_size])
        if input_labels is not None:
            l = np.copy(input_labels[i * batch_size:(i + 1) * batch_size])
            yield d, b, l
        else:
            yield d, b
            

def split_rnn(input_data, input_words, bill_ids, batch_size, window_length, input_labels=None):
    all_bills = np.sort(list(set(bill_ids)))
    bills = np.sort(list(set(bill_ids)))
    cpb = {c: list(np.where(bill_ids == c)[0]) for c in bills}
    br = range(window_length)
    while len(bills) > 0:
        sample_data = []
        sample_words = []
        sample_labels = []
        if window_length <= len(bills):
            b_ids = bills[br]
        else:
            b_ids = np.append(bills, all_bills[(len(all_bills) - (window_length - len(bills))):])
        for i in b_ids:
            # Needed so batches have the same size. If there are less congressmen left to sample
            # than the batch size we want, use already sampled ones.
            if len(cpb[i]) >= batch_size:
                c_ids = np.random.choice(cpb[i], batch_size, replace=False)
                for c in c_ids:
                    cpb[i].remove(c)
            else:
                c_ids = cpb[i]
                for c in c_ids:
                    cpb[i].remove(c)
                if batch_size - len(c_ids) < len(np.where(bill_ids == i)[0]):
                    c_ids.extend(np.random.choice(np.where(bill_ids == i)[0],
                                                  batch_size - len(c_ids),
                                                  replace=False))
                elif batch_size - len(c_ids) == len(np.where(bill_ids == i)[0]):
                    c_ids.extend(np.where(bill_ids == i)[0])
                else:
                    c_ids.extend(np.random.choice(np.where(bill_ids == i)[0],
                                                  batch_size - len(c_ids),
                                                  replace=True))

            sample_data.append(input_data[c_ids])
            sample_words.append(input_words[c_ids])
            sample_labels.append(input_labels[c_ids])
            
            if len(cpb[i]) == 0:
                bills = np.delete(bills, np.where(bills == i))

        yield np.vstack(sample_data), np.vstack(sample_words), np.vstack(sample_labels)

            
def add_oversampling(data, bills, labels, N, k, prop, os_label):
    os = SMOTE(N, k, np.hstack([data, bills])[labels == os_label])
    n_to_insert = int((len(labels) * prop) - sum(labels == os_label))
    src_indices = [np.random.choice(range(len(os))) for _ in range(n_to_insert)]
    dest_indices = [np.random.choice(range(len(labels))) for _ in range(n_to_insert)]
    new_data = np.insert(data, dest_indices, os[src_indices, :data.shape[1]], axis=0)
    new_bills = np.insert(bills, dest_indices, os[src_indices, data.shape[1]:], axis=0)
    new_labels = np.insert(labels, dest_indices, os_label, axis=0)
    print ("Inserted %d samples with SMOTE size %d, each synthetic sample was used on average %.2f times."
           % (n_to_insert, len(os), float(n_to_insert) / len(os)))

    return new_data, new_bills, new_labels
            

def SMOTE(N, k, samples):
    if N < 1:
        rnd_indices = np.random.choice(range(np.floor(N * len(samples))))
        sam = samples[rnd_indices]
        N = 1
    else:
        N = np.int32(np.floor(N))
        sam = samples

    smote_samples = np.zeros((N * sam.shape[0], sam.shape[1]))
    neighbor_finder = NearestNeighbors(n_neighbors=k)
    neighbor_finder.fit(sam)
    
    for i in range(len(sam)):
        nneighbours = neighbor_finder.kneighbors(sam[i, :], return_distance=False)[0]
        smote_sample = populate(N, i, nneighbours, sam)
        smote_samples[i:i + N] = smote_sample
    
    return smote_samples

    
def populate(N, i, neighbors, sample):
    created = np.zeros((N, sample.shape[1]))
    x = sample[i]
    for t in range(N):
        n_ix = np.random.choice(range(len(neighbors)))
        n = sample[neighbors[n_ix]]
        dif = n - x
        gap = np.random.sample()
        created[t] = x + gap * dif
    return created


def save_oversampled(features, bills, labels):
    of, ob, ol = add_oversampling(features, bills, labels, 4, 8, 0.2, 0)
    np.save("../data/oversample_feats.npy", of)
    np.save("../data/oversample_bills.npy", ob)
    np.save("../data/oversample_labels.npy", ol)
    
    
def build_vocabulary(bill_lookup):
    vocab = {}
    idx = 0
    for b in bill_lookup:
        for w in bill_lookup[b]:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


def change_words(vocab, lookup):
    nbl = {}
    for b in lookup:
        nbl[b] = {}
        for w in lookup[b]:
            nbl[b][vocab[w]] = lookup[b][w]
    return nbl

    
def compute_tfidf(wm):
    tf = np.multiply(1 / (np.sum(wm, axis=1) + 1), wm.T).T
    idf = np.log(float(len(wm)) / (np.sum((wm > 0), axis=0) + 1))
    return tf * idf
