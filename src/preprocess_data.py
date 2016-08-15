from pandas import read_csv, Series
import numpy as np
import pickle


non_word_columns = range(54) + range(10024, 10047)
word_columns = [x for x in range(10047) if x not in non_word_columns]
cols_to_drop = [0, 1, 2, 3, 6, 7, 9, 11, 18, 28, 29, 42, 45, 46, 48, 50, 51, 52, 53] + \
    range(30, 42) + [10024, 10025, 10026]
feature_cols = [c for c in range(10047) if c not in cols_to_drop + word_columns]
lab_vote = "vote"
lab_pass = "result_x"
bill_id = "hr"
names = "name"
test_perc = 0.01
valid_perc = 0.02
path_112 = "../data/merged_count_112.csv"


def one_hot_encode(data, var_name):
    for v in data[var_name].unique():
        ohv_name = var_name + "_" + v
        data[ohv_name] = np.float64(data[var_name] == v)


def to_np_array(data):
    pass

    
def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def create_bill_dict(bill_ids, words):
    words[bill_id] = bill_ids
    dict = {}
    for b in bill_ids.unique():
        w = words.ix[words[bill_id] == b, :].iloc[0].drop(bill_id)
        w = w[w > 0]
        # TODO: Should use tf-idf instead
        w = w / w.sum()
        dict[b] = {t: w[t] for t in w.index}
    words.drop(bill_id, axis=1, inplace=True)
    return dict
    
    
def main():
    print "Loading data..."
    d = read_csv(path_112)
    
    # Data must be sorted correctly for everything to work
    print "Sorting and splitting..."
    d.sort(["hr", "name"], inplace=True)
    w = d.ix[:, word_columns]
    f = d.ix[:, feature_cols]

    # thing is huge better free up some memory
    del d

    # Drop observations with votes labelled 6 and 9.
    # I assume these are abstention/absence.
    v = f.index[f["vote"].isin([6, 9])]
    w.drop(v, inplace=True)
    f.drop(v, inplace=True)

    assert(len(w) == len(f))
    
    print "Creating labels..."
    # Extract vote labels and pass/reject labels for a separate file
    # I do this because tensorflow wants labels in a separate array
    vote_labels = f[lab_vote]
    
    # We only need one pass label per bill
    # this is borderline obfuscated. That sweet dict comprehension tho.
    pass_labels = {x: np.float64((f.ix[f[bill_id] == x, lab_pass] == "Passed").head(1))
                   for x in f[bill_id].unique()}
    
    # Now drop labels from data
    f.drop(["vote", "result_x"], 1, inplace=True)

    # Also store bill number and congressman id and drop them
    bill_ids = f[bill_id]
    bill_dict = {bill_ids.unique()[k]: k for k in range(len(bill_ids.unique()))}
    b = Series([bill_dict[b] for b in bill_ids], index=f.index)
    cong_ids = f[names]
    cong_dict = {cong_ids.unique()[k]: k for k in range(len(cong_ids.unique()))}
    c = Series([cong_dict[c] for c in cong_ids], index=f.index)
    f.drop([bill_id, names], 1, inplace=True)
    
    # We can also drop the date since bill hr are in chronological order
    f.drop("date", axis=1, inplace=True)
    
    # Standardize non 0/1 variables
    print "Encoding features..."
    numeric_vars = [x for x in f.columns if f.dtypes[x] == "float64" and (~f[x].isin([0, 1])).any()]
    for v in numeric_vars:
        f[v] = normalize(f[v])

    # encode party as 1 if republican 0 if democrat
    f["republican"] = f["party_x"].apply(lambda x: 1.0 if x == 200 else 0.0)
    f.drop("party_x", 1, inplace=True)

    # one hot encoding for categorical variables
    cat_vars = f.columns[f.dtypes == "object"]
    for v in cat_vars:
        # This function modifies the dataframe directly as a side effect
        # I'm doing it like this because the df will likely be large and I
        # don't wanna go around copying stuff too much
        one_hot_encode(f, v)
        f.drop(v, 1, inplace=True)

    # Convert ints to floats (tensorflow likes them better)
    to_convert = [v for v in f.columns if f.dtypes[v] != "float64"]
    for v in to_convert:
        f[v] = f[v].astype(np.float64)

    # Make sure all features are floats
    assert(len(f.dtypes.unique()) == 1)

    # assert there are no NAs among wordcounts
    assert(w.isnull().sum().sum() == 0)
    
    # impute NAs in features. I use the column mean.
    # will have to rethink this at some point.
    print "Imputing NAs..."
    to_fill = [v for v in f.columns if f.dtypes[v] == "float64" and
               f[v].isnull().sum() > 0]
    f[to_fill] = f[to_fill].fillna(f[to_fill].mean())

    assert(f.isnull().sum().sum() == 0)

    # Create and save bill lookup dictionary
    bd = create_bill_dict(b, w)
    with open("../data/bill_lookup.pickle", "wb") as of:
        pickle.dump(bd, of)
    
    # Split into training/test/validation
    test_n = np.ceil(len(f) * test_perc)
    valid_n = np.ceil(len(f) * valid_perc)

    # take out the most recent bills to test and validate
    held_out_rows = f.index[np.int64(np.arange(len(f) - test_n - valid_n, len(f)))]
    t_rows = np.random.choice(held_out_rows, np.int64(test_n))
    v_rows = np.int64([r for r in held_out_rows if r not in t_rows])
  
    datasets = {"test_feats": f.ix[t_rows, :],
                "test_words": w.ix[t_rows, :],
                "test_bills": b.ix[t_rows],
                "test_cong": c.ix[t_rows],
                "test_labels": vote_labels[t_rows],
                "valid_feats": f.ix[v_rows, :],
                "valid_words": w.ix[v_rows, :],
                "valid_bills": b.ix[v_rows],
                "valid_cong": c.ix[v_rows],
                "valid_labels": vote_labels[v_rows],
                "train_feats": f.drop(held_out_rows, 0),
                "train_words": w.drop(held_out_rows, 0),
                "train_bills": b.drop(held_out_rows),
                "train_cong": c.drop(held_out_rows),
                "train_labels": vote_labels.drop(held_out_rows)}

    print "Saving output..."
    # Convert to np array and save
    for d in datasets:
        np.save("../data/" + d, datasets[d].as_matrix())
    
        
if __name__ == "__main__":
    main()
