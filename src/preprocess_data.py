from pandas import read_csv, Series
import numpy as np
import pickle


to_drop = ['v1',
           'v',
           'bill',
           'candidate',
           'comment',
           'comms',
           'cong',
           'congress',
           'correlation',
           'date',
           'date_voted',
           'defeat',
           'description',
           'eh1',
           'eh2',
           'geometric_mean',
           'icpsrno',
           'id',
           'log_likelihood',
           'name',
           'naytotal',
           'num_errors',
           'num_votes',
           'number',
           'party_char',
           'question',
           'result',
           's',
           's_hr',
           'sponsor',
           'state',
           'state_ab',
           'state_id',
           'state_name',
           'x_bse',
           'x_coord',
           'y_bse',
           'y_coord',
           'year',
           'yeatotal',
           'session_t']
 
lab_vote = "new_vote"
bill_id = "hr"
test_perc = 0.01
valid_perc = 0.02
path_112_topics = "../data/just_topics_112.csv"


def one_hot_encode(data, var_name):
    for v in data[var_name].unique():
        ohv_name = var_name + "_" + v
        data[ohv_name] = np.float64(data[var_name] == v)

    
def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def process_with_topics():
    print "Loading data..."
    f = read_csv(path_112_topics)
    f.drop(to_drop, 1, inplace=True)
    
    # Data must be sorted correctly for everything to work
    print "Sorting and splitting..."
    f.sort(["hr", "dist"], inplace=True)
    
    # Drop observations with votes labelled 6 and 9.
    # I assume these are abstention/absence.
    v = f.index[~f[lab_vote].isin([0, 1])]
    f.drop(v, inplace=True)

    # Split into training/test/validation
    test_n = np.ceil(len(f) * test_perc)
    valid_n = np.ceil(len(f) * valid_perc)

    # take out the most recent bills to test and validate
    held_out_rows = f.index[np.int64(np.arange(len(f) - test_n - valid_n, len(f)))]
    # Get the index of the first observation in the held out rows, all the votes on that bill
    # will be included in the validation/test set. This is to preserve the chronological
    # structure of the problem.
    add = f.index[f.ix[f["hr"] == f.ix[held_out_rows[0], "hr"], :].index[0]:held_out_rows[0]]
    held_out_rows = add.tolist() + held_out_rows.tolist()
    t_rows = np.random.choice(held_out_rows, np.int64(test_n))
    v_rows = np.int64([r for r in held_out_rows if r not in t_rows])

    f.drop(["hr", "dist"], 1, inplace=True)
    
    un1 = f["un1"]
    f.drop("un1", 1, inplace=True)
    
    print "Creating labels..."
    # Extract vote labels and pass/reject labels for a separate file
    # I do this because tensorflow wants labels in a separate array
    f[lab_vote] = np.float32(f[lab_vote] == 1)
    vote_labels = f[lab_vote]
    f.drop(lab_vote, 1, inplace=True)

    print "list of features included: ", f.columns.tolist()
    
    # Standardize non 0/1 variables
    print "Encoding features..."
    numeric_vars = [x for x in f.columns if f.dtypes[x] == "float64" and (~f[x].isin([0, 1])).any()]
    for v in numeric_vars:
        f[v] = normalize(f[v])

    # encode party as 1 if republican 0 if democrat
    f["republican"] = f["party"].apply(lambda x: 1.0 if x == 200 else 0.0)
    f.drop("party", 1, inplace=True)

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
    
    # impute NAs in features. I use the column mean.
    # will have to rethink this at some point.
    print "Imputing NAs..."
    to_fill = [v for v in f.columns if f.dtypes[v] == "float64" and
               f[v].isnull().sum() > 0]
    f[to_fill] = f[to_fill].fillna(f[to_fill].mean())

    assert(f.isnull().sum().sum() == 0)
    
    datasets = {"test_feats": f.ix[t_rows, :],
                "test_labels": vote_labels[t_rows],
                "valid_feats": f.ix[v_rows, :],
                "valid_labels": vote_labels[v_rows],
                "train_feats": f.drop(held_out_rows, 0),
                "train_labels": vote_labels.drop(held_out_rows),
                "unif_train_feats": f.ix[un1 <= .75, :],
                "unif_train_labels": vote_labels.ix[un1 <= .75],
                "unif_valid_feats": f.ix[un1 > .75, :],
                "unif_valid_labels": vote_labels.ix[un1 > .75]}

    print "Saving output..."
    # Convert to np array and save
    for d in datasets:
        np.save("../data/" + d, datasets[d].as_matrix())
