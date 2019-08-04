import langdetect
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


def is_uk_language(text):
    try:
        if langdetect.detect(text) == 'uk':
            return True
    except:
        return False

    return False


def get_all_txt_files_from_dir(dir_path):
    files = []
    for r, d, f in os.walk(dir_path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))

    return files


def get_filename_without_extension(file_path):
    filename = os.path.splitext(file_path)[0]
    filename = filename.split('/')[-1]
    return filename




def create_dict_labels_from_files(files):
    targets_dict = {}
    for i in range(0, len(files)):
        filename = get_filename_without_extension(files[i])
        targets_dict[filename] = i
    return targets_dict


def parse_paragraphs_from_file(file):
    with open(file) as f:
        paragraphs = re.split(r'\d{7}\n', f.read())

    paragraphs = list(filter(lambda paragraph: is_uk_language(paragraph), paragraphs[1:]))
    return paragraphs


def load_dataset_from_file(file, targets_dict):
    label = get_filename_without_extension(file)
    paragraphs = parse_paragraphs_from_file(file)
    targets = len(paragraphs) * [targets_dict[label]]
    return list(zip(paragraphs, targets))


def load_dataset_from_dir(dir_path):
    files = get_all_txt_files_from_dir(dir_path)
    targets_dict = create_dict_labels_from_files(files)

    all_data = []
    for file in files:
        data = load_dataset_from_file(file, targets_dict)
        all_data += data

    return all_data, targets_dict


def text2vector(text, model, stop_words, tf_idfs=None, feature_names_dict=None):
    words = re.findall(r'\w+', text)
    vector = np.zeros(300)
    for word in words:
        try:
            if stop_words is not None and word in stop_words:
                continue
            v = model.wv[word.lower()]
            vector += v
            if tf_idfs is not None and feature_names_dict is not None:
                idx = feature_names_dict[word.lower()]
                weight = tf_idfs[idx].item(0, 0)
                vector *= weight
        except:
            pass
    v = vector / np.linalg.norm(vector)
    if np.linalg.norm(vector) == 0.0:
        return None

    return v


def create_tf_idf_vector(corpus):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(corpus)
    count_vector = cv.transform(corpus)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    tf_idf_vector = tfidf_transformer.transform(count_vector)

    feature_names = cv.get_feature_names()

    return tf_idf_vector, feature_names


def dataset2vector(data, model, stop_words=None, use_tf_idf_weights=False):
    if use_tf_idf_weights:
        tf_idf_vector, feature_names = create_tf_idf_vector(list(list(zip(*data))[0]))
        feature_names_dict = dict(zip(feature_names, range(0, len(feature_names))))
    new_data = []
    for i in range(0, len(data)):
        if use_tf_idf_weights:
            vector = text2vector(data[i][0], model, stop_words, tf_idf_vector[i].T.todense(), feature_names_dict)
        else:
            vector = text2vector(data[i][0], model, stop_words)
        if vector is None:
            continue
        new_data.append((vector, data[i][1]))
    return new_data



def read_stop_words(file):
    with open(file) as f:
        stop_words = f.read().split('\n')

    return stop_words


def dataset2tfidf_vector(data):
    all_texts = list(list(zip(*data))[0])
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)


    vector_data = []
    for data_item in data:
        v = vectorizer.transform([data_item[0]]).toarray()[0]

        vector_data.append((v, data_item[1]))

    return vector_data


def get_train_and_test_data(vector_data):

    np.random.shuffle(vector_data)

    X = np.array([data_item[0] for data_item in vector_data])
    Y = np.array([data_item[1] for data_item in vector_data])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    return X_train, X_test, y_train, y_test









