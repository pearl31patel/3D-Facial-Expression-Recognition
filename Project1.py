import os
import sys
from tqdm import tqdm
from typing import List
import numpy as np
from math import acos, cos, sin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

VALID_TYPES = {"o", "t", "x", "y", "z"}

def error_handle(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

def calculate_pi():
    return round(2 * acos(0.0), 3)

class Sample:
    def __init__(self, subject, label, path):
        self.subject = subject
        self.label = label
        self.path = path

def read_bnd_file(path):

    pts = []

    with open(path, "r") as f:
        for line in f:
            parts = line.split()

            if len(parts) >= 4: 
                try:
                    pts.append([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ])
                except:
                    continue

    if len(pts) == 0:
        print("No landmark data found in", path)
        sys.exit(1)

    pts = np.array(pts)

    if pts.shape[0] > 83:
        pts = pts[:83, :]

    if pts.shape != (83, 3):
        print("WARNING: Wrong landmark size in", path)

    return pts


def center_points(points):
    mean_xyz = np.mean(points, axis=0)
    return points - mean_xyz


def rotation_matrix(axis, pi):

    c = cos(pi)
    s = sin(pi)

    if axis == "x":
        return np.array([[1,0,0],[0,c,s],[0,-s,c]])

    elif axis == "y":
        return np.array([[c,0,-s],[0,1,0],[s,0,c]])

    elif axis == "z":
        return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def rotate_points(points, axis):

    pi = calculate_pi()
    R = rotation_matrix(axis, pi)

    return np.dot(points, R.T)

def flatten_points(points):
    return points.reshape(-1)

def transform_points(points,data_type):

    if data_type == "o":
        return points

    elif data_type == "t":
        return center_points(points)

    elif data_type in ["x", "y", "z"]:
        return rotate_points(points, data_type)
    else:
        print("Invalid data type:", data_type)
        sys.exit(1)

def get_subject_folders(root_dir):

    subjects = []

    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)

        if os.path.isdir(full_path):
            subjects.append(name)

    return sorted(subjects)

def get_samples(root_dir):

    samples = []
    subjects = get_subject_folders(root_dir)

    for s in subjects:
        subj_path = os.path.join(root_dir, s)

        for expr in os.listdir(subj_path):
            expr_path = os.path.join(subj_path, expr)

            if os.path.isdir(expr_path):
                for file in os.listdir(expr_path):
                    if file.endswith(".bnd"):
                        samples.append(
                            Sample(s, expr, os.path.join(expr_path, file))
                        )

    return samples

def prepare_dataset(samples, data_type):

    X_list = []
    y_list = []
    subject_list = []
    label_set = set()

    for sample in tqdm(samples, desc="Preparing Dataset"):

        pts = read_bnd_file(sample.path)
        pts = transform_points(pts, data_type)

        feature = flatten_points(pts)

        X_list.append(feature)
        y_list.append(sample.label)
        subject_list.append(sample.subject)

        label_set.add(sample.label)

    X = np.vstack(X_list)
    y = np.array(y_list)
    subjects = np.array(subject_list)
    labels = sorted(label_set)

    return X, y, subjects, labels


def run_loso(X, y, subjects, labels):

    unique_subjects = list(set(subjects))

    #test 
    # unique_subjects = unique_subjects[:3]


    y_true_all = []
    y_pred_all = []

   
    # model = RandomForestClassifier(n_estimators=300)
    # model = RandomForestClassifier(n_estimators=80, n_jobs=-1)
    model = RandomForestClassifier(n_estimators=150, n_jobs=-1,class_weight="balanced",random_state=42)
    #test
    # model = RandomForestClassifier(n_estimators=10, n_jobs=-1)

    for test_subj in tqdm(unique_subjects, desc="Running LOSO"):

        test_mask = (subjects == test_subj)
        train_mask = (subjects != test_subj)

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average="macro")
    rec = recall_score(y_true_all, y_pred_all, average="macro")

    return acc, prec, rec, cm


def print_confusion_matrix(cm, labels):

    print("\nConfusion Matrix")
    print("Labels:", labels)

    for i in range(len(labels)):
        print(labels[i], cm[i])

def save_results(file_path, data_type, acc, prec, rec, cm, labels):

    with open(file_path, "w") as f:

        f.write("Data type: " + data_type + "\n\n")

        f.write("Accuracy: " + str(acc) + "\n")
        f.write("Precision: " + str(prec) + "\n")
        f.write("Recall: " + str(rec) + "\n\n")

        f.write("Labels: " + str(labels) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

def main():

    data_type = sys.argv[1]   # o / t / x / y / z
    root_dir = sys.argv[2]    # data/BU4DFE_BND_V1.1

    samples = get_samples(root_dir)

    X, y, subjects, labels = prepare_dataset(samples, data_type)

    acc, prec, rec, cm = run_loso(X, y, subjects, labels)

    print("Data type:", data_type)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)

    print_confusion_matrix(cm, labels)

    out_file = "results_" + data_type + ".txt"
    save_results(out_file, data_type, acc, prec, rec, cm, labels)

    print("Saved:", out_file)


if __name__ == "__main__":
    main()

















