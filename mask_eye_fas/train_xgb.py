import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pprint

def pad_array(lst, size=30):
    arr = np.zeros(size, dtype=np.float32)
    length = min(len(lst), size)
    arr[:length] = lst[:length]
    return arr

from collections import Counter

def check_class_distribution_counter(y):
    c = Counter(y)
    total = sum(c.values())
    print("Class distribution via Counter:")
    for lbl, cnt in sorted(c.items()):
        ratio = cnt/total*100
        print(f"  Label {lbl}: {cnt} ({ratio:.2f}%)")


if __name__ == '__main__':
    npy_path = "/purestorage/AILAB/AI_1/tyk/3_CUProjects/fas_lab/mask_eye_fas/output_npy/final_data.npy"
    arr = np.load(npy_path, allow_pickle=True)
    data_list = arr.tolist()

    X = []
    y = []

    for item in data_list:
        # left_feats = item["eye_left_feats"]
        # right_feats= item["eye_right_feats"]
        # label = item["label"]  # 0 or 1

        # # print(left_feats)
        # # print(type(left_feats))


        # left_array = []
        # if "freq" in left_feats:
        #     left_array.extend(left_feats["freq"])

        # if "reflection" in left_feats:
        #     left_array.extend(left_feats["reflection"])

        # left_array = np.array(left_array, dtype=np.float32)


        # right_array = []
        # if "freq" in right_feats:
        #     right_array.extend(right_feats["freq"])
        # if "reflection" in right_feats:
        #     right_array.extend(right_feats["reflection"])
        # right_array = np.array(right_array, dtype=np.float32)

        # print(len(left_array), len(right_array))
        # if len(left_array)==60 or len(right_array)==60:
        #     continue

        # feature_vec = np.concatenate([left_array, right_array], axis=0) 
        # X.append(feature_vec)
        # y.append(label)

        # break
        left_feats  = item["eye_left_feats"] # dict
        right_feats = item["eye_right_feats"]
        label       = item["label"]
        
        edge_l = left_feats.get("edge", [])
        shadow_l = left_feats.get("shadow", [])
        refl_l = left_feats.get("reflection", [])
        freq_l = left_feats.get("freq", [])


        edge_l_pad = pad_array(edge_l, 30)
        shadow_l_pad = pad_array(shadow_l, 30)
        refl_l_pad = pad_array(refl_l, 30)
        freq_l_pad = pad_array(freq_l, 30)

        left_merged = np.concatenate(
            [edge_l_pad, shadow_l_pad, refl_l_pad, freq_l_pad], axis=0
        )

        edge_r = right_feats.get("edge", [])
        shadow_r = right_feats.get("shadow", [])
        refl_r = right_feats.get("reflection", [])
        freq_r = right_feats.get("freq", [])
        
        edge_r_pad = pad_array(edge_r, 30)
        shadow_r_pad = pad_array(shadow_r, 30)
        refl_r_pad = pad_array(refl_r, 30)
        freq_r_pad = pad_array(freq_r, 30)

        right_merged= np.concatenate(
            [edge_r_pad, shadow_r_pad, refl_r_pad, freq_r_pad], axis=0
        )

        feature_vec = np.concatenate([left_merged, right_merged], axis=0) # shape=(120,)
        
        X.append(feature_vec)
        y.append(label)
        

        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    pp = pprint.PrettyPrinter(indent=2, width=120)
    pp.pprint(y)

    check_class_distribution_counter(y)

    print(X.shape, y.shape)

    test_size=0.2
    random_state=42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )


    check_class_distribution_counter(y_train)
    check_class_distribution_counter(y_test)

    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(y_test)
    print(y_pred)