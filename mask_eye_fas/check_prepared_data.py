import numpy as np
import pprint

if __name__ == "__main__":


    loaded = np.load("/purestorage/AILAB/AI_1/tyk/3_CUProjects/fas_lab/mask_eye_fas/output_npy/final_data.npy", allow_pickle=True)

    data_list = loaded.tolist()

    pp = pprint.PrettyPrinter(indent=2, width=120)

    pp.pprint(data_list)
    # print("Loaded dictionary:", loaded)
