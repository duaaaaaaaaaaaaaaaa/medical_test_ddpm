import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from Diffusion.Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 500,
        "batch_size": 16,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 128,
        "grad_clip": 1.,
        "device": "cuda", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "9next_64_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledImgName": "Sampled.png",
        "nrow": 4
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
