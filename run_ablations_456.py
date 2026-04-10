import os
from envs.adapter import load_family_scene
from eval.ablations import ablation_noise_level, ablation_likelihood_misspec

SCENES_DIR = "/Users/rheas/SURP/enviornment/data/scenes"
sparse_files = sorted(f for f in os.listdir(SCENES_DIR)
                      if f.startswith("sparse_clutter"))[:30]

sparse_scenes, sparse_classes = [], []
for fname in sparse_files:
    s = load_family_scene(os.path.join(SCENES_DIR, fname), "medium")
    sparse_scenes.append(s)
    sparse_classes.append([o["true_class"] for o in s["obstacles"]])

ablation_noise_level(sparse_files, SCENES_DIR)
ablation_likelihood_misspec(sparse_scenes, sparse_classes)