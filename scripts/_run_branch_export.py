"""Wrapper to run AdaptCLIP test.py with hardcoded args."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'methods', 'adaptclip'))

from test import test
import argparse

class Args:
    test_data_path = r"d:\STUDY\My_github\sci_project\experiments\dynamic_fusion\v3\v3_2_mpdd_s0_k1_branches\staged_mpdd_s0_k1"
    save_path = r"d:\STUDY\My_github\sci_project\experiments\dynamic_fusion\v3\v3_2_mpdd_s0_k1_branches\logs"
    pretrained_model = "ViT-L/14@336px"
    checkpoint_path = r"d:\STUDY\My_github\sci_project\methods\adaptclip\adaptclip_checkpoints\12_4_128_train_on_visa_3adapters_batch8\epoch_15.pth"
    dataset = "mpdd"
    features_list = [6, 12, 18, 24]
    batch_size = 1
    image_size = 224
    n_ctx = 12
    seed = 0
    sigma = 4
    k_shots = 1
    visual_learner = True
    textual_learner = True
    pq_learner = True
    eval_metrics = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO']
    fusion_type = "average_mean"
    vl_reduction = 4
    pq_mid_dim = 128
    pq_context = True
    class_name = None
    prediction_cache_dir = r"d:\STUDY\My_github\sci_project\outputs\dynamic_fusion\v3_2_branches\v3_2_mpdd_s0_k1"
    sample_id_root = r"d:\STUDY\My_github\sci_project\data\mpdd_raw\MPDD"
    skip_metrics = True
    export_branches = True

print("Running AdaptCLIP branch export...", flush=True)
test(Args())
print("DONE", flush=True)
