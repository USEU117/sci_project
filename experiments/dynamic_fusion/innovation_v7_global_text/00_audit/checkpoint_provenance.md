# AnomalyCLIP checkpoint provenance (task book 17 s.2.2 item 10)

- checkpoint: `D:\STUDY\My_github\sci_project\methods\AnomalyCLIP-main\checkpoints\9_12_4_multiscale_visa\epoch_15.pth`
- sha256: `415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4`
- origin: AnomalyCLIP official training on **VisA** (directory `9_12_4_multiscale_visa`), prompt-learner head over a frozen OpenAI CLIP ViT-L/14@336px visual/text tower.
- prompt: generic class `object`, learned context (design: Prompt_length=12, depth=9, text-n-ctx=4); normal state `{}`, abnormal state `damaged {}`.
- role on MPDD/BTAD/MVTec: **target-domain zero-shot transfer** of the prompt learner. The system is NOT 'fully training-free' and VisA is a source/in-domain dataset, NOT an independent external validation set.
- frozen inference: image_size 518, DAPM layer 20, encode_image -> global embedding @ learned text prompts -> softmax(/0.07) -> abnormal-class probability.