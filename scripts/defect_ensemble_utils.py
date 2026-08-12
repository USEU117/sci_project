"""Memory-efficient defect ensemble text feature builder.

Uses standard CLIP model (design_details=None) so encode_text works properly.
Keeps all tensors on GPU to avoid CPU memory fragmentation.
"""
import torch
from typing import List


# Per-category base object names
CATEGORY_OBJECT_NAMES = {
    "bracket_black":  "bracket",
    "bracket_brown":  "bracket",
    "bracket_white":  "bracket",
    "connector":      "connector",
    "metal_plate":    "metal plate",
    "tubes":          "tube",
}


# Copy of the prompt templates from prompt_ensemble.py
PROMPT_NORMAL = [
    '{}', 'flawless {}', 'perfect {}', 'unblemished {}',
    '{} without flaw', '{} without defect', '{} without damage',
]
PROMPT_ABNORMAL = [
    'damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage',
]
PROMPT_TEMPLATES = [
    'a bad photo of a {}.', 'a low resolution photo of the {}.',
    'a bad photo of the {}.', 'a cropped photo of the {}.',
    'a bright photo of a {}.', 'a dark photo of the {}.',
    'a photo of my {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.',
    'a photo of one {}.', 'a close-up photo of the {}.',
    'a photo of a {}.', 'a low resolution photo of a {}.',
    'a photo of a large {}.', 'a blurry photo of a {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    'a photo of the small {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'a dark photo of a {}.',
    'a photo of a cool {}.', 'a photo of a small {}.',
    'there is a {} in the scene.', 'there is the {} in the scene.',
    'this is a {} in the scene.', 'this is the {} in the scene.',
    'this is one {} in the scene.',
]

# Defect word variants
DEFECT_VARIANTS = [
    "damaged {}", "broken {}", "defective {}",
    "scratched {}", "stained {}", "deformed {}",
    "bent {}", "dented {}", "cracked {}",
    "fractured {}", "contaminated {}", "missing part of {}",
    "misaligned {}", "displaced {}", "warped {}",
    "corroded {}", "{} with manufacturing defect", "{} with surface anomaly",
]

DEFECT_VARIANTS_FAST = [
    "damaged {}", "broken {}", "scratched {}",
    "deformed {}", "cracked {}", "stained {}",
]


def build_prompt_sentences(object_text: str) -> List[str]:
    """Build 420 prompted sentences for a given object text.
    
    For each state word (7 normal + 5 abnormal = 12), format with object_text,
    then fill into each of the 35 photo templates.
    Total: 12 x 35 = 420 sentences.
    """
    prompt_state = [PROMPT_NORMAL, PROMPT_ABNORMAL]
    sentences = []
    for state_list in prompt_state:
        prompted_states = [state.format(object_text) for state in state_list]
        for s in prompted_states:
            for template in PROMPT_TEMPLATES:
                sentences.append(template.format(s))
    return sentences


def tokenize(texts, context_length=77):
    """Tokenize texts using the simple tokenizer."""
    import sys
    from pathlib import Path
    _method_root = Path(__file__).resolve().parents[1] / "methods" / "AnomalyCLIP-main"
    if str(_method_root) not in sys.path:
        sys.path.insert(0, str(_method_root))
    from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
    from pkg_resources import packaging
    _tokenizer = _Tokenizer()

    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def encode_text_batched(model, sentences: List[str], device: str, batch_size: int = 40):
    """Encode text in batches. Keeps results on GPU.

    Returns text_features of shape [1, 2, embed_dim]:
      - channel 0: normal state features
      - channel 1: abnormal state features
    
    Sentences are organized as:
      normal = 7 states x 35 templates = 245
      abnormal = 5 states x 35 templates = 175
    """
    n_normal = len(PROMPT_NORMAL) * len(PROMPT_TEMPLATES)  # 245
    n_abnormal = len(PROMPT_ABNORMAL) * len(PROMPT_TEMPLATES)  # 175

    # We'll compute running mean on GPU to avoid CPU accumulation
    normal_sum = None
    abnormal_sum = None
    normal_count = 0
    abnormal_count = 0

    for start in range(0, len(sentences), batch_size):
        batch_sents = sentences[start:start + batch_size]
        tokens = tokenize(batch_sents).to(device)
        with torch.inference_mode():
            embeddings = model.encode_text(tokens).float()
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # Split batch into normal/abnormal portions
        batch_end = min(start + batch_size, len(sentences))
        n_in_batch = batch_end - start

        # Determine how many of these are normal vs abnormal
        normal_end = min(start + n_in_batch, n_normal)
        n_norm = max(0, normal_end - start)
        n_abnorm = n_in_batch - n_norm

        if n_norm > 0:
            norm_emb = embeddings[:n_norm]
            if normal_sum is None:
                normal_sum = norm_emb.sum(dim=0)
            else:
                normal_sum = normal_sum + norm_emb.sum(dim=0)
            normal_count += n_norm

        if n_abnorm > 0:
            abnorm_emb = embeddings[n_norm:]
            if abnormal_sum is None:
                abnormal_sum = abnorm_emb.sum(dim=0)
            else:
                abnormal_sum = abnormal_sum + abnorm_emb.sum(dim=0)
            abnormal_count += n_abnorm

    normal_feat = normal_sum / normal_count
    abnormal_feat = abnormal_sum / abnormal_count
    normal_feat = normal_feat / normal_feat.norm()
    abnormal_feat = abnormal_feat / abnormal_feat.norm()

    # Stack as [1, 2, embed_dim] on GPU
    return torch.stack([normal_feat, abnormal_feat], dim=0).unsqueeze(0)


def build_ensemble_text_features(
    model, object_name: str, device: str, fast: bool = False, batch_size: int = 40
) -> torch.Tensor:
    """Build ensemble text features by averaging across defect word variants.

    For each defect variant (e.g. "scratched bracket"):
      1. Build 420 prompted sentences
      2. Encode in batches -> [1, 2, embed_dim]
      3. Collect and average

    Returns averaged text_features of shape [1, 2, embed_dim] on device.
    """
    variants = DEFECT_VARIANTS_FAST if fast else DEFECT_VARIANTS
    all_features = []

    for variant_template in variants:
        defect_text = variant_template.format(object_name)
        sentences = build_prompt_sentences(defect_text)
        feats = encode_text_batched(model, sentences, device, batch_size=batch_size)
        all_features.append(feats)

    # Average across variants on GPU
    stacked = torch.cat(all_features, dim=0)  # [V, 2, embed_dim]
    ensemble = stacked.mean(dim=0, keepdim=True)  # [1, 2, embed_dim]
    ensemble = ensemble / ensemble.norm(dim=-1, keepdim=True)
    return ensemble
