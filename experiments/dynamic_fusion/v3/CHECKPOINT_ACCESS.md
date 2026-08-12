# AdaptCLIP official checkpoint access

The official repository points to `https://huggingface.co/csgaobb/AdaptCLIP`.
The model is public but uses automatic gated access. This machine currently has
no `HF_TOKEN`, no `HUGGING_FACE_HUB_TOKEN`, and no saved Hugging Face login.
Unauthenticated download returns HTTP 401, so the automation did not bypass the
gate or write a partial checkpoint.

## One-time user action

1. Sign in to Hugging Face in a browser.
2. Open `https://huggingface.co/csgaobb/AdaptCLIP` and accept/request access if
   the page displays an access form.
3. Create a read-only Hugging Face token.
4. In PowerShell, run:

   ```powershell
   D:\STUDY\My_github\sci_project\.venv-adaptclip\Scripts\huggingface-cli.exe login
   ```

   Paste the token only into the CLI prompt. Do not save it in project files or
   send it in chat.

## Required artifact

- Repository revision: `e9f5c06c9abd4015b73a8bb4e0477e48b7bd7b86`
- File: `adaptclip_checkpoints/12_4_128_train_on_visa_3adapters_batch8/epoch_15.pth`
- Size: 7,520,074 bytes
- SHA256: `777821da141eb57d159acef46868440faf773a2dd0acf5c276ec3f258c27edee`
- Local destination:
  `methods/adaptclip/adaptclip_checkpoints/12_4_128_train_on_visa_3adapters_batch8/epoch_15.pth`

After the authenticated download, do not start inference manually. First run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_v3_adaptclip_mpdd_gate_a.ps1 -ValidateOnly
```

Only if that command passes should the same launcher be run without
`-ValidateOnly`. It is bounded to MPDD seed0/K1, batch size 1, no initial metric
computation, and unified prediction-cache export.
