# Releasing a model for field use

The field build must serve exactly the model that was evaluated, and nothing
else. Two mechanisms enforce that; this page is how to use them.

## What the application does on its own

**It fails closed.** `api/readiness.py` is consulted on every scoring request,
and scoring is refused with HTTP 503 unless all of these hold:

| Check | Fails when |
|---|---|
| `text_encoder` | torch/transformers are missing, so text features would be mock embeddings |
| `acoustic_encoder` | the served acoustic model did not load (only relevant under `DEP_ACOUSTIC=wav2vec2`) |
| `fusion_head` | no trained weights, or weights of the wrong width, so scores would be random |
| `transcriber` | no speech-recognition backend, so transcripts would be placeholder text |
| `release` | the weights on disk are not byte-identical to a tagged release |

Every one of these used to degrade silently into plausible-looking output.
Development still needs that behaviour, so `DEP_ALLOW_UNVALIDATED=1` lets
scoring proceed - but every such result is stored with `validated: false`
and its reasons, and every page shows a banner.

**Only scoring stops.** Clinical notes, the risk assessment, scale responses
and the referral flag keep working, because the safety path must never depend
on the model.

A non-blocking warning is raised when `DEP_ASR_PROVIDER` names a hosted
provider but a different transcriber is in use - usually a missing API key.

`GET /health` reports all of this, and every score's `model` block records the
release name and weights hash, so a stored result can always be traced to the
exact weights that produced it.

## Cutting a release

Choose the model **before** looking at any field result, by the rule written
into the thesis (for example: lowest cross-validated HAM-D error among models
that pass the permutation test; simpler model on a tie). Then:

```bash
# 1. pin the weights
python scripts/tag_release.py write --release model-2026.09.1 \
    --notes "Wave 1 + wave 2, text+prosody ridge; see model card"

# 2. review outputs/weights/RELEASE.json, then commit it with the weights
git add outputs/weights/
git commit -m "Release model-2026.09.1"

# 3. verify and tag
python scripts/tag_release.py tag
git push origin model-2026.09.1
```

`tag` refuses if the weights no longer match the manifest, if anything the
model depends on (`api`, `src`, `scripts`, `web`, `data/kobo`,
`outputs/weights`, `requirements.txt`) has uncommitted changes, or if the name
is already tagged. **Releases are immutable:** a refit is a new release.

Names follow `model-YYYY.MM.N`.

## Handing a machine over

On the machine going to the field, in the environment it will actually run in:

```bash
git checkout model-2026.09.1
python scripts/preflight.py
```

`preflight.py` clears `DEP_ALLOW_UNVALIDATED` before checking, loads the API
exactly as the server would, and exits non-zero unless scoring is ready. **Do
not hand over a machine that fails it.**

What preflight does not check, and still needs doing by hand:

- a hosted transcription key is set if the site will use one (`DEP_ASR_PROVIDER`
  plus its key), and the site has the connectivity for it
- `ANTHROPIC_API_KEY` is set if the note, MSE and summary features are wanted
- the ethics approval covers the site, and any data leaving the country
- a short known-recording check: score two or three recordings whose results are
  already known, and confirm they match
