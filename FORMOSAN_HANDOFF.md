# Formosan (族語) finetune data pipeline — handoff

Goal: finetune this repo's Higgs Audio v3 model on Formosan-language TTS
data drawn from four Hugging Face datasets, then train. This doc is a
complete handoff of what's decided, what's built, what's validated, and
what's still open — written so a fresh Claude Code session (on this
server or a different one) can pick up the work without re-deriving any
of it. Code lives in `scripts/formosan/`; this doc explains the *why*
behind it, which the code comments summarize but don't fully justify.

## 1. Source data

Four datasets, all under the `formospeech` org on HF Hub, all **gated**
(the active HF account needs org access — this session used `txya900619`,
which has it):

| dataset | configs (dialects) | rows | splits |
|---|---|---|---|
| `formospeech/ntu_formosan_corpus` | 10 | 14,130 | train only |
| `formospeech/klokah` | 42 | 388,351 (train) + 24,522 (eval) | train + eval |
| `formospeech/ithuan_formosan` | 3 | ~11,000 + 39 (tiny) eval | train + eval |
| `formospeech/nchc_formosan` | 6 | 18,120 | train only |

Common schema across all four: `id, audio, duration, text, ipa, mandarin,
language, lang_code, dnsmos_ovrl, dnsmos_sig, dnsmos_bak, dnsmos_p808`,
plus `speaker` (present only in `ithuan_formosan`/`nchc_formosan`) and
`split` (klokah/ithuan/nchc only — mirrors the train/eval filename, **not**
an ASR/TTS flag; there is no ASR/TTS flag anywhere in this data, unlike
older Formosan corpora).

Unlike the old (pre-existing, unrelated) Formosan TTS work in
`/mnt/md1/user_wayne/formosan_dialects/` (Chatterbox-based) and
`/mnt/md1/user_wayne/chatterbox/`, this new data has no pre-split
ASR/TTS subsets and no pre-assigned voice-cloning reference audio — both
had to be derived from scratch (see §3).

## 2. Decisions made (with rationale)

All of these were explicitly discussed and confirmed in conversation, not
unilaterally assumed:

- **Quality filter**: `dnsmos_ovrl >= 3.0` (null/NaN fails), **and**
  `mandarin` non-null/non-empty after `.strip()`. Both are hard
  requirements, applied together in `materialize.py`.
- **No per-language threshold relaxation.** Early concern: a flat 3.0 cut
  looked like it might wipe out `ckv`(噶瑪蘭)/`xsy`/`tsu` almost entirely —
  but that was only true within `ntu_formosan_corpus` alone. Checked the
  *combined* yield across all 4 datasets directly with pyarrow (column-
  pruned parquet reads, no audio download): ckv 11.6h, xsy 9.5h, tsu 10.7h
  after filtering — healthy. No special-casing needed.
- **Text field**: use `ipa`, with the `-` phone-delimiters stripped
  (`t-a-ɾ-a` → `taɾa`). Confirmed against the *old* (unrelated) Chatterbox
  Formosan pipeline's `dataset.py:_raw_text` — it strips the same
  delimiters for the same reason ("NeMo phone-tokenizer format,
  meaningless to a grapheme/BPE tokenizer"). Higgs' Qwen3 backbone uses a
  byte-level BPE tokenizer, so the same logic applies.
- **No language/dialect tag prefix on `text`.** Two reasons: (a) the user
  judged the `ipa` text itself already carries enough language signal;
  (b) confirmed Higgs has **no built-in language/dialect special token** —
  dumped the full 84-entry `added_tokens` list from
  `multimodalart/higgs-audio-v3-tts-4b-transformers`'s `tokenizer.json`:
  only structural tokens (`<|tts|>`, `<|ref_audio|>`, ...) and
  emotion/env/style/sfx/prosody control tokens exist, nothing for language.
- **Audio format**: 24kHz mono wav (matches `scripts/prepare_data.py`'s
  default `--sample-rate`).
- **eval doubles as test**: `klokah`/`ithuan_formosan`'s `eval` split is
  used both for validation and as the final held-out test set;
  `test.jsonl` is a byte-identical copy of `eval.jsonl` (not a further
  split — `ithuan_formosan`'s eval is only 39 rows, splitting it further
  wasn't worth the loss of eval-set size).
- **Ref-audio (voice-cloning) pairing is required (per-sample optional in
  the model, but we always try):**
  - `ithuan_formosan`/`nchc_formosan` (real `speaker` column): group by
    speaker within the *same* `(dataset, lang_code, split)`, pick a random
    *other* utterance from the same speaker. A speaker with only one
    utterance gets **no ref** (never force a cross-speaker match).
  - `ntu_formosan_corpus`/`klokah` (no speaker column) — see §3 below.
  - **4 of `nchc_formosan`'s 6 configs have a `speaker` column that's just
    a constant copy of `lang_code`** — not informative. Detected at
    *runtime* (`common.has_usable_speaker_column`: usable iff ≥2 distinct
    non-null values within the manifest), not hardcoded, so it also
    correctly falls back to embedding-based pairing for these.
  - **Ref pairing never crosses train/eval.** This mirrors the old
    Chatterbox pipeline's `rebuild_context_pairing.py`, which deliberately
    paired "train only with train, val only with val" — crossing them
    would leak eval-set audio into training conditioning and inflate
    downstream eval numbers.

## 3. Speaker-embedding model selection (for ntu_formosan_corpus/klokah)

Since these two datasets have no `speaker` column, same-speaker grouping
for ref-audio pairing has to be inferred from audio via speaker embeddings
+ clustering. This was **not** assumed — it was validated with an
explicit experiment using the two datasets that *do* have real speaker
labels (`ithuan_formosan`, `nchc_formosan`, 19,814 rows after the DNSMOS/
mandarin filter, 5 (dataset, lang_code) groups) as ground truth.

Five candidate embedding models were extracted and clustered/scored
against the real labels:

| model | clustering ARI (full 19,814 rows, best HDBSCAN) | EER | minDCF(p=0.01) |
|---|---|---|---|
| Resemble/Chatterbox VoiceEncoder | 0.933 | 9.09% | 0.0036 |
| SpeechBrain ECAPA-TDNN | 0.997 | 1.80% | 0.0007 |
| CAM++ (`Wespeaker/wespeaker-voxceleb-campplus`, ONNX, CPU) | 0.981 | 0.91% | 0.0004 |
| ReDimNet2-B6 English (`vb2+vox2_v0-lm`) | 1.000 | 0.16% | 0.0001 |
| **ReDimNet2-B6 Multilingual (`vb2+vox2+cnc2_v0-lm`)** | **1.000** | **0.07%** | **0.0000** |

**Winner: ReDimNet2-B6 Multilingual.** Loaded via
`torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name="b6",
train_type="lm", dataset="vb2+vox2+cnc2_v0", pretrained=True,
trust_repo=True)`. Clustering: **HDBSCAN, `metric="precomputed"` on a
cosine-distance matrix, `min_cluster_size=5, min_samples=1`** — chosen
over agglomerative-with-a-fixed-threshold because HDBSCAN's density-based
approach needs no per-dataset threshold tuning (agglomerative hit similar
peak scores but only at a very specific, non-transferable threshold — a
small shift dropped its ARI from 0.998 to 0.835).

Caveats worth remembering:
- **ReDimNet2-B6 needs a GPU to be practical.** ~1.2s/sample on CPU (would
  be many CPU-days for the ~500k rows in ntu+klokah combined) vs
  ~0.02-0.1s/sample batched on GPU (~3h for the full ntu+klokah volume on
  4 GPUs). `speaker_embed.py` defaults to `cuda:0` but works on CPU too
  (just slow) — **this stage should be scheduled whenever a GPU is briefly
  free**, it's a one-time offline batch job, not a sustained multi-hour
  hold like training.
- A small subset comparison (400 rows/group, easier task) showed several
  models tied at a perfect ARI and ECAPA-TDNN *dropping below*
  VoiceEncoder — this was noise from an easier/smaller slice, not real
  signal. **Trust the full 19,814-row numbers above, not subset numbers.**
- The clustering-ARI ranking and the EER/minDCF ranking disagree on
  CAM++ vs ECAPA-TDNN (CAM++ wins on EER, ECAPA-TDNN wins on ARI) — only
  because ARI is sensitive to the specific clustering algorithm's
  behavior, not just embedding quality. ReDimNet2 wins on both metrics
  either way, which is why it's the unambiguous pick.
- HDBSCAN's `min_cluster_size=5` means any `(dataset, lang_code, split)`
  group with very few surviving rows after filtering will end up
  **mostly/entirely "no ref"** — confirmed in the smoke test (tiny
  `--limit 20` groups saw 100% noise in several configs). This is the
  intended safe behavior (no ref beats a forced bad match), not a bug, but
  don't be surprised when full-scale low-yield configs show a high
  no-ref rate.

## 4. Pipeline code (`scripts/formosan/`)

Four stages, each independently resumable (skip-if-output-exists unless
`--overwrite`):

1. **`materialize.py`** — HF parquet → filtered 24kHz mono wav files +
   raw manifest jsonl. Applies the DNSMOS/mandarin filter, strips IPA
   dashes, copies `speaker` through verbatim if present (informativeness
   is judged later, not here).
2. **`speaker_embed.py`** — for manifests with no usable `speaker` column:
   ReDimNet2-B6-Multilingual embeddings, batched (duration-bucketed to
   minimize padding waste), cached to `<manifest_stem>_spk_emb.pt`.
3. **`pair_ref_audio.py`** — assigns `ref_audio_filepath`/`ref_text` per
   row, strictly within its own `(dataset, lang_code, split)`. Speaker-
   column path or HDBSCAN-cluster path, chosen automatically per manifest
   (see §3).
4. **`build_jsonl.py`** — combines every `*.reffed.jsonl` into
   `train.jsonl`/`eval.jsonl`, copies `eval.jsonl` → `test.jsonl`. Output
   schema matches this repo's README exactly: `{"audio", "text",
   "ref_audio"?, "ref_text"?}` (the last two omitted, not null, when no
   ref was found).

Config lives in `scripts/formosan/common.py` (paths, thresholds, model
IDs — read its module docstring first, it explains the pipeline order and
rationale inline). Output root: `/mnt/md1/user_wayne/formosan_final/`
(`audio/`, `manifests/`, `higgs_jsonl/`).

**Validated end-to-end** with a `--limit 20`-per-config smoke test
covering both `ithuan_formosan` (speaker-column path) and
`ntu_formosan_corpus` (embedding-cluster path, including empty-manifest,
singleton-speaker, and all-noise edge cases) → `build_jsonl.py` → fed the
resulting `train.jsonl` into the *existing* `scripts/prepare_data.py`
(unmodified) → successfully produced `[T, N=8]` audio codes. The smoke
test's output under `formosan_final/` was deleted afterward so the real
run starts clean.

**Not yet run at full scale.** The 4 real datasets (~500k rows combined)
still need to go through all 4 stages for real. `speaker_embed.py` is the
one stage that needs a GPU to be practical (see §3).

## 5. Pre-existing bugs this work surfaced (now fixed)

`scripts/prepare_data.py` (and `model/modeling.py`'s `get_audio_codec()`)
load the codec via `AutoModel.from_pretrained("bosonai/higgs-audio-v2-tokenizer",
trust_remote_code=True)`. This **failed** with `transformers==5.0.0`
(the repo's pinned version): `KeyError: 'higgs_audio_v2_tokenizer'` /
"Transformers does not recognize this architecture". Root cause: the
codec's `model_type` (`higgs_audio_v2_tokenizer`) is **natively built into
transformers as of 5.3.0** (confirmed from the model card:
`uv pip install "transformers>=5.3.0"`) — it's not a `trust_remote_code`/
`auto_map` situation at all (the checkpoint repo has no custom `.py`
files), the repo's pin was just one minor version too old. This affects
the *entire* repo (training + inference + data prep), not just this data
pipeline.

(Note the 5.3.0 floor covers only the *codec*. The v3 TTS model card asks for
`transformers >= 5.5` for the LM itself, so 5.5 is the real minimum; §8.1
moved the pin to 5.16.1 anyway.)

**Fixed**: bumped `requirements.txt` to `transformers==5.3.0`, verified
the repo's own custom model classes (`model/modeling.py`,
`model/configuration.py`) still import fine, and confirmed the codec now
loads and encodes correctly. If you're setting this repo up fresh
elsewhere, just installing from the (now-updated) `requirements.txt` picks
this up automatically.

## 6. Known open gaps (not yet addressed — flagged, not fixed)

- **`sft.py` has no eval/validation loop and no `--eval-jsonl` argument.**
  Once `eval.jsonl` exists, there is currently nothing in the training
  code that consumes it for validation loss or checkpoint selection.
- **`sft.py` has no `--resume-from-checkpoint`.** A long run that gets
  interrupted currently has to restart from scratch.
- **No batch scoring/eval harness exists** for running the model against
  `test.jsonl` after training and computing objective metrics (e.g. WER
  via ASR back-transcription, speaker similarity, DNSMOS on generated
  audio). `scripts/infer.py` only does single-utterance synthesis.
- **Train/eval leakage across datasets was flagged early, never actually
  checked.** Several `lang_code`s appear in both a train-only dataset
  (`ntu_formosan_corpus`/`nchc_formosan`) and an eval-bearing one
  (`klokah`/`ithuan_formosan`) — e.g. `trv-x-tgdy`, `ckv`, `trv-x-truku`,
  `ami-x-skl`. If the *same recordings or speakers* appear in both, eval
  numbers would be optimistic. Worth a speaker/id-overlap check before
  trusting eval metrics, especially since `ithuan_formosan`/
  `nchc_formosan` have real speaker IDs to check against.

## 7. Running this on a different server

**Import order matters in `scripts/formosan/materialize.py`.** `common.py`
points HF_HOME at the shared cache via `os.environ.setdefault`, but
`huggingface_hub` freezes its cache paths at *import* time. Importing anything
from `huggingface_hub` before `common` therefore sends every download to
`~/.cache/huggingface` instead, silently -- which on this machine is the small
root partition, and put 60GB of klokah parquet there before it was noticed.
`materialize.py` imports `hf_hub_download` after `common` for exactly this
reason; keep it that way, and prefer setting `HF_HOME` in the environment too.

One environment wart to watch for on a fresh machine:
`$HF_HOME/modules/transformers_modules` gets created by whichever process
touches it first, so if anything ever runs as root there, every later
`trust_remote_code` load fails with `PermissionError`. That happened here and
was fixed with `chown`; `HF_MODULES_CACHE=<writable dir>` is the workaround if
you cannot chown.


Nothing here is irreplaceably tied to this machine. What a new server
needs:

- **Same HF account/token access** to the gated `formospeech` org
  datasets (this session used `txya900619`). Set
  `HF_HOME=/path/to/cache` before any HF-related command (this repo's own
  scripts already do `os.environ.setdefault("HF_HOME", ...)` in
  `common.py`, but adjust the hardcoded path there for the new machine).
- **Matching (or adjusted) CUDA version** — `requirements.txt` pins
  `torch==2.9.1+cu128`/`torchaudio==2.9.1+cu128`; the README already notes
  to adjust these if the new machine's CUDA differs. Install via:
  `uv pip install -r requirements.txt --extra-index-url
  https://download.pytorch.org/whl/cu128 --index-strategy
  unsafe-best-match` (the plain PyPI index doesn't carry the `+cu128`
  wheel variants).
- **`speechbrain`, `onnxruntime`, `scikit-learn`** were installed ad hoc
  during the model-selection experiment but are **not required** for the
  production pipeline (final choice was ReDimNet2, loaded via
  `torch.hub`, not speechbrain/onnxruntime) — only `scikit-learn` (for
  HDBSCAN in `pair_ref_audio.py`) is a real dependency; add it to
  `requirements.txt` if it isn't already covered by another package.
- Nothing from `/mnt/md1/user_wayne/formosan_dialects/` or
  `/mnt/md1/user_wayne/chatterbox/` (the old, unrelated project) is a
  dependency of the final pipeline — those were only consulted for
  historical rationale (§2-3) and are not imported/read by any
  `scripts/formosan/*.py` file.
- The repo already has a git remote (`origin/main`) — clone it rather
  than copying files by hand; this doc travels with it.

## 8. Status: the full pipeline has been run

All four stages completed on a **different server** than §1-7 were written on
(`/mnt/md1/user_wayne` does not exist there and `/mnt/md1` is not writable, so
paths moved to `/mnt/md0/user_wayne/...` and are now overridable via
`FORMOSAN_ROOT` / `FORMOSAN_HF_HOME` instead of hardcoded).

Final output in `/mnt/md0/user_wayne/formosan_final/higgs_jsonl/`:

| file | rows | with ref_audio |
|---|---|---|
| `train.jsonl` | 270,812 | 270,812 (100%) |
| `eval.jsonl` | 12,420 | 12,420 (100%) |
| `test.jsonl` | 12,420 | byte-identical copy of eval |

284,655 rows survived the DNSMOS/mandarin filter out of ~530k source rows,
covering all 42 lang_codes, 72GB of 24kHz mono wav. Verified end-to-end
through the unmodified `scripts/prepare_data.py` -> `[T, N=8]` codes.

### 8.1 Environment rebuilt (`pyproject.toml`, cu130)

`requirements.txt` was replaced by `pyproject.toml` (uv). Versions moved
torch 2.9.1+cu128 -> **2.14.0+cu130**, torchcodec 0.8.1 -> **0.16.0**,
transformers 5.3.0 -> **5.16.1**. Three dependency bugs were fixed:

- `datasets` and `scikit-learn` were **missing entirely** from
  `requirements.txt` despite being direct imports (sklearn only worked by
  accident, as a transitive dep of librosa).
- `torchcodec`'s CUDA build could not load **at all** under cu128: its
  `libtorchcodec_core*.so` dlopens `libnppicc.so.12`, which pip ships in
  `nvidia-npp-cu12` but torch never puts on the loader path. That breaks all
  `datasets` audio decoding, so stage 1 was dead on arrival. The cu130 build
  has no such problem, so the workaround is gone.
- `librosa`, `scipy`, `soundfile`, `gradio`, `einops`, `tiktoken`, `orjson`
  were pinned but imported nowhere; `peft` and `deepspeed` *are* imported
  (lazily, by `sft.py`/`infer.py`) but were **not** declared. They are now
  `[lora]` / `[deepspeed]` extras.

### 8.2 torchaudio -> torchcodec

Every audio file read/write in this repo now goes through torchcodec.
`AudioDecoder(src, sample_rate=..., num_channels=1)` replaces
load+downmix+resample in one call and accepts raw `bytes`, so parquet audio
decodes with no temp file.

**torchaudio could not be removed**, for one reason that is not ours:
transformers' own `higgs_audio_v2_tokenizer` is decorated
`@requires(backends=("torchaudio",))` and calls
`torchaudio.functional.resample()` inside `_extract_semantic_features()`.
That is a tensor->tensor resample, the one operation torchcodec has no
equivalent for (it only resamples while decoding/encoding a stream).
`model/modeling.py:_encode_reference` keeps torchaudio for the same reason.

### 8.3 Three scaling bugs found by actually running it

- **`sf.write()` was the stage-1 bottleneck, not the filesystem.** Workers sat
  in `D` state on `jbd2_log_wait_commit` and the first diagnosis blamed ext4
  metadata for 270k small files. It was wrong: torchcodec's `AudioEncoder`
  writes the *same* number of files **21x faster** (551 vs 26 files/sec,
  bit-identical output). libsndfile appears to force a sync per close. A
  planned tar-container redesign was dropped once this was measured -- the
  path-based interface `prepare_data.py` expects never had to change.
- **`speaker_embed.py` OOMed at scale.** It sorted by duration then batched a
  fixed 64, so trailing batches held 64 of the *longest* clips (~1550s of
  padded audio). Now bounded by padded duration with an OOM-halving retry.
  On torch 2.14 the allocator also fragments badly here, so the script sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself (11GB -> 5GB, and
  OOM retries went to zero).
- **`targets[shard::num_shards]` split by manifest count, not work.**
  Manifests sort into an alternating `<config>_eval` / `<config>_train`
  sequence and a klokah train manifest holds up to 70x the audio of its eval
  counterpart, so striding by 6 gave shards 0/2/4 only eval files: **5.6h vs
  133.4h**, a 23.8x imbalance with half the GPUs idle. Replaced with greedy
  LPT packing weighted by total duration -> 63.32h vs 63.47h (1.002x). Every
  shard derives the same assignment independently, so it needs no locking.

### 8.4 Ref-audio pairing, and why HDBSCAN was replaced

`has_usable_speaker_column` required >=2 distinct values. That was wrong for
**single-speaker manifests**: all three ithuan_formosan `eval` splits have one
speaker and were being routed to clustering on ~14-20 rows. The real signal is
not the count but whether the value equals `lang_code` (the degenerate nchc
case), so the test is now keyed on that.

Two empirical results, both worth keeping:

- nchc `ckv`/`trv-x-truku` label speakers `'male'`/`'female'`, which looked
  like gender rather than identity. Embedding all 2,109 ckv rows produced
  **exactly 2 clusters, zero noise, matching the labels** (same-label cosine
  0.717 vs cross-label 0.246): they really are one speaker each.
- klokah's pairwise-cosine distribution is **bimodal** -- modes near 0.25 and
  0.70 with a clear trough at 0.45-0.55 -- while proven single-speaker nchc
  `pyu-*` manifests are unimodal around 0.85. klokah is genuinely
  multi-speaker, and its speakers are cleanly separable.

**HDBSCAN (§3's choice) was replaced by agglomerative average linkage at
`REF_CLUSTER_DISTANCE = 0.40` cosine distance.** §3's experiment was sound but
scoped too narrowly in two ways, and both only show up at klokah's scale:

1. *It only measured manifests with 2-3 speakers.* klokah has dozens per
   config. HDBSCAN is density-based, so a speaker whose recordings vary reads
   as low density and gets fragmented or called noise: 5-36% of klokah rows,
   i.e. no ref at all. Lowering `min_cluster_size` made it **worse**, not
   better (sxr: 11% noise at 5, 45% at 3), which is itself a sign the density
   model does not fit.
2. *It scored with ARI, which is the wrong objective here.* Splitting one
   speaker across several clusters is harmless for pairing -- each cluster is
   still that speaker -- while merging two speakers is the failure that
   matters. ARI punishes the harmless case, which is exactly what made
   agglomerative look "threshold-sensitive" in §3.

Rescored on **cluster purity** (share of rows in a single-speaker cluster)
over the five labelled manifests, agglomerative holds **100% purity across the
whole band d=0.40..0.60** and only fails at 0.65, where ithuan `ami-x-skl`'s
two speakers merge into one cluster and purity drops to 0%. Run
`scripts/formosan/validate_clustering.py` to reproduce that table; it exists
so this threshold never gets changed without re-checking purity.

0.40 is chosen over the more permissive end of the band deliberately:

| | d=0.40 | d=0.50 | d=0.60 |
|---|---|---|---|
| rows with a ref | 99.51% | 99.89% | ~99.97% |
| margin to the purity cliff at 0.65 | **0.25** | 0.15 | 0.05 |
| median candidate pool (klokah) | 357-486 | 441-754 | 464-1801 |

Coverage differs in the third decimal; safety margin differs by multiples, and
the failure is asymmetric -- one unpaired row costs that row (refs are
optional per-sample), one merged cluster points a whole cluster's refs at the
wrong person. The cliff was also located using only 6 speaker pairs, while a
klokah config has hundreds, so the true cliff there is likely earlier than
0.65. Even at 0.40 each row still has ~400 candidates, so nothing is lost to
reduced diversity.

This removed code rather than adding it: the earlier `REF_PAIR_MIN_COSINE`
cosine fallback and its `REF_PAIR_SINGLE_SPEAKER_MEAN_COSINE` gate existed
solely to repair HDBSCAN's noise labels, and both are gone. Coverage went from
79.14% to **99.51%** (rows with no ref: 59,381 -> 1,382).

### 8.5 §6's leakage question, answered: there was leakage

The check finally ran (locally over the manifests, which is fast; scanning the
Hub over the network timed out after 40 minutes on one dataset).

- **Every one of ithuan_formosan `trv-x-tgdy`'s 19 eval rows is a byte-for-byte
  duplicate of a train row** -- same id, duration, dnsmos and ipa;
  `trv-x-truku` adds one more. Since rows are written to `<id>.wav`, a shared
  id is a shared *file*, which is also why those eval rows' `ref_audio` pointed
  at "train" audio. `build_jsonl.py` now drops train rows whose audio appears
  in eval (20 rows) and strips `ref_audio` from train rows that reference eval
  audio (21 rows). Final state: **zero audio overlap, zero cross-split refs in
  either direction.**
- **Text overlap is measured and deliberately left in place.** 13-32% of each
  klokah eval shard's sentences also occur in train (ithuan `trv-x-tgdy`:
  89.5%), as different recordings of the same sentence. This was raised and the
  decision was to keep them: removing them would cost roughly a fifth of the
  eval set, and the audio is disjoint either way (see the two fixes above), so
  the leak is textual, not acoustic.

  **Read eval numbers with that in mind.** Anything sensitive to having seen
  the sentence before -- WER via ASR back-transcription most of all -- will
  read optimistically, and eval loss will sit below what unseen text would
  give. Speaker similarity and DNSMOS are largely unaffected, since neither
  depends on the transcript. If a clean text-held-out number is ever needed,
  filter eval rows whose `ipa` appears in any train manifest; the check that
  produced these figures is in §8.5.

### 8.6 Rows without a reference are dropped (`--require-ref`)

Clustering leaves 1,134 train and 269 eval rows with no ref audio (0.42% /
2.12%) -- rows that ended up alone in their cluster. `build_jsonl.py` now drops
them from both splits by default (`--require-ref all`).

Reference audio is optional per-sample, and the base model card documents a
"Zero-shot TTS" mode that takes none, so at first glance those rows look like
free training signal for that mode. They are not, for this project. Without a
reference the prompt is just `<|tts|> <|text|> ... <|audio|>` -- no speaker
embedding, no default speaker id -- and the voice comes purely from sampling.
Measured directly: six generations of one sentence differing only by seed gave
a mean pairwise speaker similarity of **0.369** (range 0.21-0.59), inside this
project's cross-speaker band (0.25-0.35) and nowhere near its same-speaker band
(0.72-0.88). Every generation is a different person. Upstream
[issue #14](https://github.com/boson-ai/higgs-audio/issues/14) asks how to
stabilize this and is still unanswered.

So for a voice-cloning finetune those rows train the *opposite* of the goal --
invent a voice when no reference is given, rather than follow the reference.
They also make eval metrics ill-defined: speaker similarity has nothing to
compare against. `--require-ref none` keeps them if zero-shot synthesis is ever
wanted.

Beware a terminology trap when reading around this: almost all Higgs
documentation and discussion says "zero-shot voice cloning" to mean *cloning
from a 3-5s clip without finetuning*, which still uses a reference. That is a
different thing from the reference-free "Zero-shot TTS" in the model card's
usage example.

### 8.6.1 Validation loop (`--eval-jsonl`)

`sft.py` now consumes `eval.jsonl`. It evaluates at each epoch boundary, plus
every `--eval-steps` optimizer steps, and keeps the lowest-loss step as
`checkpoint-best/`.

The one subtlety worth preserving: the training loss is normalised **per
token**, not per sample (`all_sum_losses.sum() / total_tokens` in
`model/modeling.py`), so averaging per-batch eval losses would quietly
over-weight short utterances. `evaluate()` instead accumulates the per-codebook
loss sums and token counts across the entire eval set -- both are already
returned as `all_sum_losses` / `all_token_nums` -- and then applies the model's
own aggregation, including the channelwise-weighted branch. The result is
directly comparable to the training loss. `gather_for_metrics` is used rather
than a plain gather because Accelerate pads the last batch of a prepared
dataloader by repeating samples; gathering per-sample rows lets it drop those.

Smoke-tested on the real data (40 train / 16 eval rows through the 4B model
with LoRA): eval loss tracked, `checkpoint-best/` written with
`best_eval_loss` in its `finetune_args.json`, and runs without `--eval-jsonl`
behave exactly as before.

## 9. Still open

- `sft.py` still has **no `--resume-from-checkpoint`**.
- **No batch eval harness** for scoring `test.jsonl` after training (WER via
  ASR back-transcription, speaker similarity, DNSMOS on generated audio).
  `scripts/infer.py` is single-utterance only.
