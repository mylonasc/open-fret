# Code, training data and model-release rights

Reviewed **2026-11-09**. This is a provenance/licensing check, not legal advice
or a determination of copyright exceptions in a particular jurisdiction.

## Repository code

The repository README already declared MIT. The root [LICENSE](../LICENSE)
now contains the standard MIT text for original Open-Fret software and
documentation, to the extent the contributors own the relevant rights.
Third-party components retain their own licenses/notices. The code license
does **not** grant rights to training data, underlying music/tab arrangements,
or automatically establish a license for trained model artifacts. This review
is not a complete third-party source-code/dependency license audit.

## Dataset actually used

The download pipeline uses **`vldsavelyev/guitar_tab`**, not a similarly named
dataset or the dataset used in the Fretting-Transformer paper. The published
metadata inspected resolves to revision
`e6a6dbbccf7cad2e057770cae5ebf8257ec36382`. That identifies the reviewed upstream
snapshot; it does not independently prove the revision of every historical
local training download.

| Evidence | Finding |
| --- | --- |
| [Pinned dataset card](https://huggingface.co/datasets/vldsavelyev/guitar_tab/blob/e6a6dbbccf7cad2e057770cae5ebf8257ec36382/README.md) | No `license:` field and no express licensing/permission terms in the card. Describes GuitarPro tablatures converted into AlphaTex. |
| [Hugging Face API metadata/file listing](https://huggingface.co/api/datasets/vldsavelyev/guitar_tab) | No declared license in card metadata/tags; the listed repository files contain no standalone license file. Public/ungated access is not a rights grant. |
| [Pinned dataset loader](https://huggingface.co/datasets/vldsavelyev/guitar_tab/blob/e6a6dbbccf7cad2e057770cae5ebf8257ec36382/guitar_tab.py) | `DatasetInfo` supplies no license. Its description identifies GP3/GP4/GP5 files downloaded from a RuTracker archive, topic `2888130`. This is an upstream provenance statement, not proof of authorization or infringement. |
| [Linked upstream repository](https://github.com/vladsavelyev/guitartab/tree/73f47081b1c80e2305b75413890859ff8bd53bdd) | Its README describes conversion of an archive of tablatures. The inspected Git tree contains no license/terms/notice file; GitHub metadata reports `license: null`, and the license endpoint returns 404. A converter's code license, if later supplied, would not by itself license the songs/tab arrangements. |

**Conclusion: training-data rights are not established by the published material
reviewed.** This is not an MIT, CC0, or confirmed public-domain training corpus
on the evidence available. Absence of a license is not an affirmative ban on
every possible use, but it provides no express permission to rely on here.

The archive's internal files, the original forum's terms, individual works'
ownership/permissions, and private agreements were **not** audited. A license
embedded elsewhere would need its scope and the grantor's authority checked;
the conclusion above concerns the publicly documented grant, not proof that no
permission could exist anywhere. The fact that the uploader also published
trained models is not evidence that downstream release rights were cleared.

## Can weights trained on it be released?

**We cannot confirm a license-based right to release the current weights from
these sources. Do not describe the model as cleared for unrestricted public or
commercial release based on this dataset's availability or this repository's
MIT license.**

Training-data licensing, the legality of training, and the status/distribution
of weights are related but distinct questions. It is not correct to assume
that all trained weights are automatically derivative copies of every training
work, nor that weights are automatically free of third-party rights. Applicable
copyright exceptions, contractual terms, database rights and possible memorized
expression depend on the facts and jurisdiction. Research or noncommercial use
is not a blanket exemption. Architecture changes, format conversion, training
from scratch or further fine-tuning do not themselves clear those questions.

Before publishing these weights:

1. Ask the dataset maintainer for the exact license and provenance/authority to
   grant rights in the underlying material, including training and distribution
   of resulting weights for the intended commercial/noncommercial use. An
   uploader's permission alone may not cover third-party compositions or tab
   arrangements they do not control.
2. Have qualified counsel assess the actual source material, applicable laws/
   terms and intended release if relying on anything other than clear rights.
3. If clearance cannot be established, the more defensible path is to train a
   **new model from scratch** on an authorized corpus (original/commissioned
   material with appropriate agreements, or verified public-domain/licensed
   compositions **and** arrangements). Fine-tuning the existing weights on
   clean data is not a demonstrated legal cure.
4. Separately review any pretrained initialization's model license if using
   `--init-from`. Architecture reuse without loading pretrained weights is not
   the same as using another model's weights.

Until then, keep dataset/checkpoint publication on hold and publish original
code separately under MIT, subject to third-party code obligations. No dataset
or weights were published as part of adding the repository license.

General background on missing licenses:
[GitHub Choose a License: No License](https://choosealicense.com/no-permission/).
