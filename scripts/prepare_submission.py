"""## Executive summary (read this first)

Create a Development submission draft from the shared toolkit's published fixture.
Example: python scripts/prepare_submission.py --image docker.io/user/image@sha256:...
--out /private/tmp/t2-submission/submission.json. Use a pushed, anonymously pullable
image. The official pack command adds the real team ID, digest and team proof.
This script never reads a Team Key and does not publish or upload anything.
"""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest
from qfbench2_common.contracts.errors import ContractError

ROOT = Path(__file__).resolve().parents[1]


def submission_draft(image: str) -> dict:
    """Validate all non-team fields using the shared C5 parser, then remove placeholders."""
    try:
        location, digest = image.split("@")
        registry, repository = location.split("/", 1)
    except ValueError:
        raise ValueError("Use registry/repository@sha256:<64 hex>, without a tag.") from None
    fixture = files("qfbench2_common").joinpath("contracts/fixtures/c5/forecasting_dev.json")
    draft = json.loads(fixture.read_text(encoding="utf-8"))
    draft["image"] = {"registry": registry, "repository": repository, "digest": digest}
    draft["image_access"] = "public"
    draft["license"] = "MIT"
    draft["models"] = [{
        "name": "nvidia/nemotron-3-super-120b-a12b",
        "version": "rl-030326-fp8",
        "revision": "rl-030326-fp8",
        "training_cutoff": "unpublished",
        "access": "api",
    }]
    # The fixture's team ID is only used for format validation. Never put it into the draft.
    SubmissionDescriptor.from_mapping(seal_descriptor_digest(draft))
    draft.pop("team_id")
    draft.pop("descriptor_digest")
    return draft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Pushed public image reference by digest")
    parser.add_argument("--out", type=Path, required=True, help="New JSON file outside this repo")
    args = parser.parse_args(argv)
    try:
        destination = args.out.expanduser().resolve()
        if destination == ROOT or ROOT in destination.parents:
            raise ValueError("Keep submission drafts and zip files outside this public repository.")
        draft = submission_draft(args.image)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(draft, indent=2) + "\n")
    except (ValueError, ContractError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote submission draft: {destination}")
    print("Next: qfbench2 submission pack with your real team number and hidden Team Key prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
