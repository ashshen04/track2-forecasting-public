"""## Executive summary (read this first)

Use a synthetic image and team to check a draft can be packed by the shared toolkit.
The draft must not reuse a fixture team ID. Bad image references must be refused.
No real credential, registry request or competition upload is used.
"""

import json
import zipfile

import pytest
from qfbench2_common.contracts.descriptor import SubmissionDescriptor
from qfbench2_common.contracts.errors import ContractError
from qfbench2_common.team_claim import pack_submission, derive_team_alias

from scripts.prepare_submission import submission_draft


def test_draft_packs_with_the_actual_team_and_house_disclosure(tmp_path):
    image = "docker.io/synthetic-team/forecaster@sha256:" + "a" * 64
    draft = submission_draft(image)
    assert "team_id" not in draft and "descriptor_digest" not in draft
    out = tmp_path / "submission.zip"
    key = "synthetic-test-key"
    pack_submission(draft, 123, key, out)
    with zipfile.ZipFile(out) as archive:
        assert sorted(archive.namelist()) == ["submission.json", "team-claim.json"]
        raw = archive.read("submission.json")
        descriptor = json.loads(raw)
        assert key.encode() not in raw
        assert key.encode() not in archive.read("team-claim.json")
        SubmissionDescriptor.from_mapping(descriptor)
        assert descriptor["team_id"] == derive_team_alias(123, key)
        assert descriptor["competition_id"] == "agenthon2026-forecasting-dev"
        assert descriptor["models"][0]["name"] == "nvidia/nemotron-3-super-120b-a12b"
        assert descriptor["models"][0]["training_cutoff"] == "unpublished"


@pytest.mark.parametrize("image", ["user/image:latest", "docker.io/user/image@sha256:short",
                                 "docker.io/user/image:latest@sha256:" + "a" * 64])
def test_draft_refuses_noncanonical_image_references(image):
    with pytest.raises((ValueError, ContractError)):
        submission_draft(image)
