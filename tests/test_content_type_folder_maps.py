"""Regression: content-type folder maps stay unified and complete."""

import pytest

from agent.archival import CONTENT_TYPE_TO_MINIO_FOLDER, CONTENT_TYPE_TO_VAULT_FOLDER
from agent.core import note as note_module
from agent.pipelines import import_all_pipelines, list_pipelines
from agent.pipelines import see_also as see_also_module


@pytest.fixture(scope="module", autouse=True)
def _load_pipelines():
    import_all_pipelines()


@pytest.mark.parametrize(
    "content_type",
    ["rumble", "substack_video", "mega_video"],
)
def test_video_content_types_map_to_videos_folder(content_type):
    assert CONTENT_TYPE_TO_VAULT_FOLDER[content_type] == "videos"
    assert CONTENT_TYPE_TO_MINIO_FOLDER[content_type] == "videos"


def test_every_registered_pipeline_has_folder_mappings():
    missing_vault = [ct for ct in list_pipelines() if ct not in CONTENT_TYPE_TO_VAULT_FOLDER]
    missing_minio = [ct for ct in list_pipelines() if ct not in CONTENT_TYPE_TO_MINIO_FOLDER]
    assert missing_vault == [], f"missing vault map: {missing_vault}"
    assert missing_minio == [], f"missing minio map: {missing_minio}"


def test_note_and_see_also_import_shared_vault_map():
    assert note_module.CONTENT_TYPE_FOLDERS is CONTENT_TYPE_TO_VAULT_FOLDER
    assert see_also_module.CONTENT_TYPE_FOLDERS is CONTENT_TYPE_TO_VAULT_FOLDER
