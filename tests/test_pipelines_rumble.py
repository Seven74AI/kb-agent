"""Rumble pipeline tests — Flaresolverr resolution + CDN download."""
from unittest.mock import patch

from agent.pipelines.rumble import resolve_rumble


class TestResolveRumble:

    @patch("agent.pipelines.rumble._flaresolverr_get")
    def test_resolves_slug_to_real_id_and_formats(self, mock_fs):
        # slug v2y5lfu-… resolves to the REAL id v2vk68c (redirect alias)
        mock_fs.side_effect = [
            '<html><script type="application/ld+json">{"@type":"VideoObject",'
            '"embedUrl":"https://rumble.com/embed/v2vk68c/"}</script></html>',
            '<pre>{"title":"Evidence of Crimes","ua":{"mp4":{'
            '"360":{"url":"https://hugh.cdn.rumble.cloud/video/...kaa.mp4","meta":{"h":360}},'
            '"720":{"url":"https://hugh.cdn.rumble.cloud/video/...baa.mp4","meta":{"h":720}}'
            '}}}</pre>',
        ]
        title, formats = resolve_rumble("https://rumble.com/v2y5lfu-evidence-of-crimes-against-humanity-darkfield-blood-microscopy")
        assert title == "Evidence of Crimes"
        assert [f["height"] for f in formats] == [360, 720]
        assert formats[1]["url"].endswith("baa.mp4")

    @patch("agent.pipelines.rumble._flaresolverr_get")
    def test_no_video_id_raises(self, mock_fs):
        mock_fs.return_value = "<html><body>no embed here</body></html>"
        try:
            resolve_rumble("https://rumble.com/vnosuchvideo")
        except Exception as exc:  # StepError(RETRY)
            assert "video id not found" in str(exc)
        else:
            raise AssertionError("expected StepError")

    @patch("agent.pipelines.rumble._flaresolverr_get")
    def test_no_mp4_formats_raises(self, mock_fs):
        mock_fs.side_effect = [
            '<a href="https://rumble.com/embed/v2vk68c/">embed</a>',
            '<pre>{"title":"T","ua":{}}</pre>',
        ]
        try:
            resolve_rumble("https://rumble.com/v2vk68c-foo")
        except Exception as exc:
            assert "No mp4 formats" in str(exc)
        else:
            raise AssertionError("expected StepError")
