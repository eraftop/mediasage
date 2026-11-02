"""Tests for playlist generation."""

import json
from unittest.mock import MagicMock, patch

import pytest


def _parse_sse_events(generator):
    """Parse SSE events from generate_playlist_stream into (event, data) tuples."""
    events = []
    for raw in generator:
        if raw.startswith(":"):
            continue  # SSE comment (heartbeat)
        for line in raw.strip().split("\n"):
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                events.append((event_type, json.loads(line[len("data: "):])))
    return events


class TestPlaylistGeneration:
    """Tests for playlist generation (streaming)."""

    def test_generate_validates_tracks_against_library(self, mocker, mock_plex_tracks):
        """Generated playlist should only contain tracks from library."""
        from backend.generator import generate_playlist_stream
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content=json.dumps([
                {"artist": "Radiohead", "album": "The Bends", "title": "Fake Plastic Trees"},
                {"artist": "Pearl Jam", "album": "Ten", "title": "Black"},
            ]),
            input_tokens=1000,
            output_tokens=100,
            model="test-model"
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client.analyze.return_value = LLMResponse(
                content='{"title": "Test", "narrative": "Test narrative."}',
                input_tokens=100, output_tokens=50, model="test-model"
            )
            mock_client.parse_json_response.side_effect = [
                json.loads(mock_response.content),
                {"title": "Test", "narrative": "Test narrative."},
            ]
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_current_media_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks[:5]
                mock_plex.return_value = mock_plex_client

                with patch("backend.generator.library_cache.has_cached_tracks", return_value=False):
                    with patch("backend.generator.library_cache.save_result", return_value="abc123"):
                        events = _parse_sse_events(generate_playlist_stream(
                            prompt="90s alternative",
                            genres=["Alternative", "Rock"],
                            decades=["1990s"],
                            track_count=25,
                            exclude_live=True,
                        ))

                        # Collect track rating keys from track batch events
                        track_keys = []
                        for etype, data in events:
                            if etype == "tracks":
                                track_keys.extend(t["rating_key"] for t in data["batch"])

                        library_keys = {t.rating_key for t in mock_plex_tracks}
                        for key in track_keys:
                            assert key in library_keys

    def test_generate_handles_empty_filter_results(self, mocker):
        """Should emit error event when no tracks match filters."""
        from backend.generator import generate_playlist_stream

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_llm.return_value = MagicMock()

            with patch("backend.generator.get_current_media_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = []
                mock_plex.return_value = mock_plex_client

                with patch("backend.generator.library_cache.has_cached_tracks", return_value=False):
                    events = _parse_sse_events(generate_playlist_stream(
                        prompt="nonexistent genre",
                        genres=["Nonexistent"],
                        decades=["1800s"],
                        track_count=25,
                        exclude_live=True,
                    ))

                    error_events = [d for t, d in events if t == "error"]
                    assert len(error_events) == 1
                    assert "No tracks" in error_events[0]["message"]

    def test_fuzzy_matching_finds_similar_titles(self, mocker, mock_plex_tracks):
        """Should fuzzy match LLM responses to library tracks."""
        from backend.generator import generate_playlist_stream
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content=json.dumps([
                {"artist": "Radiohead", "album": "The Bends", "title": "Fake Plastic Tree"},
            ]),
            input_tokens=1000,
            output_tokens=100,
            model="test-model"
        )

        with patch("backend.generator.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client.analyze.return_value = LLMResponse(
                content='{"title": "Test", "narrative": "Test."}',
                input_tokens=100, output_tokens=50, model="test-model"
            )
            mock_client.parse_json_response.side_effect = [
                json.loads(mock_response.content),
                {"title": "Test", "narrative": "Test."},
            ]
            mock_llm.return_value = mock_client

            with patch("backend.generator.get_current_media_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_tracks_by_filters.return_value = mock_plex_tracks[:5]
                mock_plex.return_value = mock_plex_client

                with patch("backend.generator.library_cache.has_cached_tracks", return_value=False):
                    with patch("backend.generator.library_cache.save_result", return_value="abc123"):
                        events = _parse_sse_events(generate_playlist_stream(
                            prompt="radiohead",
                            genres=["Alternative"],
                            decades=["1990s"],
                            track_count=25,
                            exclude_live=True,
                        ))

                        # Should complete without error
                        error_events = [d for t, d in events if t == "error"]
                        assert len(error_events) == 0


class TestTrackMatching:
    """Tests for track matching utilities."""

    def test_simplify_string_removes_punctuation(self):
        """Should remove punctuation from strings."""
        from backend.plex_client import simplify_string

        assert simplify_string("Don't Stop") == "dont stop"
        assert simplify_string("Rock & Roll") == "rock  roll"
        assert simplify_string("(Remastered)") == "remastered"

    def test_simplify_string_normalizes_unicode(self):
        """Should normalize unicode characters."""
        from backend.plex_client import simplify_string

        assert simplify_string("Café") == "cafe"
        assert simplify_string("Motörhead") == "motorhead"

    def test_normalize_artist_handles_and_variations(self):
        """Should handle 'and' vs '&' variations."""
        from backend.plex_client import normalize_artist

        variations = normalize_artist("Simon & Garfunkel")
        assert "Simon & Garfunkel" in variations
        assert "Simon and Garfunkel" in variations

        variations = normalize_artist("Tom and Jerry")
        assert "Tom and Jerry" in variations
        assert "Tom & Jerry" in variations

class TestNarrativeGeneration:
    """Tests for curator narrative generation."""

    @pytest.fixture(autouse=True)
    def no_retry_delay(self, monkeypatch):
        monkeypatch.setattr("backend.generator.NARRATIVE_RETRY_DELAY", 0)

    def test_generate_narrative_returns_title_and_narrative(self, mocker):
        """Should generate creative title and narrative from track selections."""
        from backend.generator import generate_narrative
        from backend.llm_client import LLMResponse

        track_selections = [
            {"artist": "Radiohead", "title": "Fake Plastic Trees", "reason": "Melancholic atmosphere"},
            {"artist": "Pearl Jam", "title": "Black", "reason": "Emotional depth"},
        ]

        mock_response = LLMResponse(
            content='{"title": "Rainstorm Reverie", "narrative": "This playlist weaves through Radiohead\'s Fake Plastic Trees and Pearl Jam\'s Black for a moody journey."}',
            input_tokens=500,
            output_tokens=50,
            model="test-model"
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        mock_client.parse_json_response.return_value = {
            "title": "Rainstorm Reverie",
            "narrative": "This playlist weaves through Radiohead's Fake Plastic Trees and Pearl Jam's Black for a moody journey."
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert "Rainstorm Reverie" in title
        # Title should include date suffix
        assert " - " in title
        assert "Fake Plastic Trees" in narrative or len(narrative) > 0

    def test_generate_narrative_fallback_on_failure(self, mocker):
        """Should return fallback title on LLM failure."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.analyze.side_effect = Exception("LLM error")

        title, narrative = generate_narrative(track_selections, mock_client)

        # Should return fallback title with date
        assert "Playlist" in title
        assert narrative == ""
        assert mock_client.analyze.call_count == 2  # retried once

    def test_generate_narrative_retries_transient_failure(self):
        """A rate limit or timeout on the first try shouldn't leave a generic title (#14)."""
        from backend.generator import generate_narrative

        mock_client = MagicMock()
        mock_client.analyze.side_effect = [RuntimeError("Error code: 429"), MagicMock()]
        mock_client.parse_json_response.return_value = {"title": "Velvet Midnight", "narrative": "Smoky."}

        title, narrative = generate_narrative([{"artist": "A", "title": "B"}], mock_client)

        assert title.startswith("Velvet Midnight - ")
        assert narrative == "Smoky."

    def test_generate_narrative_accepts_key_variants(self):
        from backend.generator import generate_narrative

        for reply in (
            {"Title": "Velvet Midnight", "Narrative": "Smoky."},
            {"name": "Velvet Midnight", "description": "Smoky."},
            {"playlist_title": "Velvet Midnight", "story": "Smoky."},
        ):
            mock_client = MagicMock()
            mock_client.parse_json_response.return_value = reply
            title, narrative = generate_narrative([{"artist": "A", "title": "B"}], mock_client)
            assert title.startswith("Velvet Midnight - "), reply
            assert narrative == "Smoky.", reply

    def test_generate_narrative_falls_back_to_request(self):
        """When the AI title can't be had, name the playlist after the request."""
        from backend.generator import generate_narrative

        mock_client = MagicMock()
        mock_client.analyze.side_effect = TimeoutError("timed out")

        title, narrative = generate_narrative(
            [{"artist": "A", "title": "B"}], mock_client, "late night mellow jazz"
        )

        assert title.startswith("Late Night Mellow Jazz - ")
        assert narrative == ""

    def test_fallback_title_is_kept_short(self):
        from backend.generator import _fallback_title

        title = _fallback_title("upbeat instrumental jazz for a long dinner party with old friends tonight")
        assert len(title) <= 40
        assert title == "Upbeat Instrumental Jazz For A Long"

    def test_generate_narrative_passes_through_long_narrative(self, mocker):
        """Should pass through narrative without truncation (LLM prompt guides length)."""
        from backend.generator import generate_narrative
        from backend.llm_client import LLMResponse

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        long_narrative = "A" * 600
        mock_response = LLMResponse(
            content=json.dumps({"title": "Test", "narrative": long_narrative}),
            input_tokens=500,
            output_tokens=50,
            model="test-model"
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        mock_client.parse_json_response.return_value = {
            "title": "Test",
            "narrative": long_narrative
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        # No truncation - LLM prompt guides length instead
        assert narrative == long_narrative

    def test_generate_narrative_handles_array_wrapped_response(self, mocker):
        """Should handle array-wrapped JSON responses from some LLMs."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        # Some LLMs wrap their response in an array like [{...}]
        mock_client.parse_json_response.return_value = [
            {"title": "Wrapped Title", "narrative": "This is wrapped in an array."}
        ]

        title, narrative = generate_narrative(track_selections, mock_client)

        assert "Wrapped Title" in title
        assert narrative == "This is wrapped in an array."

    def test_generate_narrative_handles_alternate_key_names(self, mocker):
        """Should try alternate keys like description, text, content."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        # LLM uses "description" instead of "narrative"
        mock_client.parse_json_response.return_value = {
            "title": "Alt Key Test",
            "description": "Using description key instead of narrative."
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert "Alt Key Test" in title
        assert narrative == "Using description key instead of narrative."

    def test_generate_narrative_handles_text_key(self, mocker):
        """Should fall back to 'text' key for narrative."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Text Key Test",
            "text": "Using text key."
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Using text key."

    def test_generate_narrative_handles_content_key(self, mocker):
        """Should fall back to 'content' key for narrative."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Content Key Test",
            "content": "Using content key."
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Using content key."

    def test_generate_narrative_empty_array_returns_fallback(self, mocker):
        """Should handle empty array response gracefully."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = []

        title, narrative = generate_narrative(track_selections, mock_client)

        # Should return fallback
        assert "Playlist" in title
        assert narrative == ""

    def test_generate_narrative_prefers_narrative_key_over_alternatives(self, mocker):
        """Should prefer 'narrative' key when multiple keys present."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Priority Test",
            "narrative": "Primary value",
            "description": "Should not use this"
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Primary value"

    def test_generate_narrative_empty_string_uses_fallback_key(self, mocker):
        """Should try alternate keys when narrative key is empty string."""
        from backend.generator import generate_narrative

        track_selections = [{"artist": "Test", "title": "Song", "reason": "Test"}]

        mock_client = MagicMock()
        mock_client.parse_json_response.return_value = {
            "title": "Empty Primary Test",
            "narrative": "",
            "description": "Fallback description used"
        }

        title, narrative = generate_narrative(track_selections, mock_client)

        assert narrative == "Fallback description used"


class TestLiveVersionFiltering:
    """Tests for live version detection."""

    def test_is_live_version_detects_live_keyword(self):
        """Should detect 'live' in track or album title."""
        from backend.plex_client import is_live_version

        class MockTrack:
            def __init__(self, title, album_title):
                self.title = title
                self._album_title = album_title

            def album(self):
                return MagicMock(title=self._album_title)

        assert is_live_version(MockTrack("Song - Live", "Album")) is True
        assert is_live_version(MockTrack("Song", "Live at Madison Square Garden")) is True
        assert is_live_version(MockTrack("Song", "Album")) is False

    def test_is_live_version_detects_concert_keyword(self):
        """Should detect 'concert' in track or album title."""
        from backend.plex_client import is_live_version

        class MockTrack:
            def __init__(self, title, album_title):
                self.title = title
                self._album_title = album_title

            def album(self):
                return MagicMock(title=self._album_title)

        assert is_live_version(MockTrack("Song", "Concert Recording")) is True

    def test_is_live_version_detects_date_patterns(self):
        """Should detect date patterns in album titles."""
        from backend.plex_client import is_live_version

        class MockTrack:
            def __init__(self, title, album_title):
                self.title = title
                self._album_title = album_title

            def album(self):
                return MagicMock(title=self._album_title)

        assert is_live_version(MockTrack("Song", "2023-05-15 Show")) is True
        assert is_live_version(MockTrack("Song", "1999/12/31 New Years")) is True
        assert is_live_version(MockTrack("Song", "Regular Album 2023")) is False


class TestArtistFilter:
    """Artist filters reach the track source and the AI instructions (#9, #18)."""

    def _track(self, key, artist):
        from backend.models import Track
        return Track(rating_key=key, title=f"Song {key}", artist=artist, album="A", duration_ms=1)

    def test_fallback_without_cache_filters_by_artist(self, monkeypatch):
        from backend.generator import _get_tracks_from_cache_or_plex

        monkeypatch.setattr("backend.generator.library_cache.has_cached_tracks", lambda: False)
        client = MagicMock()
        client.get_tracks_by_filters.return_value = [
            self._track("1", "Radiohead"), self._track("2", "Oasis"), self._track("3", "radiohead"),
        ]

        tracks = _get_tracks_from_cache_or_plex(
            client, genres=None, decades=None, exclude_live=True, min_rating=0,
            max_tracks_to_ai=500, artists=["Radiohead"],
        )

        assert [t.rating_key for t in tracks] == ["1", "3"]
        assert client.get_tracks_by_filters.call_args.kwargs["limit"] == 0  # fetch all, then narrow

    def test_cache_path_passes_artist_filters(self, monkeypatch):
        from backend.generator import _get_tracks_from_cache_or_plex

        monkeypatch.setattr("backend.generator.library_cache.has_cached_tracks", lambda: True)
        get = MagicMock(return_value=[])
        monkeypatch.setattr("backend.generator.library_cache.get_tracks_by_filters", get)

        _get_tracks_from_cache_or_plex(
            MagicMock(), genres=None, decades=None, exclude_live=True, min_rating=0,
            max_tracks_to_ai=500, artists=["Radiohead"], exclude_artists=["Oasis"],
        )

        assert get.call_args.kwargs["artists"] == ["Radiohead"]
        assert get.call_args.kwargs["exclude_artists"] == ["Oasis"]

    def test_generation_prompt_names_requested_artists(self, monkeypatch):
        from backend import generator
        from backend.llm_client import LLMResponse

        monkeypatch.setattr(generator, "NARRATIVE_RETRY_DELAY", 0)
        monkeypatch.setattr(generator, "_get_tracks_from_cache_or_plex",
                            lambda *a, **k: [self._track("1", "Radiohead")])
        monkeypatch.setattr(generator.library_cache, "has_cached_tracks", lambda: True)
        llm = MagicMock()
        llm.generate.return_value = LLMResponse(content="[]", input_tokens=1, output_tokens=1, model="m")
        llm.parse_json_response.return_value = []
        monkeypatch.setattr(generator, "get_llm_client", lambda: llm)
        monkeypatch.setattr(generator, "get_current_media_client", lambda: MagicMock())

        list(generator.generate_playlist_stream(prompt="a Radiohead playlist", genres=[], decades=[],
                                                artists=["Radiohead"]))

        prompt = llm.generate.call_args.args[0]
        assert "these artists, so every track is by them: Radiohead" in prompt
        system = llm.generate.call_args.args[1]
        assert "unless the user asked for specific artists" in system


class TestNoMatches:
    """When nothing the AI picked is in the library, say so instead of an empty playlist."""

    @pytest.fixture(autouse=True)
    def no_retry_delay(self, monkeypatch):
        monkeypatch.setattr("backend.generator.NARRATIVE_RETRY_DELAY", 0)

    def _run(self, monkeypatch, picks, narrative=None):
        from backend import generator
        from backend.llm_client import LLMResponse
        from backend.models import Track

        library = [Track(rating_key="1", title="Creep", artist="Radiohead", album="Pablo Honey", duration_ms=1)]
        monkeypatch.setattr(generator, "_get_tracks_from_cache_or_plex", lambda *a, **k: library)
        monkeypatch.setattr(generator.library_cache, "has_cached_tracks", lambda: True)
        llm = MagicMock()
        llm.generate.return_value = LLMResponse(content="[]", input_tokens=1, output_tokens=1, model="m")
        llm.parse_json_response.side_effect = [picks, narrative or {}]
        monkeypatch.setattr(generator, "get_llm_client", lambda: llm)
        monkeypatch.setattr(generator, "get_current_media_client", lambda: MagicMock())
        events = list(generator.generate_playlist_stream(prompt="rainy day", genres=[], decades=[]))
        return events, llm

    def test_no_matching_picks(self, monkeypatch):
        events, llm = self._run(monkeypatch, [{"artist": "Nobody", "title": "Imaginary Song"}])

        assert "event: error" in events[-1]
        assert "None of the AI's 1 picks matched tracks in your library" in events[-1]
        llm.analyze.assert_not_called()  # no title/story call for an empty playlist

    def test_no_picks(self, monkeypatch):
        events, _ = self._run(monkeypatch, [])
        assert "The AI didn't pick any tracks" in events[-1]

    def test_non_object_picks_are_skipped(self, monkeypatch):
        events, _ = self._run(
            monkeypatch,
            ["Radiohead - Creep", {"artist": "Radiohead", "title": "Creep"}],
            narrative={"title": "Rainy", "narrative": "Wet."},
        )

        complete = next(e for e in events if "event: complete" in e)
        assert '"track_count": 1' in complete

class TestAlbumCap:
    """Tests for max_tracks_per_album, which caps how many tracks one album may
    contribute to a playlist.

    The system prompt has always asked the model to "vary the selection - don't
    pick too many tracks from the same album", but nothing enforced it. Seed
    playlists suffer most: the filtered candidate pool is itself clustered by
    album, so the model picks from a biased pool and is then merely asked to be
    varied.
    """

    @staticmethod
    def _album_heavy_library():
        """Six tracks: five from one album, one from another."""
        from backend.models import Track
        tracks = [
            Track(
                rating_key=str(i),
                title=f"Heavy Track {i}",
                artist="One Band",
                album="One Album",
                duration_ms=200000,
                year=2001,
                genres=["Rock"],
            )
            for i in range(1, 6)
        ]
        tracks.append(
            Track(
                rating_key="9",
                title="Lonely Track",
                artist="Other Band",
                album="Other Album",
                duration_ms=200000,
                year=2002,
                genres=["Rock"],
            )
        )
        return tracks

    def _run(self, tracks, selections, **kwargs):
        """Drive generate_playlist_stream with a fixed LLM reply."""
        from backend.generator import generate_playlist_stream
        from backend.llm_client import LLMResponse

        reply = LLMResponse(
            content=json.dumps(selections), input_tokens=10, output_tokens=10,
            model="test-model",
        )
        with patch("backend.generator.get_llm_client") as mock_llm:
            client = MagicMock()
            client.generate.return_value = reply
            client.analyze.return_value = LLMResponse(
                content='{"title": "T", "narrative": "N."}',
                input_tokens=1, output_tokens=1, model="test-model",
            )
            client.parse_json_response.side_effect = [
                selections, {"title": "T", "narrative": "N."},
            ]
            mock_llm.return_value = client
            with patch("backend.generator.get_current_media_client") as mock_media:
                media = MagicMock()
                # Mock BOTH paths: generate_playlist_stream calls
                # get_tracks_by_filters only when genres/decades/rating/artists
                # are set, and get_random_tracks otherwise. Mocking one leaves
                # the other returning a MagicMock, which matches nothing and
                # looks exactly like a broken cap.
                media.get_tracks_by_filters.return_value = tracks
                media.get_random_tracks.return_value = tracks
                mock_media.return_value = media
                with patch("backend.generator.library_cache.has_cached_tracks", return_value=False):
                    with patch("backend.generator.library_cache.save_result", return_value="x"):
                        events = _parse_sse_events(
                            generate_playlist_stream(track_count=5, **kwargs)
                        )
        keys = []
        for etype, data in events:
            if etype == "tracks":
                keys.extend(t["rating_key"] for t in data["batch"])
        return keys, {k: v for k, v in events}

    def test_without_a_cap_one_album_can_dominate(self):
        """Baseline: the current behaviour, which is what the cap exists to fix."""
        tracks = self._album_heavy_library()
        selections = [
            {"artist": "One Band", "album": "One Album", "title": f"Heavy Track {i}"}
            for i in range(1, 6)
        ]
        keys, _ = self._run(tracks, selections, prompt="rock")
        assert len(keys) == 5
        assert keys == ["1", "2", "3", "4", "5"]

    def test_cap_limits_tracks_from_one_album(self):
        """With max_tracks_per_album=2, only two of the five may be used."""
        tracks = self._album_heavy_library()
        selections = [
            {"artist": "One Band", "album": "One Album", "title": f"Heavy Track {i}"}
            for i in range(1, 6)
        ]
        keys, _ = self._run(tracks, selections, prompt="rock", max_tracks_per_album=2)
        from_one_album = [k for k in keys if k in {"1", "2", "3", "4", "5"}]
        assert len(from_one_album) == 2, f"expected 2 from One Album, got {from_one_album}"

    def test_cap_backfills_from_other_albums(self):
        """A capped slot is refilled from the library rather than left empty."""
        tracks = self._album_heavy_library()
        selections = [
            {"artist": "One Band", "album": "One Album", "title": f"Heavy Track {i}"}
            for i in range(1, 6)
        ]
        keys, _ = self._run(tracks, selections, prompt="rock", max_tracks_per_album=2)
        assert "9" in keys, "the track from Other Album should have been backfilled in"

    def test_cap_of_zero_means_no_limit(self):
        """0 preserves the previous behaviour exactly, so the default is safe."""
        tracks = self._album_heavy_library()
        selections = [
            {"artist": "One Band", "album": "One Album", "title": f"Heavy Track {i}"}
            for i in range(1, 6)
        ]
        keys, _ = self._run(tracks, selections, prompt="rock", max_tracks_per_album=0)
        assert keys == ["1", "2", "3", "4", "5"]

    def test_same_album_name_by_different_artists_counts_separately(self):
        """'Greatest Hits' is not one album. Cap on (artist, album), not album."""
        from backend.models import Track
        tracks = [
            Track(rating_key="a1", title="A1", artist="Band A", album="Greatest Hits",
                  duration_ms=1000, year=2000, genres=["Rock"]),
            Track(rating_key="a2", title="A2", artist="Band A", album="Greatest Hits",
                  duration_ms=1000, year=2000, genres=["Rock"]),
            Track(rating_key="b1", title="B1", artist="Band B", album="Greatest Hits",
                  duration_ms=1000, year=2000, genres=["Rock"]),
        ]
        selections = [
            {"artist": "Band A", "album": "Greatest Hits", "title": "A1"},
            {"artist": "Band A", "album": "Greatest Hits", "title": "A2"},
            {"artist": "Band B", "album": "Greatest Hits", "title": "B1"},
        ]
        keys, _ = self._run(tracks, selections, prompt="hits", max_tracks_per_album=2)
        assert set(keys) == {"a1", "a2", "b1"}, (
            "Band B's Greatest Hits must not be capped by Band A's"
        )
