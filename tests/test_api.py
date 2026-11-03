"""Tests for API endpoints."""

import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.models import DefaultsConfig


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    # Import here to avoid module-level import issues
    from backend.main import app
    return TestClient(app)


def create_mock_config(
    plex_url="http://test:32400",
    plex_token="token",
    music_library="Music",
    llm_provider="anthropic",
    llm_api_key="key",
    model_analysis="claude-sonnet-4-5",
    model_generation="claude-haiku-4-5",
    track_count=25,
    ollama_url="http://localhost:11434",
    custom_url="",
    custom_context_window=32768,
):
    """Create a properly structured mock config."""
    mock = MagicMock()
    mock.media_server = "plex"
    mock.jellyfin.url = ""
    mock.jellyfin.token = ""
    mock.jellyfin.music_library = "Music"
    mock.plex.url = plex_url
    mock.plex.token = plex_token
    mock.plex.music_library = music_library
    mock.llm.provider = llm_provider
    mock.llm.api_key = llm_api_key
    mock.llm.model_analysis = model_analysis
    mock.llm.model_generation = model_generation
    mock.llm.ollama_url = ollama_url
    mock.llm.custom_url = custom_url
    mock.llm.custom_context_window = custom_context_window
    mock.defaults = DefaultsConfig(track_count=track_count)
    return mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_status(self, client):
        """Should return health status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_current_media_client") as mock_plex:
                mock_config.return_value = create_mock_config()
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert data["status"] == "healthy"

    def test_health_check_shows_plex_status(self, client):
        """Should show Plex connection status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_current_media_client") as mock_plex:
                mock_config.return_value = create_mock_config()
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "plex_connected" in data
                assert data["plex_connected"] is True

    def test_health_check_shows_llm_status(self, client):
        """Should show LLM configuration status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_current_media_client") as mock_plex:
                mock_config.return_value = create_mock_config(llm_api_key="key")
                mock_plex.return_value = None  # No Plex client

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "llm_configured" in data
                assert data["llm_configured"] is True


class TestConfigEndpoints:
    """Tests for configuration endpoints."""

    def test_get_config_returns_safe_values(self, client):
        """GET /api/config should return config without secrets."""
        with patch("backend.main.get_config") as mock_get_config:
            with patch("backend.main.get_plex_client") as mock_plex:
                mock_get_config.return_value = create_mock_config(
                    plex_url="http://test:32400",
                    plex_token="secret-token",
                    llm_provider="anthropic",
                    llm_api_key="secret-api-key",
                )
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/config")

                assert response.status_code == 200
                data = response.json()

                # Should include URL but not token
                assert data["plex_url"] == "http://test:32400"
                assert "secret-token" not in str(data)

                # Should show provider but not API key
                assert data["llm_provider"] == "anthropic"
                assert "api_key" not in data
                assert "secret-api-key" not in str(data)

    def test_post_config_validates_plex_url(self, client):
        """POST /api/config should validate Plex URL format."""
        with patch("backend.main.update_config_values") as mock_update:
            with patch("backend.main.get_plex_client") as mock_plex:
                with patch("backend.main.init_plex_client"):
                    mock_config = create_mock_config(plex_url="http://new-server:32400")
                    mock_update.return_value = mock_config
                    mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                    response = client.post(
                        "/api/config",
                        json={"plex_url": "http://new-server:32400"}
                    )

                    assert response.status_code == 200

    def test_post_config_updates_llm_provider(self, client):
        """POST /api/config should allow changing LLM provider."""
        with patch("backend.main.update_config_values") as mock_update:
            with patch("backend.main.get_plex_client") as mock_plex:
                with patch("backend.main.init_plex_client"):
                    mock_config = create_mock_config(llm_provider="openai")
                    mock_update.return_value = mock_config
                    mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                    response = client.post(
                        "/api/config",
                        json={"llm_provider": "openai"}
                    )

                    assert response.status_code == 200


class TestIndexPage:
    """Tests for index page serving."""

    def test_index_returns_response(self, client):
        """Should return some response for root path."""
        response = client.get("/")
        # Either returns HTML or JSON message
        assert response.status_code == 200


class TestOllamaEndpoints:
    """Tests for Ollama API endpoints."""

    def test_ollama_status_connected(self, client):
        """GET /api/ollama/status should return connected status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=True,
                    model_count=3,
                    error=None,
                )

                response = client.get("/api/ollama/status")

                assert response.status_code == 200
                data = response.json()
                assert data["connected"] is True
                assert data["model_count"] == 3

    def test_ollama_status_not_connected(self, client):
        """GET /api/ollama/status should return error when not connected."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=False,
                    model_count=0,
                    error="Connection refused",
                )

                response = client.get("/api/ollama/status")

                assert response.status_code == 200
                data = response.json()
                assert data["connected"] is False
                assert data["error"] == "Connection refused"

    def test_ollama_models_list(self, client):
        """GET /api/ollama/models should return list of models."""
        from backend.models import OllamaModel, OllamaModelsResponse

        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.list_ollama_models") as mock_models:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_models.return_value = OllamaModelsResponse(
                    models=[
                        OllamaModel(name="llama3:8b", size=4661224676, modified_at="2024-01-15T00:00:00Z"),
                        OllamaModel(name="mistral:latest", size=3825819904, modified_at="2024-01-14T00:00:00Z"),
                    ],
                    error=None,
                )

                response = client.get("/api/ollama/models")

                assert response.status_code == 200
                data = response.json()
                assert len(data["models"]) == 2
                assert data["models"][0]["name"] == "llama3:8b"

    def test_ollama_model_info(self, client):
        """GET /api/ollama/model-info should return model details."""
        from backend.models import OllamaModelInfo

        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_model_info") as mock_info:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_info.return_value = OllamaModelInfo(
                    name="llama3:8b",
                    context_window=8192,
                    parameter_size="8B",
                )

                response = client.get("/api/ollama/model-info?model=llama3:8b")

                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "llama3:8b"
                assert data["context_window"] == 8192

    def test_ollama_model_info_not_found(self, client):
        """GET /api/ollama/model-info should return 404 for unknown model."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_model_info") as mock_info:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_info.return_value = None  # Model not found

                response = client.get("/api/ollama/model-info?model=nonexistent")

                assert response.status_code == 404

    def test_ollama_status_with_custom_url(self, client):
        """GET /api/ollama/status should accept custom URL parameter."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=True,
                    model_count=1,
                    error=None,
                )

                response = client.get("/api/ollama/status?url=http://custom-host:11434")

                assert response.status_code == 200
                # Verify the custom URL was passed
                mock_status.assert_called_once_with("http://custom-host:11434")


class TestCloudModelsEndpoint:
    """Tests for GET /api/models."""

    def test_lists_models_per_provider(self, client):
        """Should return each cloud provider's models with its defaults."""
        response = client.get("/api/models")

        assert response.status_code == 200
        providers = response.json()["providers"]
        assert set(providers) == {"anthropic", "openai", "gemini"}
        for catalog in providers.values():
            ids = [m["id"] for m in catalog["models"]]
            assert catalog["default_analysis"] in ids
            assert catalog["default_generation"] in ids

    def test_model_entries_include_limits_and_prices(self, client):
        """Entries should carry context, track capacity and prices for the UI."""
        response = client.get("/api/models")

        gemini = response.json()["providers"]["gemini"]["models"]
        flash_lite = next(m for m in gemini if m["id"] == "gemini-3.5-flash-lite")
        assert flash_lite["context_window"] == 1_000_000
        assert flash_lite["max_tracks"] > 20_000
        assert flash_lite["cost_per_million_input"] == 0.30
        assert flash_lite["legacy"] is False


class TestConfigLongContextPricing:
    """Tests for long-context pricing in GET /api/config."""

    def test_included_for_models_with_long_pricing(self, client):
        mock_config = create_mock_config(
            llm_provider="openai", model_analysis="gpt-6.1-sol", model_generation="gpt-6-luna"
        )
        with patch("backend.main.get_config", return_value=mock_config), \
             patch("backend.main.get_plex_client", return_value=None):
            data = client.get("/api/config").json()

        assert data["long_context_threshold"] == 272_000
        assert data["long_context_cost_per_million_input"] == 0.20

    def test_absent_for_flat_priced_models(self, client):
        with patch("backend.main.get_config", return_value=create_mock_config()), \
             patch("backend.main.get_plex_client", return_value=None):
            data = client.get("/api/config").json()

        assert data["long_context_threshold"] is None


class TestLibraryStatsJellyfin:
    """Library stats come from the cache once synced, matching the filters (#30)."""

    def test_uses_cache_when_synced(self, client):
        mock_config = create_mock_config()
        mock_config.media_server = "jellyfin"
        jellyfin = MagicMock(is_connected=MagicMock(return_value=True))
        with patch("backend.main.get_config", return_value=mock_config), \
             patch("backend.main.get_current_media_client", return_value=jellyfin), \
             patch("backend.main.library_cache.has_cached_tracks", return_value=True), \
             patch("backend.main.library_cache.get_sync_state", return_value={"track_count": 42}), \
             patch("backend.main.library_cache.get_cached_genre_decade_stats",
                   return_value={"genres": [{"name": "Jazz", "count": 42}], "decades": []}):
            data = client.get("/api/library/stats").json()

        assert data["total_tracks"] == 42
        assert data["genres"] == [{"name": "Jazz", "count": 42}]
        jellyfin.get_library_stats.assert_not_called()


class TestMediaServerSwitch:
    """Switching between Plex and Jellyfin replaces the cached library."""

    def _post_config(self, client, previous, new, updates):
        before = create_mock_config()
        before.media_server = previous
        after = create_mock_config()
        after.media_server = new
        with patch("backend.main.get_config", return_value=before), \
             patch("backend.main.update_config_values", return_value=after), \
             patch("backend.main.init_plex_client"), \
             patch("backend.main.init_jellyfin_client"), \
             patch("backend.main.get_current_media_client", return_value=None), \
             patch("backend.main._resync_for_new_media_server") as resync:
            response = client.post("/api/config", json=updates)
        assert response.status_code == 200
        return resync

    def test_switching_server_resyncs(self, client):
        resync = self._post_config(client, "jellyfin", "plex", {"media_server": "plex"})
        resync.assert_called_once()

    def test_saving_same_server_does_not_resync(self, client):
        resync = self._post_config(client, "plex", "plex", {"media_server": "plex", "music_library": "Music"})
        resync.assert_not_called()

    def test_plex_setup_selects_plex_and_resyncs_from_jellyfin(self, client):
        temp_client = MagicMock(is_connected=MagicMock(return_value=True))
        temp_client.get_music_libraries.return_value = ["Music"]
        temp_client._server.friendlyName = "My Plex Server"
        previous = create_mock_config()
        previous.media_server = "jellyfin"
        with patch("backend.main.PlexClientInstance", return_value=temp_client), \
             patch("backend.main.get_config", return_value=previous), \
             patch("backend.main.update_config_values") as update, \
             patch("backend.main.init_plex_client"), \
             patch("backend.main._resync_for_new_media_server") as resync:
            response = client.post("/api/setup/validate-plex", json={
                "plex_url": "http://plex:32400", "plex_token": "abc", "music_library": "Music",
            })

        assert response.json()["success"] is True
        assert update.call_args.args[0]["media_server"] == "plex"
        resync.assert_called_once()

    def test_resync_invalidates_now_and_replaces_in_background(self):
        from backend.main import _resync_for_new_media_server

        with patch("backend.main.library_cache") as cache, \
             patch("backend.main.asyncio.to_thread", new=MagicMock()) as to_thread, \
             patch("backend.main.asyncio.create_task"):
            _resync_for_new_media_server()

        cache.invalidate_cache.assert_called_once()
        cache.clear_cache.assert_not_called()  # a running sync holds the DB; never block the request
        to_thread.assert_called_once()
        assert to_thread.call_args.args[0] is cache.replace_library
        assert to_thread.call_args.args[2] == cache.invalidate_cache.return_value


class TestArtistEndpoints:
    """Artist autocomplete and artist filters in the preview (#9)."""

    def test_artists_empty_without_cache(self, client):
        with patch("backend.main.library_cache.has_cached_tracks", return_value=False):
            assert client.get("/api/library/artists?q=radio").json() == []

    def test_artists_search(self, client):
        with patch("backend.main.library_cache.has_cached_tracks", return_value=True), \
             patch("backend.main.library_cache.search_artists",
                   return_value=[{"name": "Radiohead", "count": 2}]) as search:
            data = client.get("/api/library/artists?q=%20radio%20").json()

        assert data == [{"name": "Radiohead", "count": 2}]
        search.assert_called_once_with("radio")

    def test_preview_passes_artist_filters(self, client):
        with patch("backend.main.get_config", return_value=create_mock_config()), \
             patch("backend.main.library_cache.has_cached_tracks", return_value=True), \
             patch("backend.main.library_cache.count_tracks_by_filters", return_value=12) as count:
            data = client.post("/api/filter/preview", json={
                "genres": [], "decades": [], "artists": ["Radiohead"], "exclude_artists": ["Oasis"],
            }).json()

        assert data["matching_tracks"] == 12
        assert count.call_args.kwargs["artists"] == ["Radiohead"]
        assert count.call_args.kwargs["exclude_artists"] == ["Oasis"]


class TestLibraryFreshness:
    """The library cache re-syncs once it's older than the auto-refresh setting (default 24 hours)."""

    def _run(self, *, stale=True, connected=True, syncing=False, hours=24.0):
        from backend.main import _sync_if_stale

        config = create_mock_config()
        config.library_sync_hours = hours
        media_client = MagicMock(is_connected=MagicMock(return_value=connected))
        with patch("backend.main.get_config", return_value=config), \
             patch("backend.main.library_cache") as cache, \
             patch("backend.main.get_current_media_client", return_value=media_client), \
             patch("backend.main.asyncio.to_thread", new=MagicMock()) as to_thread, \
             patch("backend.main.asyncio.create_task"):
            cache.get_sync_progress.return_value = {"is_syncing": syncing}
            cache.is_cache_stale.return_value = stale
            _sync_if_stale()
        return cache, to_thread

    def test_syncs_stale_cache(self):
        cache, to_thread = self._run()
        cache.is_cache_stale.assert_called_once_with(24.0)
        assert to_thread.call_args.args[0] is cache.sync_library

    def test_leaves_fresh_cache(self):
        _, to_thread = self._run(stale=False)
        to_thread.assert_not_called()

    def test_skips_when_disconnected_or_already_syncing(self):
        assert not self._run(connected=False)[1].called
        assert not self._run(syncing=True)[1].called

    def test_custom_interval_and_off(self):
        cache, _ = self._run(hours=168.0)
        cache.is_cache_stale.assert_called_once_with(168.0)
        cache, to_thread = self._run(hours=0)
        cache.is_cache_stale.assert_not_called()
        to_thread.assert_not_called()


class TestApiKeyProviders:
    def test_lists_providers_with_a_key(self, client, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CUSTOM_LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        mock_config = create_mock_config(llm_provider="gemini")
        mock_config.llm.api_keys = {"gemini": "saved"}
        with patch("backend.main.get_config", return_value=mock_config), \
             patch("backend.main.get_current_media_client", return_value=None):
            data = client.get("/api/config").json()

        assert data["api_key_providers"] == ["openai", "gemini"]

class TestAlbumCapWiring:
    """The /api/generate route must actually pass max_tracks_per_album through.

    Exists because the field was added to GenerateRequest and the generator in
    one change and NOT connected in the route -- and the 355 existing tests all
    stayed green, because every one of them calls generate_playlist_stream()
    directly. A field that validates but never reaches the generator is the
    exact shape of bug this asserts against.
    """

    def test_route_forwards_the_cap_to_the_generator(self, client):
        captured = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)
            yield "event: progress\ndata: {}\n\n"

        media = MagicMock()
        media.is_connected.return_value = True
        with patch("backend.main.get_current_media_client", return_value=media), \
             patch("backend.main.get_llm_client", return_value=MagicMock()), \
             patch("backend.main.generate_playlist_stream", side_effect=fake_stream):
            r = client.post("/api/generate/stream", json={
                "prompt": "rock",
                "genres": [],
                "decades": [],
                "max_tracks_per_album": 3,
            })
        assert r.status_code == 200
        # Consume the stream so the generator body actually runs.
        r.read()
        assert captured.get("max_tracks_per_album") == 3, (
            f"route did not forward the cap; got {captured.get('max_tracks_per_album')!r}"
        )

    def test_route_defaults_the_cap_to_zero(self, client):
        captured = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)
            yield "event: progress\ndata: {}\n\n"

        media = MagicMock()
        media.is_connected.return_value = True
        with patch("backend.main.get_current_media_client", return_value=media), \
             patch("backend.main.get_llm_client", return_value=MagicMock()), \
             patch("backend.main.generate_playlist_stream", side_effect=fake_stream):
            r = client.post("/api/generate/stream", json={
                "prompt": "rock", "genres": [], "decades": [],
            })
        assert r.status_code == 200
        r.read()
        assert captured.get("max_tracks_per_album") == 0

    def test_route_rejects_a_negative_cap(self, client):
        r = client.post("/api/generate/stream", json={
            "prompt": "rock", "genres": [], "decades": [], "max_tracks_per_album": -1,
        })
        assert r.status_code == 422

class TestExtendLibrary:
    """Tests for /api/extend/stream -- "which albums am I missing for this playlist".

    The loop is the new logic and the part that can quietly misbehave: it calls
    select_discovery_albums repeatedly (3 albums per call) until the target is
    met, and must deduplicate across rounds and stop paying for rounds that
    repeat themselves.
    """

    @staticmethod
    def _album(artist, album):
        from backend.models import AlbumRecommendation
        return AlbumRecommendation(rank="secondary", album=album, artist=artist)

    @staticmethod
    def _result(tracks=None):
        return {
            "id": "r1", "type": "prompt_playlist", "title": "Test Playlist",
            "prompt": "moody prog", "track_count": 2,
            "snapshot": {"tracks": tracks if tracks is not None else [
                {"artist": "Pink Floyd", "album": "Atom Heart Mother", "title": "A"},
            ]},
        }

    class _FakeVerifier:
        """Stands in for MusicResearchClient.verify_album. Every album is
        verified as-is unless `statuses` maps its (artist, album) to
        "not_found", "unavailable", or an AlbumVerification to return."""

        def __init__(self, statuses=None):
            self.statuses = statuses or {}
            self.calls = []

        async def verify_album(self, artist, album, year=None):
            from backend.models import AlbumVerification
            self.calls.append((artist, album))
            st = self.statuses.get((artist, album), "verified")
            if isinstance(st, AlbumVerification):
                return st
            if st == "verified":
                return AlbumVerification(status="verified", musicbrainz_id="00000000-0000-0000-0000-000000000000",
                                         artist=artist, album=album, year=year)
            return AlbumVerification(status=st)

    def _patches(self, pipeline, result=None, has_cache=True, verifier=None, candidates=None):
        return [
            patch("backend.main._get_research_client",
                  return_value=verifier if verifier is not None else self._FakeVerifier()),
            patch("backend.main._get_pipeline", return_value=pipeline),
            patch("backend.main.library_cache.get_result",
                  return_value=self._result() if result is None else result),
            patch("backend.main.library_cache.has_cached_tracks", return_value=has_cache),
            patch("backend.main.library_cache.get_album_candidates", return_value=candidates or []),
        ]

    def _run(self, pipeline, body, **kw):
        with ExitStack() as stack:
            for p_ in self._patches(pipeline, **kw):
                stack.enter_context(p_)
            from backend.main import app
            r = TestClient(app).post("/api/extend/stream", json=body)
        return r, self._complete(r.text)

    # --- MusicBrainz verification ------------------------------------------

    def test_albums_not_in_musicbrainz_are_dropped_and_reported(self):
        batches = [
            [self._album("A", "1"), self._album("B", "fake"), self._album("C", "3")],
            [self._album("D", "4"), self._album("E", "5"), self._album("F", "6")],
        ]
        pipeline = self._pipeline(batches)
        v = self._FakeVerifier({("B", "fake"): "not_found"})
        r, data = self._run(pipeline, {"result_id": "r1", "target_albums": 4}, verifier=v)
        names = [a["album"] for a in data["albums"]]
        assert "fake" not in names
        assert names == ["1", "3", "4", "5"]
        assert data["rejected"] == [{"artist": "B", "album": "fake", "reason": "not_found"}]
        assert all(a["verified"] is True for a in data["albums"])

    def test_single_is_dropped_with_its_own_reason(self):
        pipeline = self._pipeline([[self._album("Tool", "Opiate²"), self._album("A", "1")], []])
        v = self._FakeVerifier({("Tool", "Opiate²"): "not_album"})
        _, data = self._run(pipeline, {"result_id": "r1", "target_albums": 2}, verifier=v)
        assert [a["album"] for a in data["albums"]] == ["1"]
        assert data["rejected"] == [{"artist": "Tool", "album": "Opiate²", "reason": "not_album"}]

    def test_dropped_album_is_fed_back_so_it_is_not_offered_again(self):
        batches = [
            [self._album("B", "fake"), self._album("A", "1")],
            [self._album("C", "2")],
        ]
        pipeline = self._pipeline(batches)
        self._run(pipeline, {"result_id": "r1", "target_albums": 2},
                  verifier=self._FakeVerifier({("B", "fake"): "not_found"}))
        fed_back = pipeline.select_discovery_albums.call_args_list[1].kwargs["previously_recommended"]
        assert "B|||fake" in fed_back

    def test_canonical_names_and_year_replace_the_models(self):
        from backend.models import AlbumVerification
        pipeline = self._pipeline([[self._album("Radiohead", "OK Computers")]])
        v = self._FakeVerifier({("Radiohead", "OK Computers"): AlbumVerification(
            status="verified", musicbrainz_id="b1392450-e666-3926-a536-22c65f834433",
            artist="Radiohead", album="OK Computer", year=1997)})
        _, data = self._run(pipeline, {"result_id": "r1", "target_albums": 1}, verifier=v)
        a = data["albums"][0]
        assert (a["album"], a["year"], a["musicbrainz_id"]) == (
            "OK Computer", 1997, "b1392450-e666-3926-a536-22c65f834433")

    def test_misspelled_owned_album_is_caught_after_canonicalisation(self):
        """The exclusion list compares the model's spelling; this compares
        MusicBrainz's, which is the one that matches the library."""
        from backend.models import AlbumVerification, TasteProfile
        pipeline = self._pipeline([[self._album("Beatles", "Abey Road"), self._album("X", "Y")], []])
        pipeline.build_playlist_taste_profile.return_value = TasteProfile(
            owned_albums=[{"artist": "The Beatles", "album": "Abbey Road (Remastered)"}])
        v = self._FakeVerifier({("Beatles", "Abey Road"): AlbumVerification(
            status="verified", musicbrainz_id="x", artist="The Beatles", album="Abbey Road", year=1969)})
        _, data = self._run(pipeline, {"result_id": "r1", "target_albums": 2}, verifier=v)
        assert [a["album"] for a in data["albums"]] == ["Y"]
        assert data["rejected"][0]["reason"] == "owned"

    def test_outage_keeps_albums_as_unverified_and_stops_checking(self):
        """An outage is not evidence of invention: keep, flag, and stop paying
        retry delays after two failures in a row."""
        batches = [[self._album("A", "1"), self._album("B", "2"), self._album("C", "3"), self._album("D", "4")]]
        pipeline = self._pipeline(batches)
        v = self._FakeVerifier({("A", "1"): "unavailable", ("B", "2"): "unavailable"})
        _, data = self._run(pipeline, {"result_id": "r1", "target_albums": 4}, verifier=v)
        assert [a["album"] for a in data["albums"]] == ["1", "2", "3", "4"]
        assert all(a["verified"] is False for a in data["albums"])
        assert data["unverified"] == 4
        assert data["rejected"] == []
        assert len(v.calls) == 2, f"kept checking during an outage: {v.calls}"

    def test_single_blip_does_not_stop_checking(self):
        batches = [[self._album("A", "1"), self._album("B", "fake"), self._album("C", "3")], []]
        pipeline = self._pipeline(batches)
        v = self._FakeVerifier({("A", "1"): "unavailable", ("B", "fake"): "not_found"})
        _, data = self._run(pipeline, {"result_id": "r1", "target_albums": 3}, verifier=v)
        assert len(v.calls) == 3
        assert [(a["album"], a["verified"]) for a in data["albums"]] == [("1", False), ("3", True)]

    def test_capped_albums_are_not_looked_up(self):
        """The cap is free; a lookup costs a second of rate limit."""
        batches = [[self._album("A", "1"), self._album("A", "2"), self._album("B", "3")]]
        pipeline = self._pipeline(batches)
        v = self._FakeVerifier()
        self._run(pipeline, {"result_id": "r1", "target_albums": 2, "max_albums_per_artist": 1}, verifier=v)
        assert ("A", "2") not in v.calls

    def test_artist_cap_counts_spelling_variants_as_one_artist(self):
        batches = [[self._album("The Beatles", "Revolver"), self._album("Beatles", "Help!"), self._album("B", "3")]]
        pipeline = self._pipeline(batches)
        _, data = self._run(pipeline, {"result_id": "r1", "target_albums": 2, "max_albums_per_artist": 1})
        assert [a["album"] for a in data["albums"]] == ["Revolver", "3"]

    def test_twenty_is_reachable(self):
        """Reported 2026-11-03: asked for 20, got 7. Six rounds of three could
        never exceed 18; batches of ten within the round budget can."""
        batches = [[self._album(f"Artist {r}-{i}", f"A{r}-{i}") for i in range(10)] for r in range(3)]
        pipeline = self._pipeline(batches)
        _, data = self._run(pipeline, {"result_id": "r1", "target_albums": 20})
        assert len(data["albums"]) == 20
        assert data["rounds"] == 2

    def test_extend_asks_for_full_batches(self):
        from backend.main import EXTEND_PER_ROUND
        pipeline = self._pipeline([[self._album("A", "1")], []])
        self._run(pipeline, {"result_id": "r1", "target_albums": 5})
        assert pipeline.select_discovery_albums.call_args.kwargs["max_results"] == EXTEND_PER_ROUND

    def test_capped_artists_are_named_to_the_model(self):
        """Otherwise it keeps offering another record by a band already listed
        (9 of 18 suggestions were capped away on 2026-11-03)."""
        batches = [
            [self._album("Opeth", "Blackwater Park"), self._album("Katatonia", "Brave Murder Day")],
            [self._album("Anathema", "Alternative 4")],
        ]
        pipeline = self._pipeline(batches)
        self._run(pipeline, {"result_id": "r1", "target_albums": 3, "max_albums_per_artist": 1})
        calls = pipeline.select_discovery_albums.call_args_list
        assert calls[0].kwargs["avoid_artists"] is None
        assert calls[1].kwargs["avoid_artists"] == ["Katatonia", "Opeth"]

    def test_no_cap_means_no_artists_are_avoided(self):
        batches = [[self._album("Opeth", "Blackwater Park")], [self._album("Opeth", "Damnation")], []]
        pipeline = self._pipeline(batches)
        self._run(pipeline, {"result_id": "r1", "target_albums": 3, "max_albums_per_artist": 0})
        assert all(c.kwargs["avoid_artists"] is None for c in pipeline.select_discovery_albums.call_args_list)

    def test_round_budget_scales_with_the_target(self):
        from backend.main import _extend_round_budget
        assert _extend_round_budget(5) == 4
        assert _extend_round_budget(20) == 5
        assert _extend_round_budget(30) == 6

    def test_saved_run_keeps_rejections(self):
        saved = {"id": "e1", "created_at": "t", "snapshot": {
            "albums": [], "rounds": 1, "cost": 0.01,
            "rejected": [{"artist": "B", "album": "fake", "reason": "not_found"}], "unverified": 2}}
        from backend.main import app
        with patch("backend.main.library_cache.get_latest_extend_for_playlist", return_value=saved):
            data = TestClient(app).get("/api/extend/latest/r1").json()
        assert data["rejected"] == saved["snapshot"]["rejected"]
        assert data["unverified"] == 2

    def test_saved_run_from_before_verification_still_loads(self):
        saved = {"id": "e1", "created_at": "t", "snapshot": {"albums": [], "rounds": 1, "cost": 0.0}}
        from backend.main import app
        with patch("backend.main.library_cache.get_latest_extend_for_playlist", return_value=saved):
            data = TestClient(app).get("/api/extend/latest/r1").json()
        assert (data["rejected"], data["unverified"]) == ([], 0)

    def _pipeline(self, batches):
        """A pipeline whose selector returns each batch in turn."""
        pipeline = MagicMock()
        pipeline.create_session.return_value = "s1"
        pipeline.select_discovery_albums.side_effect = batches
        pipeline.get_session_costs.return_value = (0, 0.0)
        # A real TasteProfile, not a MagicMock: the route stores it on a
        # RecommendSessionState, and pydantic rejects a mock there.
        from backend.models import TasteProfile
        pipeline.build_playlist_taste_profile.return_value = TasteProfile()
        return pipeline

    @staticmethod
    def _complete(body):
        """Pull the `complete` event payload out of an SSE body."""
        import json as _json
        for chunk in body.split("\n\n"):
            if "event: complete" in chunk:
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        return _json.loads(line[6:])
        return None

    def test_unknown_playlist_is_404(self, client):
        pipeline = self._pipeline([[]])
        with patch("backend.main._get_pipeline", return_value=pipeline), \
             patch("backend.main.library_cache.get_result", return_value=None):
            r = client.post("/api/extend/stream", json={"result_id": "nope"})
        assert r.status_code == 404

    def test_playlist_without_a_snapshot_is_400(self, client):
        pipeline = self._pipeline([[]])
        with patch("backend.main._get_pipeline", return_value=pipeline), \
             patch("backend.main.library_cache.get_result", return_value=self._result(tracks=[])):
            r = client.post("/api/extend/stream", json={"result_id": "r1"})
        assert r.status_code == 400

    def test_empty_library_is_400(self, client):
        """Knowing what is MISSING requires knowing what is there."""
        pipeline = self._pipeline([[]])
        with patch("backend.main._get_pipeline", return_value=pipeline), \
             patch("backend.main.library_cache.get_result", return_value=self._result()), \
             patch("backend.main.library_cache.has_cached_tracks", return_value=False):
            r = client.post("/api/extend/stream", json={"result_id": "r1"})
        assert r.status_code == 400

    def test_loops_until_the_target_is_reached(self, client):
        batches = [
            [self._album("A", "1"), self._album("B", "2"), self._album("C", "3")],
            [self._album("D", "4"), self._album("E", "5"), self._album("F", "6")],
        ]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={"result_id": "r1", "target_albums": 6})
            data = self._complete(r.text)
        assert r.status_code == 200
        assert data is not None, r.text
        assert len(data["albums"]) == 6
        assert data["rounds"] == 2

    def test_deduplicates_across_rounds(self, client):
        """A repeat in round 2 must not be counted twice."""
        batches = [
            [self._album("A", "1"), self._album("B", "2"), self._album("C", "3")],
            [self._album("A", "1"), self._album("D", "4"), self._album("E", "5")],
        ]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={"result_id": "r1", "target_albums": 5})
            data = self._complete(r.text)
        keys = [(a["artist"], a["album"]) for a in data["albums"]]
        assert len(keys) == len(set(keys)), f"duplicates got through: {keys}"
        assert len(keys) == 5

    def test_stops_early_when_a_round_adds_nothing_new(self, client):
        """Guards against paying for rounds that repeat themselves."""
        batches = [
            [self._album("A", "1"), self._album("B", "2"), self._album("C", "3")],
            [self._album("A", "1"), self._album("B", "2"), self._album("C", "3")],
            [self._album("X", "9")],  # should never be requested
        ]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={"result_id": "r1", "target_albums": 10})
            data = self._complete(r.text)
        assert len(data["albums"]) == 3
        assert data["rounds"] == 2, "should stop after the unproductive round, not keep going"
        assert pipeline.select_discovery_albums.call_count == 2


    def test_caps_albums_per_artist(self, client):
        """One album per band by default: a round of three by one artist yields one."""
        batches = [
            [self._album("Anathema", "A"), self._album("Anathema", "B"), self._album("Opeth", "C")],
            [self._album("Anathema", "D"), self._album("Katatonia", "E"), self._album("Opeth", "F")],
            [],  # model exhausted: fresh == 0, so the loop stops here
        ]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={
                "result_id": "r1", "target_albums": 10, "max_albums_per_artist": 1,
            })
            data = self._complete(r.text)
        artists = [a["artist"] for a in data["albums"]]
        assert sorted(artists) == ["Anathema", "Katatonia", "Opeth"], artists
        assert len(artists) == len(set(artists)), f"an artist appears twice: {artists}"

    def test_artist_cap_of_two_allows_two(self, client):
        batches = [
            [self._album("Anathema", "A"), self._album("Anathema", "B"), self._album("Anathema", "C")],
            [self._album("Opeth", "D")],
        ]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={
                "result_id": "r1", "target_albums": 3, "max_albums_per_artist": 2,
            })
            data = self._complete(r.text)
        artists = [a["artist"] for a in data["albums"]]
        assert artists.count("Anathema") == 2, artists

    def test_artist_cap_of_zero_means_no_limit(self, client):
        batches = [[self._album("Anathema", "A"), self._album("Anathema", "B"), self._album("Anathema", "C")]]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={
                "result_id": "r1", "target_albums": 3, "max_albums_per_artist": 0,
            })
            data = self._complete(r.text)
        assert [a["artist"] for a in data["albums"]] == ["Anathema"] * 3

    def test_artist_cap_is_case_and_whitespace_insensitive(self, client):
        batches = [
            [self._album("Anathema", "A"), self._album(" anathema ", "B"), self._album("Opeth", "C")],
            [],  # model exhausted
        ]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={
                "result_id": "r1", "target_albums": 5, "max_albums_per_artist": 1,
            })
            data = self._complete(r.text)
        artists = [a["artist"].strip().lower() for a in data["albums"]]
        assert artists.count("anathema") == 1, artists

    def test_capped_artists_are_fed_back_so_the_model_stops_suggesting_them(self, client):
        """A rejected album still goes into previously_recommended.

        Without this the loop asks the model for more, gets the same artist
        again, rejects it again, and bills for every round of that.
        """
        batches = [
            [self._album("Anathema", "A"), self._album("Anathema", "B"), self._album("Opeth", "C")],
            [self._album("Katatonia", "D")],
        ]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline)
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            client.post("/api/extend/stream", json={
                "result_id": "r1", "target_albums": 4, "max_albums_per_artist": 1,
            }).read()
        second_call = pipeline.select_discovery_albums.call_args_list[1]
        passed = second_call.kwargs.get("previously_recommended") or []
        assert "Anathema|||B" in passed, (
            f"the rejected album must be excluded next round; got {passed}"
        )


    def test_a_completed_run_is_saved_and_its_id_returned(self, client):
        batches = [[self._album("A", "1"), self._album("B", "2")], []]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline) + [
            patch("backend.main.library_cache.save_result", return_value="saved1"),
        ]
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={
                "result_id": "r1", "target_albums": 2, "max_albums_per_artist": 1,
            })
            data = self._complete(r.text)
        assert data["result_id"] == "saved1"

    def test_a_save_failure_still_returns_the_albums(self, client):
        """A result the user can see is worth more than a stored one."""
        batches = [[self._album("A", "1")], []]
        pipeline = self._pipeline(batches)
        ps = self._patches(pipeline) + [
            patch("backend.main.library_cache.save_result", side_effect=RuntimeError("disk full")),
        ]
        with ExitStack() as stack:
            for p_ in ps:
                stack.enter_context(p_)
            r = client.post("/api/extend/stream", json={"result_id": "r1", "target_albums": 1})
            data = self._complete(r.text)
        assert len(data["albums"]) == 1
        assert data["result_id"] is None

    def test_latest_endpoint_reports_not_found_cleanly(self, client):
        with patch("backend.main.library_cache.get_latest_extend_for_playlist", return_value=None):
            r = client.get("/api/extend/latest/nope")
        assert r.status_code == 200 and r.json() == {"found": False}

    def test_latest_endpoint_returns_a_previous_run(self, client):
        saved = {
            "id": "e1", "created_at": "2026-11-02 10:00:00",
            "snapshot": {"albums": [{"artist": "A", "album": "1"}], "rounds": 2,
                         "cost": 0.4, "max_albums_per_artist": 1},
        }
        with patch("backend.main.library_cache.get_latest_extend_for_playlist", return_value=saved):
            r = client.get("/api/extend/latest/playlist-1")
        body = r.json()
        assert body["found"] is True and body["result_id"] == "e1"
        assert len(body["albums"]) == 1 and body["cost"] == 0.4

    def test_rejects_an_out_of_range_target(self, client):
        r = client.post("/api/extend/stream", json={"result_id": "r1", "target_albums": 99})
        assert r.status_code == 422
