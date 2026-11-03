"""Offline tests for the strict MusicBrainz existence check behind Extend Library.

Candidate dicts use the shape the release-group search API returns. The rules
were first checked against the LIVE API on 2026-11-03 (15 known cases, real
and invented); these pin them so a refactor cannot quietly loosen them.
"""

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.music_research import (
    MusicResearchClient,
    album_keys,
    match_release_group,
    normalize_name,
)


def rg(title, artist, ptype="Album", date="", rid="id1", credits=None, secondary=None):
    return {
        "id": rid, "title": title, "primary-type": ptype, "first-release-date": date,
        "secondary-types": secondary,
        "artist-credit": credits if credits is not None else [{"name": artist, "artist": {"name": artist}}],
    }


class TestNormalize:
    @pytest.mark.parametrize("a,b", [
        ("The Beatles", "Beatles"),
        ("Sigur Rós", "Sigur Ros"),
        ("Ágætis byrjun", "Agaetis byrjun"),           # æ does not decompose under NFKD
        ("Simon & Garfunkel", "Simon and Garfunkel"),
        ("Godspeed You! Black Emperor", "Godspeed You Black Emperor"),
        ("Μάνος Χατζιδάκις", "ΜΑΝΟΣ ΧΑΤΖΙΔΑΚΙΣ"),        # Greek survives, case folds
    ])
    def test_equivalent(self, a, b):
        assert normalize_name(a) == normalize_name(b)

    def test_greek_is_not_erased(self):
        assert normalize_name("Ο Μεγάλος Ερωτικός") != ""

    def test_owned_keys_ignore_edition_suffixes(self):
        assert album_keys("Radiohead", "OK Computer (Remastered)") & album_keys("Radiohead", "OK Computer")


class TestMatch:
    def test_exact(self):
        assert match_release_group([rg("Spirit of Eden", "Talk Talk")], "Talk Talk", "Spirit of Eden")

    def test_phrase_search_trap_is_rejected(self):
        """MusicBrainz's phrase search returns "Kind of Blue" for "Blue"."""
        assert match_release_group([rg("Kind of Blue", "Miles Davis")], "Miles Davis", "Blue") is None

    def test_real_title_wrong_artist_is_rejected(self):
        assert match_release_group([rg("Kind of Blue", "Miles Davis")], "Portishead", "Kind of Blue") is None

    def test_single_does_not_verify_an_album(self):
        assert match_release_group([rg("Paranoid Android", "Radiohead", ptype="Single")],
                                   "Radiohead", "Paranoid Android") is None

    def test_ep_is_accepted(self):
        assert match_release_group([rg("Airbag / How Am I Driving?", "Radiohead", ptype="EP")],
                                   "Radiohead", "Airbag / How Am I Driving?")

    def test_fuzzy_title_with_matching_artist(self):
        c = [rg("Lift Yr. Skinny Fists Like Antennas to Heaven!", "Godspeed You! Black Emperor")]
        assert match_release_group(c, "Godspeed You Black Emperor",
                                   "Lift Your Skinny Fists Like Antennas to Heaven")

    @pytest.mark.parametrize("claimed,real", [
        ("Kid B", "Kid A"),
        ("The Colour of Autumn", "The Colour of Spring"),
    ])
    def test_invented_sibling_titles_are_rejected(self, claimed, real):
        assert match_release_group([rg(real, "X")], "X", claimed) is None

    def test_exact_beats_fuzzy(self):
        c = [rg("OK Computers", "Radiohead", rid="fuzzy"), rg("OK Computer", "Radiohead", rid="exact")]
        assert match_release_group(c, "Radiohead", "OK Computer")["id"] == "exact"

    def test_deluxe_qualifier_on_either_side(self):
        assert match_release_group([rg("In Rainbows", "Radiohead")], "Radiohead", "In Rainbows (Deluxe Edition)")
        assert match_release_group([rg("In Rainbows (Disk 2)", "Radiohead")], "Radiohead", "In Rainbows")

    def test_joint_credit_as_printed(self):
        credits = [{"name": "Lou Reed", "joinphrase": " & ", "artist": {"name": "Lou Reed"}},
                   {"name": "John Cale", "artist": {"name": "John Cale"}}]
        c = [rg("Songs for Drella", "", credits=credits)]
        assert match_release_group(c, "Lou Reed & John Cale", "Songs for Drella")
        assert match_release_group(c, "John Cale", "Songs for Drella")  # either credited artist

    def test_real_ep_beats_remix_bootleg_album(self):
        """The live failure of 2026-11-03, using the exact candidates
        MusicBrainz returned for it."""
        c = [
            rg("Jar of Flies", "Alice in Chains", ptype="EP", date="1994-01-18", rid="real"),
            rg("Jar of Flies/ Sap", "Alice in Chains", date="2016-10-14", rid="c1", secondary=["Compilation"]),
            rg("Jar of Flies / Sap", "Alice in Chains", ptype="EP", date="1994-01-18", rid="c2", secondary=["Compilation"]),
            rg("Jar of Flies / Facelift / Dirt", "Alice in Chains", date="2001-09-17", rid="c3", secondary=["Compilation"]),
            rg("Jar of Flies (Chopped Not Slopped by DJ Lil Steve)", "Alice in Chains", date="2012",
               rid="bootleg", secondary=["Remix"]),
        ]
        assert match_release_group(c, "Alice in Chains", "Jar of Flies", year=1994)["id"] == "real"
        assert match_release_group(c, "Alice in Chains", "Jar of Flies")["id"] == "real"  # without the year too

    def test_derivative_still_verifies_when_it_is_the_only_match(self):
        c = [rg("Alive!", "Kiss", date="1975", secondary=["Live"])]
        assert match_release_group(c, "Kiss", "Alive!")

    def test_exact_title_beats_a_stripped_match(self):
        c = [rg("In Rainbows (Disk 2)", "Radiohead", rid="disk2", date="2007"),
             rg("In Rainbows", "Radiohead", ptype="EP", rid="real", date="2007")]
        assert match_release_group(c, "Radiohead", "In Rainbows")["id"] == "real"

    def test_earliest_release_breaks_a_full_tie(self):
        c = [rg("Same", "A", date="2005", rid="late"), rg("Same", "A", date="1990", rid="early")]
        assert match_release_group(c, "A", "Same")["id"] == "early"

    @pytest.mark.parametrize("claimed,real", [
        ("Led Zeppelin III", "Led Zeppelin II"),
        ("Opiate²", "Opiate"),
        ("Greatest Hits Vol. 3", "Greatest Hits Vol. 2"),
        ("1999", "1998"),
    ])
    def test_fuzzy_never_bridges_a_different_number(self, claimed, real):
        assert match_release_group([rg(real, "X")], "X", claimed) is None

    def test_fuzzy_still_works_when_numbers_agree(self):
        assert match_release_group([rg("Physical Graffiti 2", "X")], "X", "Physical Grafitti 2")

    def test_year_breaks_ties(self):
        c = [rg("Same", "A", date="1990", rid="a"), rg("Same", "A", date="2001", rid="b")]
        assert match_release_group(c, "A", "Same", year=2001)["id"] == "b"


def _resp(status, payload=None, headers=None):
    req = httpx.Request("GET", "https://musicbrainz.org/ws/2/release-group")
    return httpx.Response(status, json=payload or {}, headers=headers or {}, request=req)


class TestVerifyAlbum:
    def _client(self, responses):
        c = MusicResearchClient()
        http = MagicMock()

        async def get(*a, **kw):
            return responses.pop(0)

        http.get = get
        http.is_closed = False
        c._http = http
        return c

    def _run(self, coro):
        with patch("backend.music_research.asyncio.sleep", new=_nosleep):
            return asyncio.run(coro)

    def test_verified_returns_canonical_names(self):
        c = self._client([_resp(200, {"release-groups": [
            rg("OK Computer", "Radiohead", date="1997-05-21", rid="mb1")]})])
        v = self._run(c.verify_album("radiohead", "OK Computers"))
        assert (v.status, v.album, v.artist, v.year, v.musicbrainz_id) == (
            "verified", "OK Computer", "Radiohead", 1997, "mb1")

    def test_title_only_search_is_the_second_chance(self):
        c = self._client([
            _resp(200, {"release-groups": []}),
            _resp(200, {"release-groups": [rg("Bookends", "Simon & Garfunkel")]}),
        ])
        assert self._run(c.verify_album("Simon and Garfunkel", "Bookends")).status == "verified"

    def test_a_single_by_that_title_is_not_album_not_not_found(self):
        """Opiate² exists -- as a single. Saying "not found" would call a real
        release invented."""
        single = rg("Opiate²", "Tool", ptype="Single", date="2022-03-01")
        ep = rg("Opiate", "Tool", ptype="EP", date="1992-03-10")
        c = self._client([_resp(200, {"release-groups": [single, ep]}),
                          _resp(200, {"release-groups": [single, ep]})])
        assert self._run(c.verify_album("Tool", "Opiate²", 2022)).status == "not_album"

    def test_query_uses_the_titles_own_words(self):
        """Pre-folding "Opiate²" to "opiate2" made MusicBrainz miss it."""
        sent = []
        c = MusicResearchClient()
        http = MagicMock()

        async def get(url, params=None, **kw):
            sent.append(params["query"])
            return _resp(200, {"release-groups": []})

        http.get = get
        http.is_closed = False
        c._http = http
        self._run(c.verify_album("Tool", "Opiate² (Deluxe) [Remaster]"))
        assert sent[0] == 'artist:"Tool" AND releasegroup:(opiate²)'
        assert sent[1] == 'releasegroup:"Opiate²"'

    def test_lucene_syntax_in_titles_is_neutralised(self):
        sent = []
        c = MusicResearchClient()
        http = MagicMock()

        async def get(url, params=None, **kw):
            sent.append(params["query"])
            return _resp(200, {"release-groups": []})

        http.get = get
        http.is_closed = False
        c._http = http
        self._run(c.verify_album('AC/DC', 'Who Made Who? AND: (Live)'))
        assert sent[0] == 'artist:"AC/DC" AND releasegroup:(who made who and)'

    def test_not_found_after_both_searches(self):
        c = self._client([_resp(200, {"release-groups": []}), _resp(200, {"release-groups": []})])
        assert self._run(c.verify_album("Talk Talk", "Silent Harbour")).status == "not_found"

    def test_503_is_retried(self):
        c = self._client([
            _resp(503, headers={"x-ratelimit-who": "search-shed"}),
            _resp(200, {"release-groups": [rg("Spirit of Eden", "Talk Talk")]}),
        ])
        assert self._run(c.verify_album("Talk Talk", "Spirit of Eden")).status == "verified"

    def test_persistent_503_is_unavailable_not_not_found(self):
        """An outage must never be reported as an invented album."""
        c = self._client([_resp(503)] * 4)
        assert self._run(c.verify_album("Talk Talk", "Spirit of Eden")).status == "unavailable"

    def test_network_error_is_unavailable(self):
        c = MusicResearchClient()
        http = MagicMock()

        async def boom(*a, **kw):
            raise httpx.ConnectError("down")

        http.get = boom
        http.is_closed = False
        c._http = http
        assert self._run(c.verify_album("A", "B")).status == "unavailable"


async def _nosleep(*_a, **_kw):
    return None
