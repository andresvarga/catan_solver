"""League registry for self-play (roadmap phase 5, design doc §7).

A `League` is a persisted population of opponents a trainee can be matched
against, each tagged with a role:

- "main"           the current best trainee snapshot; the promotion target.
- "historical"     frozen past "main" snapshots, kept forever so old exploits
                   can't quietly resurface (the classic anti-forgetting fix).
- "main_exploiter" trained briefly against only the *current* frozen main to
                   find and highlight its specific weaknesses.
- "league_exploiter" (reserved for a version trained against the broad
                   historical pool rather than just current main -- not yet
                   populated by league_train.py, but sampling/rating code
                   already treats it like any other opponent role).
- "heuristic" / "random"  the phase-3 fixed agents, included as permanent,
                   free strategic diversity and as the "evaluator" tier for
                   absolute-progress reporting untouched by league dynamics.

Ratings use TrueSkill (`pip install trueskill`, as the design doc recommends)
fed by each game's full finishing order -- exactly what a 4-player
free-for-all naturally produces, unlike pairwise Elo.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field

import trueskill

DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25.0 / 3
EVALUATOR_ROLES = {"heuristic", "random"}


@dataclass
class LeagueMember:
    name: str
    role: str
    checkpoint_path: str | None = None
    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA
    games_played: int = 0
    created_iteration: int = 0


def binomial_test_pvalue(successes: int, n: int, p0: float) -> float:
    """One-sided exact binomial p-value for P(X >= successes | Binomial(n, p0)).
    Tests whether an observed win rate is significantly *above* p0, without
    needing scipy."""
    if n == 0:
        return 1.0
    return sum(
        math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
        for k in range(successes, n + 1)
    )


class League:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.members: dict[str, LeagueMember] = {}
        self.main_name: str | None = None
        # Last completed training iteration -- persisted so a resumed run
        # continues numbering (and snapshot filenames) where it left off.
        self.last_iteration: int = 0
        # draw_probability=0: episode_rating_teams always reports a strict
        # winner/loser ordering, so TrueSkill's default 10% draw prior would
        # just miscalibrate every update.
        self._env = trueskill.TrueSkill(draw_probability=0.0)

    # -- membership -----------------------------------------------------------
    def add_member(self, name: str, role: str, checkpoint_path: str | None = None,
                   iteration: int = 0, mu: float | None = None, sigma: float | None = None) -> LeagueMember:
        member = LeagueMember(
            name=name, role=role, checkpoint_path=checkpoint_path,
            mu=DEFAULT_MU if mu is None else mu, sigma=DEFAULT_SIGMA if sigma is None else sigma,
            created_iteration=iteration,
        )
        self.members[name] = member
        if role == "main":
            self.main_name = name
        return member

    def rating(self, name: str) -> trueskill.Rating:
        m = self.members[name]
        return self._env.create_rating(mu=m.mu, sigma=m.sigma)

    def main(self) -> LeagueMember | None:
        return self.members.get(self.main_name) if self.main_name else None

    # -- matchmaking (PFSP-ish) -------------------------------------------------
    def sample_opponent(self, rng: random.Random, exclude_roles: tuple[str, ...] = ("random",)
                         ) -> LeagueMember:
        """Weight opponents toward whoever's TrueSkill quality against the
        current main is highest -- i.e. the closest, most competitive
        matchups, rather than uniformly across the whole history. Falls back
        to uniform sampling if there's no main yet to compare against."""
        candidates = [m for m in self.members.values() if m.role not in exclude_roles]
        if not candidates:
            raise ValueError("no league members available to sample as an opponent")
        if self.main_name is None or self.main_name not in self.members:
            return rng.choice(candidates)
        main_rating = self.rating(self.main_name)
        weights = []
        for m in candidates:
            if m.name == self.main_name:
                weights.append(1e-6)
                continue
            quality = trueskill.quality_1vs1(main_rating, self.rating(m.name), env=self._env)
            weights.append(quality + 0.01)
        return rng.choices(candidates, weights=weights, k=1)[0]

    # -- ratings -----------------------------------------------------------------
    def update_ratings(self, ranking: list[str]) -> None:
        """`ranking`: member names in finishing order, winner first. Members
        not tracked in the league (e.g. an ad hoc name) are silently skipped
        from the update but don't break the rest of the ranking's math --
        TrueSkill's `rate` needs a rating per team regardless."""
        known = [name for name in ranking if name in self.members]
        if len(known) < 2:
            return
        teams = [[self.rating(name)] for name in known]
        new_ratings = self._env.rate(teams, ranks=list(range(len(known))))
        for name, (new_r,) in zip(known, new_ratings):
            m = self.members[name]
            m.mu, m.sigma = new_r.mu, new_r.sigma
            m.games_played += 1

    # -- promotion -----------------------------------------------------------------
    def promotion_test(self, wins: int, games: int, win_rate_threshold: float = 0.30,
                        alpha: float = 0.05, parity: float = 0.25) -> bool:
        """Promote only if the candidate's win rate clears `win_rate_threshold`
        AND is significantly above `parity` -- the win rate of an
        equal-strength player in the evaluation format. Promotion games are
        1 candidate seat vs 3 copies of the main, so parity is 1/4; testing
        against 0.5 (as this used to) demanded ~2.4x parity and made
        promotion practically unreachable (audit F-17)."""
        if games == 0 or wins / games < win_rate_threshold:
            return False
        p_value = binomial_test_pvalue(wins, games, p0=parity)
        return p_value < alpha

    def promote(self, name: str, iteration: int) -> None:
        """The named member becomes the new main. The previous main (if any)
        is automatically demoted to role="historical" and kept in the pool
        forever -- the classic anti-forgetting mechanism -- rather than
        relying on the caller to have done that first."""
        if name not in self.members:
            raise KeyError(name)
        if self.main_name is not None and self.main_name in self.members and self.main_name != name:
            self.members[self.main_name].role = "historical"
        self.members[name].role = "main"
        self.main_name = name

    # -- persistence -----------------------------------------------------------------
    def save(self) -> None:
        path = os.path.join(self.storage_dir, "league.json")
        data = {"main_name": self.main_name, "last_iteration": self.last_iteration,
                "members": {k: asdict(v) for k, v in self.members.items()}}
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, storage_dir: str) -> "League":
        league = cls(storage_dir)
        path = os.path.join(storage_dir, "league.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            league.main_name = data["main_name"]
            league.last_iteration = data.get("last_iteration", 0)  # absent in pre-existing files
            league.members = {k: LeagueMember(**v) for k, v in data["members"].items()}
        return league
