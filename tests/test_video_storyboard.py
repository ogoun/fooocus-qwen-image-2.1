"""Раскадровка: разбор грязного ответа модели, подгонка длительностей, таблица интерфейса."""

import json
from pathlib import Path

import pytest

from fooocus_qwen.video import geometry, prompt_h3, storyboard
from fooocus_qwen.video.storyboard import Shot

PROMPT_DIR = Path(__file__).resolve().parents[1] / "resources" / "prompts"

SHOT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a static close-up frames a wilted plant.\n\n"
    "overall_soundscape: Quiet room tone.\n\nnon_diegetic_music: N/A"
)


def _on_grid(seconds: float) -> bool:
    frames = round(seconds * geometry.FPS)
    return frames % geometry.FRAME_UNIT == geometry.FRAME_OFFSET and abs(frames / geometry.FPS - seconds) < 1e-9


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def complete(self, system, user, images=None, temperature=0.3, max_tokens=2048):
        self.calls.append({"system": system, "user": user})
        return self.answer


def test_parse_dirty_answer_with_think_fence_prose_and_trailing_comma():
    text = (
        "<think>[not, json]</think>Here is the plan:\n```json\n"
        '[{"title": "Problem", "seconds": "6,5 s", "transition": "Cross-Dissolve", "prompt": "a"},'
        ' {"title": "Action", "duration": 7, "transition": "Wipe Right", "prompt": "b"},'
        ' {"title": "Result", "seconds": 99, "transition": "spin", "prompt": "c"},]\n```\nHope it helps!'
    )
    shots = storyboard.parse(text)
    assert [shot.title for shot in shots] == ["Problem", "Action", "Result"]
    # У первого шота переход не нужен, незнакомый переход — склейка.
    assert [shot.transition for shot in shots] == ["cut", "wiperight", "cut"]
    assert all(_on_grid(shot.seconds) for shot in shots)
    assert shots[0].seconds == pytest.approx(158 / 24)  # 6.5 → ближайшая ступень 6.58
    assert shots[2].seconds == pytest.approx(362 / 24)  # зажато в одну генерацию


def test_parse_accepts_object_with_shots_and_field_dicts():
    payload = {"shots": [{"prompt": {"integrated_multimodal_description": "[Shot 1] x", "non_diegetic_music": "N/A"}},
                         {"text": "y", "transition": "dissolve"}, {"title": "no prompt"}]}
    shots = storyboard.parse("Answer: " + json.dumps(payload))
    assert len(shots) == 2
    assert shots[0].prompt == "integrated_multimodal_description: [Shot 1] x\n\nnon_diegetic_music: N/A"
    assert shots[0].seconds == pytest.approx(storyboard.fit_seconds(storyboard.DEFAULT_SHOT_SECONDS))
    assert shots[1].transition == "dissolve"


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", '{"title": "x"}', "<think>[{}]"])
def test_parse_failure_raises_value_error(text):
    with pytest.raises(ValueError, match="no JSON array of shots"):
        storyboard.parse(text)


def test_fit_total_matches_total_on_grid_and_rescales_timecodes():
    shots = [
        Shot("[Shot 1] a [Shot 2] At 00:07.000, b", 8.0),
        Shot("c", 8.0),
        Shot("d", 8.0),
    ]
    fitted = storyboard.fit_total(shots, 20.0)
    assert abs(sum(shot.seconds for shot in fitted) - 20.0) <= 17 / 24 / 2
    assert all(_on_grid(shot.seconds) for shot in fitted)
    # Склейка в 7 с не помещается в укороченный шот — тайм-код сжат.
    code = fitted[0].prompt.split("At ")[1].split(",")[0]
    composed = prompt_h3.compose(fitted[0].prompt, prompt_h3.MODE_T2V, fitted[0].seconds)
    assert prompt_h3.check(composed, prompt_h3.MODE_T2V, fitted[0].seconds) == [], code


def test_fit_total_keeps_shots_in_trained_range():
    # Три шота по 5.17 — уже 15.5 с: ниже обученных 124 кадров шот не ужимается.
    fitted = storyboard.fit_total([Shot("a", 8.0), Shot("b", 8.0), Shot("c", 8.0)], 15.0)
    assert [round(shot.seconds * geometry.FPS) for shot in fitted] == [124, 124, 124]
    # Шот, о котором модель просила короче, так и остаётся коротким.
    fitted = storyboard.fit_total([Shot("a", 2.0), Shot("b", 8.0)], 10.0)
    assert fitted[0].seconds < 5
    assert all(geometry.MIN_SECONDS <= shot.seconds for shot in fitted)


def test_plan_composes_t2v_prompts_and_fits_total():
    answer = json.dumps([
        {"title": "Problem", "seconds": 5, "transition": "cut", "prompt": SHOT_PROMPT},
        {"title": "Action", "seconds": 6, "transition": "fade", "prompt": SHOT_PROMPT},
        {"title": "Result", "seconds": 6, "transition": "dissolve", "prompt": SHOT_PROMPT},
    ])
    client = FakeClient("```\n" + answer + "\n```")
    shots = storyboard.plan(client, "растение оживает", 18, structure="problem_action_result", aspect="9:16",
                            prompt_dir=PROMPT_DIR)
    assert len(shots) == 3
    assert abs(sum(shot.seconds for shot in shots) - 18) <= 17 / 24 / 2
    for shot in shots:
        assert prompt_h3.check(shot.prompt, prompt_h3.MODE_T2V, shot.seconds) == []
    user = client.calls[0]["user"]
    assert "problem → action → result" in user
    assert "vertical" in user
    assert "растение оживает" in user
    assert "{{" not in client.calls[0]["system"]


def test_plan_rejects_unknown_structure():
    with pytest.raises(ValueError):
        storyboard.plan(FakeClient("[]"), "x", 10, structure="nope", prompt_dir=PROMPT_DIR)


def test_structures_have_bilingual_labels():
    assert set(storyboard.STRUCTURES) == {
        "free", "problem_action_result", "promise_action_result_next", "explainer", "story",
    }
    for structure in storyboard.STRUCTURES.values():
        ru, en = structure.labels
        assert ru and en and ru != en
    assert len(storyboard.STRUCTURES["explainer"].beats) == 5
    assert storyboard.shot_count(40, "explainer")[0] >= 5


def test_rows_round_trip():
    shots = [Shot("p1", 5.0, "cut", "A"), Shot("p2", 8.0, "fade", "B")]
    shots = [Shot(shot.prompt, storyboard.fit_seconds(shot.seconds), shot.transition, shot.title) for shot in shots]
    rows = storyboard.to_rows(shots)
    assert rows == [[1, 5.17, "cut", "A", "p1"], [2, 8.0, "fade", "B", "p2"]]
    assert storyboard.from_rows(rows) == shots


def test_from_rows_tolerates_blank_rows_and_string_numbers():
    rows = [
        ["1", "6,5", "Dissolve", "Start", "p1"],
        ["", "", "", "", ""],
        [None, None, None, None, None],
        ["3", "abc", "zoom", float("nan"), "p2"],
        ["4", 8],
    ]
    shots = storyboard.from_rows(rows)
    assert [shot.prompt for shot in shots] == ["p1", "p2"]
    assert shots[0].seconds == pytest.approx(158 / 24)
    assert shots[0].transition == "dissolve"
    assert shots[1].seconds == pytest.approx(storyboard.fit_seconds(storyboard.DEFAULT_SHOT_SECONDS))
    assert shots[1].transition == "cut" and shots[1].title == ""
    assert storyboard.from_rows(None) == []


def test_from_rows_accepts_dataframe():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame([[1, 8.0, "cut", "A", "p"]], columns=list(storyboard.COLUMNS))
    assert storyboard.from_rows(frame) == [Shot("p", 8.0, "cut", "A")]
