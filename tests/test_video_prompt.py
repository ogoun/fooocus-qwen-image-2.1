"""Промт H3: метки, строки выравнивания, сборка, проверка и буст с фальшивым клиентом."""

from pathlib import Path

import pytest
from PIL import Image

from fooocus_qwen.llm import LlmImagesRejected
from fooocus_qwen.video import geometry, prompt_h3
from fooocus_qwen.video.prompt_h3 import MODE_FLF2V, MODE_I2V, MODE_REF, MODE_T2V, NEGATIVE_LINE

PROMPT_DIR = Path(__file__).resolve().parents[1] / "resources" / "prompts"

T2V_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a baker. "
    "The baker with a raspy voice (S1) says: <d>[English] Morning.</d> "
    "[Shot 2] At 00:03.500, the camera cuts to a close-up of bread.\n\n"
    "overall_soundscape: Trays clink softly.\n\n"
    "non_diegetic_music: N/A\n\n"
    f"{NEGATIVE_LINE}"
)


class FakeClient:
    """Отвечает заготовками по очереди и записывает запросы."""

    def __init__(self, *answers, reject_images=False):
        self.answers = list(answers)
        self.reject_images = reject_images
        self.calls = []

    def complete(self, system, user, images=None, temperature=0.3, max_tokens=2048):
        self.calls.append({"system": system, "user": user, "images": images, "temperature": temperature,
                           "max_tokens": max_tokens})
        if images and self.reject_images:
            raise LlmImagesRejected("HTTP 500")
        return self.answers.pop(0)


# --- Метки ------------------------------------------------------------------------------------


def test_engine_order_puts_images_then_videos_then_audio():
    kinds = ["audio", "video", "image", "video", "image"]
    assert prompt_h3.engine_order(kinds) == [2, 4, 1, 3, 0]


def test_labels_are_numbered_per_kind_in_engine_order():
    labels = prompt_h3.reference_labels(["image", "image", "video", "audio"])
    assert labels == ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"]


def test_video_sound_track_gets_audio_label_before_its_video():
    labels = prompt_h3.reference_labels(["audio", "video", "image"], video_has_audio=[True])
    # Дорожка видео подаётся раньше самостоятельного аудио — её номер младше.
    assert labels == ["<Picture 1>", "<Audio 1>", "<Video 1>", "<Audio 2>"]


def test_labels_by_input_keep_user_order():
    per_input = prompt_h3.labels_by_input(["audio", "video", "video", "image"], video_has_audio=[False, True])
    assert per_input == [("<Audio 2>",), ("<Video 1>",), ("<Audio 1>", "<Video 2>"), ("<Picture 1>",)]


def test_bad_reference_input_is_rejected():
    with pytest.raises(ValueError):
        prompt_h3.reference_labels(["image", "gif"])
    with pytest.raises(ValueError):
        prompt_h3.reference_labels(["video"], video_has_audio=[True, False])


# --- Строки выравнивания ----------------------------------------------------------------------


def test_fl2va_line_uses_effective_duration_without_brackets():
    line = prompt_h3.alignment_line(MODE_FLF2V, geometry.duration(5), has_first=True, has_last=True)
    assert line == (
        "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
        "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.17-second mark of the "
        "target video."
    )


def test_l2va_line_has_brackets_and_final_shot():
    line = prompt_h3.alignment_line(MODE_FLF2V, 8, has_last=True, last_shot=3)
    assert line == (
        "How the reference pictures align with the target video — <Picture 1> (from [Shot 3]) aligns with the "
        "8.00-second mark of the target video."
    )


def test_i2va_line_and_modes_without_line():
    expected = (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    )
    assert prompt_h3.alignment_line(MODE_I2V, 5) == expected
    assert prompt_h3.alignment_line(MODE_FLF2V, 5, has_first=True) == expected
    assert prompt_h3.alignment_line(MODE_T2V, 5) == ""
    assert prompt_h3.alignment_line(MODE_REF, 5) == ""


def test_rounded_effective_duration_stays_on_its_frame():
    # 5.17 · 24 = 124.08: округление вверх увело бы на 141 кадр (5.88 с).
    line = prompt_h3.alignment_line(MODE_FLF2V, 5.17, has_first=True, has_last=True)
    assert "5.17-second mark" in line


# --- Сборка -----------------------------------------------------------------------------------


def test_is_structured():
    assert prompt_h3.is_structured(T2V_PROMPT)
    assert prompt_h3.is_structured("detailed_description:\n[Shot 1] x")
    assert not prompt_h3.is_structured("кот идёт по крыше")


@pytest.mark.parametrize(
    ("mode", "has_first", "has_last"),
    [(MODE_T2V, False, False), (MODE_I2V, True, False), (MODE_FLF2V, True, True), (MODE_FLF2V, False, True)],
)
def test_free_text_is_wrapped_into_valid_base_prompt(mode, has_first, has_last):
    prompt = prompt_h3.compose("A cat walks along the roof at dusk.", mode, 8, has_first, has_last)
    assert prompt.endswith(NEGATIVE_LINE)
    assert prompt.count(NEGATIVE_LINE) == 1
    assert "integrated_multimodal_description: [Shot 1]" in prompt
    assert prompt_h3.check(prompt, mode, 8, has_first=has_first, has_last=has_last) == []


def test_free_russian_text_is_wrapped_but_not_translated():
    prompt = prompt_h3.compose("Кот идёт по крыше", MODE_T2V, 5)
    assert "[Shot 1] Кот идёт по крыше" in prompt
    problems = prompt_h3.check(prompt, MODE_T2V, 5)
    assert problems == ["Text outside <d> and quotes is not in English"]


def test_free_text_for_ref_gets_six_sections_with_labels():
    labels = ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"]
    prompt = prompt_h3.compose("A woman dances in a garden.", MODE_REF, 8, labels=labels)
    names = [line.rstrip(":") for line in prompt.splitlines() if line.endswith(":")]
    assert names == list(prompt_h3.REF_FIELDS)
    assert "<Subject 2> is the visible content of <Picture 2>" in prompt
    assert "[reference generation + audio reference]" in prompt
    assert prompt_h3.check(prompt, MODE_REF, 8, labels) == []


def test_free_text_starting_with_shot_keeps_single_marker():
    prompt = prompt_h3.compose("[Shot 1] A door opens.", MODE_I2V, 5, has_first=True)
    assert prompt.count("[Shot 1]") == 2  # строка выравнивания и тело
    assert "[Shot 1] The shot begins exactly from <Picture 1>" in prompt


def test_structured_text_gets_correct_alignment_and_single_negative_line():
    wrong = (
        "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
        "5.00-second mark of the target video.\n\n" + T2V_PROMPT + "\n" + NEGATIVE_LINE
    )
    prompt = prompt_h3.compose(wrong, MODE_FLF2V, 5, has_first=False, has_last=True)
    first = prompt.splitlines()[0]
    assert first.endswith("<Picture 1> (from [Shot 2]) aligns with the 5.17-second mark of the target video.")
    assert prompt.count(NEGATIVE_LINE) == 1 and prompt.endswith(NEGATIVE_LINE)
    assert prompt_h3.check(prompt, MODE_FLF2V, 5, has_last=True) == []


def test_structured_text_loses_alignment_in_t2v_and_gains_it_in_i2v():
    with_line = prompt_h3.compose(T2V_PROMPT, MODE_I2V, 5)
    assert with_line.startswith(prompt_h3.I2VA_LINE + "\n\nintegrated_multimodal_description:")
    back = prompt_h3.compose(with_line, MODE_T2V, 5)
    assert back.startswith("integrated_multimodal_description:")


def test_structured_ref_text_is_kept():
    text = (
        "subject_definitions:\n<Subject 1> is the man in <Picture 1>.\n\nsummary:\n[reference generation] x <Subject 1>."
        "\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - kept.\n\n"
        "detailed_description:\nStyle.\n[Shot 1] <Subject 1> waves.\n\noverall_soundscape:\nWind.\n\n"
        "non_diegetic_music:\nN/A"
    )
    prompt = prompt_h3.compose(text, MODE_REF, 5, labels=["<Picture 1>"])
    assert prompt == text + "\n\n" + NEGATIVE_LINE
    assert prompt_h3.check(prompt, MODE_REF, 5, ["<Picture 1>"]) == []


# --- Проверка ---------------------------------------------------------------------------------


def test_valid_prompt_has_no_problems():
    assert prompt_h3.check(T2V_PROMPT, MODE_T2V, 5) == []


def test_check_catches_missing_and_misordered_fields():
    text = "overall_soundscape: x\n\nintegrated_multimodal_description: [Shot 1] y\n\n" + NEGATIVE_LINE
    problems = prompt_h3.check(text, MODE_T2V, 5)
    assert "Missing field 'non_diegetic_music'" in problems
    assert any(problem.startswith("Fields are out of order") for problem in problems)


def test_check_catches_timeline_errors():
    text = T2V_PROMPT.replace("[Shot 2] At 00:03.500,", "[Shot 2] At 00:06.000, x [Shot 3] At 00:02.0,")
    problems = prompt_h3.check(text, MODE_T2V, 5)
    assert "Timecode 00:06.000 is beyond the video duration of 5.17 s" in problems
    assert "[Shot 3] has no cut timecode in MM:SS.mmm format" in problems
    assert "Timecode '00:02.0' is not in MM:SS.mmm format" in problems


def test_check_catches_decreasing_cuts_and_timecode_on_first_shot():
    text = T2V_PROMPT.replace("[Shot 1] Live", "[Shot 1] At 00:00.500, Live").replace(
        "close-up of bread.", "close-up. [Shot 3] At 00:02.000, the camera cuts to bread."
    )
    problems = prompt_h3.check(text, MODE_T2V, 8)
    assert "[Shot 1] must not have a cut timecode" in problems
    assert "[Shot 3] cut at 00:02.000 is not after the previous cut" in problems


def test_check_warns_about_last_line_near_the_end():
    text = T2V_PROMPT.replace("<d>[English] Morning.</d>", "").replace(
        "close-up of bread.", "close-up; the baker (S1) says: <d>[English] One two three four five six seven.</d>"
    )
    problems = prompt_h3.check(text, MODE_T2V, 5)
    assert problems and all(problem.startswith("Warning:") for problem in problems)


def test_check_catches_dialogue_without_language_and_cloud_syntax():
    text = T2V_PROMPT.replace("<d>[English] Morning.</d>", "<d>Morning.</d> as in @image 1")
    problems = prompt_h3.check(text, MODE_T2V, 5)
    assert any(problem.startswith("Dialogue without a language tag") for problem in problems)
    assert any(problem.startswith("Cloud syntax '@image 1'") for problem in problems)


def test_check_catches_wrong_or_missing_alignment():
    assert any(p.startswith("Missing alignment line") for p in prompt_h3.check(T2V_PROMPT, MODE_I2V, 5))
    fl = prompt_h3.compose(T2V_PROMPT, MODE_FLF2V, 8, has_first=True, has_last=True)
    assert prompt_h3.check(fl, MODE_FLF2V, 8) == []
    problems = prompt_h3.check(fl, MODE_FLF2V, 5)
    assert any(problem.startswith("Alignment line does not match") for problem in problems)
    assert any("does not belong to t2v" in problem for problem in prompt_h3.check(fl, MODE_T2V, 8))


def test_check_catches_ref_labels():
    labels = ["<Picture 1>", "<Video 1>"]
    prompt = prompt_h3.compose("A man waves.", MODE_REF, 8, labels=["<Picture 1>"])
    prompt = prompt.replace("A man waves.", "A man waves at <Picture 3>.")
    problems = prompt_h3.check(prompt, MODE_REF, 8, labels)
    assert "Label <Video 1> is not defined in subject_definitions" in problems
    assert "Label <Video 1> is not used in detailed_description" in problems
    assert "Label <Picture 3> has no matching reference" in problems


def test_check_catches_missing_negative_line_and_duplicates():
    assert any(p.startswith("The prompt must end") for p in prompt_h3.check(T2V_PROMPT[: -len(NEGATIVE_LINE)], MODE_T2V, 5))
    doubled = T2V_PROMPT.replace("N/A", "N/A " + NEGATIVE_LINE)
    assert "The prohibition line appears more than once" in prompt_h3.check(doubled, MODE_T2V, 5)


# --- Системные промты и сообщение -------------------------------------------------------------


@pytest.mark.parametrize("mode", [MODE_T2V, MODE_I2V, MODE_FLF2V, MODE_REF, prompt_h3.STORYBOARD])
def test_system_prompts_exist_fit_and_have_placeholders_filled(mode):
    text = prompt_h3.system_prompt(mode, PROMPT_DIR)
    assert "{{" not in text
    assert len((PROMPT_DIR / prompt_h3._PROMPT_FILES[mode]).read_bytes()) <= 12 * 1024
    if mode != prompt_h3.STORYBOARD:
        assert NEGATIVE_LINE in text


def test_system_prompt_is_reread_and_placeholders_substituted(tmp_path):
    (tmp_path / "video_system_t2v.md").write_text("max {{MAX_SECONDS}} s, end: {{NEGATIVE_LINE}}", encoding="utf-8")
    assert prompt_h3.system_prompt(MODE_T2V, tmp_path) == f"max 15 s, end: {NEGATIVE_LINE}"
    (tmp_path / "video_system_t2v.md").write_text("edited", encoding="utf-8")
    assert prompt_h3.system_prompt(MODE_T2V, tmp_path) == "edited"
    with pytest.raises(FileNotFoundError):
        prompt_h3.system_prompt(MODE_REF, tmp_path)


def test_boost_message_states_conditions():
    message = prompt_h3.build_boost_message(
        "девушка танцует", MODE_FLF2V, 5, ["<Picture 1>", "<Picture 2>"], True, True, aspect="9:16"
    )
    assert "FL2VA" in message
    assert "5.17 seconds (124 frames at 24 fps)" in message
    assert "vertical" in message and "one main subject" in message
    assert "<Picture 1>, <Picture 2>" in message
    assert "5.17-second mark" in message
    assert "Return only the finished prompt" in message
    assert message.endswith("девушка танцует")


# --- Буст -------------------------------------------------------------------------------------


def test_boost_strips_think_and_fence_and_fixes_service_lines():
    answer = "<think>plan the shots</think>\n```text\n" + T2V_PROMPT.replace(NEGATIVE_LINE, "") + "\n```"
    client = FakeClient(answer)
    result = prompt_h3.boost(client, "пекарь", MODE_I2V, 5, prompt_dir=PROMPT_DIR, has_first=True)
    assert result.prompt.startswith(prompt_h3.I2VA_LINE)
    assert "<think>" not in result.prompt and "```" not in result.prompt
    assert result.prompt.endswith(NEGATIVE_LINE)
    assert result.problems == []
    assert client.calls[0]["temperature"] == pytest.approx(0.4)
    assert client.calls[0]["max_tokens"] == 3000


def test_boost_retries_without_images_when_rejected():
    client = FakeClient(T2V_PROMPT, reject_images=True)
    image = Image.new("RGB", (8, 8))
    result = prompt_h3.boost(client, "x", MODE_I2V, 5, prompt_dir=PROMPT_DIR, has_first=True, images=[image])
    assert result.images_skipped
    assert [call["images"] for call in client.calls] == [[image], None]
    assert client.calls[1]["user"].endswith(prompt_h3.BLIND_NOTE)


def test_boost_retries_once_on_non_english_answer():
    russian = T2V_PROMPT.replace("a baker", "пекаря")
    client = FakeClient(russian, T2V_PROMPT)
    result = prompt_h3.boost(client, "пекарь", MODE_T2V, 5, prompt_dir=PROMPT_DIR)
    assert len(client.calls) == 2
    assert "пекаря" not in result.prompt


def test_boost_falls_back_to_original_text_on_empty_answers():
    client = FakeClient("<think>hmm</think>", "  ")
    result = prompt_h3.boost(client, "A cat sleeps.", MODE_T2V, 5, prompt_dir=PROMPT_DIR)
    assert "[Shot 1] A cat sleeps." in result.prompt
    assert result.problems == []


def test_a_free_edit_becomes_a_video_editing_prompt():
    labels = prompt_h3.reference_labels(["video"], [True])
    prompt = prompt_h3.compose("Turn the dog into a fox", prompt_h3.MODE_V2V, 5.0, labels=labels)
    assert "[video editing + audio reuse] The target video is an edited version of <Video 1>." in prompt
    assert "<Audio 1>: fully_copy" in prompt
    assert prompt_h3.check(prompt, prompt_h3.MODE_V2V, 5.0, labels) == []


def test_a_silent_source_is_plain_video_editing():
    prompt = prompt_h3.compose("Make it night", prompt_h3.MODE_V2V, 5.0, labels=["<Video 1>"])
    assert "[video editing] The target video is an edited version of <Video 1>." in prompt
    assert "fully_copy" not in prompt


def test_an_edit_prompt_without_the_edit_sentence_is_reported():
    prompt = prompt_h3.compose("Turn the dog into a fox", prompt_h3.MODE_REF, 5.0, labels=["<Video 1>"])
    problems = prompt_h3.check(prompt, prompt_h3.MODE_V2V, 5.0, ["<Video 1>"])
    assert any("edited version" in problem for problem in problems)


def test_the_rewriter_is_told_it_edits_video_1():
    message = prompt_h3.build_boost_message("убери собаку", prompt_h3.MODE_V2V, 5.0, ["<Video 1>"])
    assert "video editing" in message and prompt_h3.EDIT_SENTENCE in message
