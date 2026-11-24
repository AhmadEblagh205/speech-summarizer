from cli import _safe_stem


def test_safe_stem_for_run_directory():
    assert _safe_stem("My lecture (week 1).mp3") == "My_lecture__week_1_"
