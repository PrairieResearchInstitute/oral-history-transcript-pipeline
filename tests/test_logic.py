"""Unit tests for the pure-Python parts. Run on the host or in the image: python -m pytest tests/"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import check, render, speakers  # noqa: E402

SPEAKERS = [
    {"name": "Bill Shilts", "role": "interviewee", "affiliation": None},
    {"name": "Emily Benton", "role": "interviewer", "affiliation": None},
    {"name": "Felipe Menanteau", "role": "interviewer", "affiliation": None},
]


def seg(a, b, text, words=None):
    return {"start": a, "end": b, "text": text, "words": words or []}


def test_cluster_naming_by_talk_time_and_first_appearance():
    d = [{"start": 0, "end": 20, "cluster": 2},     # first voice: lead interviewer
         {"start": 20, "end": 300, "cluster": 0},   # most talk: interviewee
         {"start": 300, "end": 310, "cluster": 1},  # later voice: second interviewer
         {"start": 310, "end": 600, "cluster": 0}]
    names, stats = speakers.name_clusters(d, SPEAKERS)
    assert names == {0: "Bill Shilts", 2: "Emily Benton", 1: "Felipe Menanteau"}
    names, _ = speakers.name_clusters(d, SPEAKERS, {2: "Felipe Menanteau", 1: "Emily Benton"})
    assert names[2] == "Felipe Menanteau" and names[1] == "Emily Benton"


def test_assign_whole_segments_and_nearest():
    d = [{"start": 0, "end": 5, "speaker": "A"}, {"start": 5, "end": 20, "speaker": "B"}]
    p = render.assign([seg(0, 4.5, "Hello."), seg(6, 10, "Hi."), seg(21, 22, "Late.")], d)
    assert [x["speaker"] for x in p] == ["A", "B", "B"]
    p = render.assign([seg(40, 41, "Far away.")], d)
    assert p[0]["speaker"] == render.UNKNOWN


def test_split_mixed_segment_by_words():
    d = [{"start": 0, "end": 4, "speaker": "A"}, {"start": 4, "end": 8, "speaker": "B"}]
    words = [{"text": " So", "start": 0.2, "end": 0.5}, {"text": " what", "start": 0.6, "end": 1.0},
             {"text": " happened?", "start": 1.1, "end": 3.5},
             {"text": " Well,", "start": 4.2, "end": 4.6}, {"text": " we", "start": 4.7, "end": 5.0},
             {"text": " moved.", "start": 5.1, "end": 7.5}]
    p = render.assign([seg(0, 8, "So what happened? Well, we moved.", words)], d)
    assert [(x["speaker"], x["text"]) for x in p] == [("A", "So what happened?"), ("B", "Well, we moved.")]


def test_paragraph_rules():
    s = "word " * 10
    sent = s.strip() + "."
    text = " ".join([sent] * 13)          # 130 words: one break after 120, tail of 10 merged back
    assert len(render.paragraphs(text)) == 1
    text = " ".join([sent] * 15)          # 150 words: tail of 30 stays its own paragraph
    assert len(render.paragraphs(text)) == 2


def test_corrections_whole_words_only():
    pieces = [seg(1, 2, "We mapped the Muhammad Aquifer near Mantino."), seg(3, 4, "Mantinos is not a word.")]
    applied = render.apply_corrections(pieces, {"Mantino": "Manteno", "Muhammad Aquifer": "Mahomet Aquifer"})
    assert pieces[0]["text"] == "We mapped the Mahomet Aquifer near Manteno."
    assert pieces[1]["text"] == "Mantinos is not a word."
    assert applied["Mantino"]["count"] == 1


def test_names_not_in_glossary():
    pieces = [seg(10, 12, "Then Ada Lovelace called me. I went to Peoria with Ann Smith."),
              seg(20, 22, "Anyway, it was hard. Peoria was cold.")]
    names = render.find_names(pieces, ["Ada Lovelace"])
    assert "Ann Smith" in names and "Peoria" in names
    assert "Ada Lovelace" not in names and "Anyway" not in names and "Then" not in names


def test_check_finds_loops_gaps_and_short_end():
    segs = [seg(0, 2, "a"), seg(2, 4, "same"), seg(4, 6, "same"), seg(6, 8, "same"), seg(30, 32, "b")]
    kinds = [i["kind"] for i in check.find_issues(segs, 100)]
    assert kinds == ["repeat", "gap", "end"]


def test_paragraph_split_keeps_closing_quotes():
    long = " ".join(["word"] * 125)
    text = f'He said, "{long}?" And then we left.'
    assert '?"' in " ".join(render.paragraphs(text))
