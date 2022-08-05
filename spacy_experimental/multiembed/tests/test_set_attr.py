import spacy
import srsly
import pytest

set_attr = spacy.registry.misc.get("spacy-experimental.set_attr.v1")


def test_set_attr():
    attr = {"this": "that"}
    srsly.write_msgpack("test.msg", attr)
    setter = set_attr("test.msg", "tagger", "attr", "output_layer")
    nlp = spacy.blank("en")
    tagger = nlp.add_pipe("tagger")
    setter(nlp)
    layer = tagger.model.get_ref("output_layer")
    assert layer.attrs["attr"] == attr


def test_set_attr_no_component():
    attr = {"this": "that"}
    srsly.write_msgpack("test.msg", attr)
    setter = set_attr("test.msg", "tagger", "attr", "output_layer")
    nlp = spacy.blank("en")
    with pytest.raises(ValueError, match="non-existing component"):
        setter(nlp)


def test_set_attr_no_layer():
    attr = {"this": "that"}
    srsly.write_msgpack("test.msg", attr)
    setter = set_attr("test.msg", "tagger", "attr", "nonsense")
    nlp = spacy.blank("en")
    nlp.add_pipe("tagger")
    with pytest.raises(ValueError, match="Haven't found nonsense"):
        setter(nlp)


def test_set_attr_multiple_occur():
    attr = {"this": "that"}
    srsly.write_msgpack("test.msg", attr)
    setter = set_attr("test.msg", "tagger", "attr", "maxout")
    nlp = spacy.blank("en")
    nlp.add_pipe("tagger")
    with pytest.raises(ValueError, match="multiple layers named maxout"):
        setter(nlp)
