from hypothesis import given
import hypothesis.strategies as st
from spacy.strings import StringStore
from spacy.util import make_tempdir

from spacy_experimental.edit_tree_lemmatizer.edit_trees import EditTrees


def test_dutch():
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add("deelt", "delen")
    assert trees.tree_to_str(tree) == "(m 0 3 () (m 0 2 (s '' 'l') (s 'lt' 'n')))"

    tree = trees.add("gedeeld", "delen")
    assert (
        trees.tree_to_str(tree) == "(m 2 3 (s 'ge' '') (m 0 2 (s '' 'l') (s 'ld' 'n')))"
    )


def test_from_to_bytes():
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add("deelt", "delen")
    trees.add("gedeeld", "delen")

    b = trees.to_bytes()

    trees2 = EditTrees(strings)
    trees2.from_bytes(b)

    # Verify that the nodes did not change.
    assert len(trees) == len(trees2)
    for i in range(len(trees)):
        assert trees.tree_to_str(i) == trees2.tree_to_str(i)

    # Reinserting the same trees should not add new nodes.
    trees2.add("deelt", "delen")
    trees2.add("gedeeld", "delen")
    assert len(trees) == len(trees2)


def test_from_to_disk():
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add("deelt", "delen")
    trees.add("gedeeld", "delen")

    trees2 = EditTrees(strings)
    with make_tempdir() as temp_dir:
        trees_file = temp_dir / "edit_trees.bin"
        trees.to_disk(trees_file)
        trees2 = trees2.from_disk(trees_file)

    # Verify that the nodes did not change.
    assert len(trees) == len(trees2)
    for i in range(len(trees)):
        assert trees.tree_to_str(i) == trees2.tree_to_str(i)

    # Reinserting the same trees should not add new nodes.
    trees2.add("deelt", "delen")
    trees2.add("gedeeld", "delen")
    assert len(trees) == len(trees2)


@given(st.text(), st.text())
def test_roundtrip(form, lemma):
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma


@given(st.text(alphabet="ab"), st.text(alphabet="ab"))
def test_roundtrip_small_alphabet(form, lemma):
    # Test with small alphabets to have more overlap.
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma


def test_unapplicable_trees():
    strings = StringStore()
    trees = EditTrees(strings)
    tree3 = trees.add("deelt", "delen")

    # Replacement fails.
    assert trees.apply(tree3, "deeld") == None

    # Suffix + prefix are too large.
    assert trees.apply(tree3, "de") == None


def test_empty_strings():
    strings = StringStore()
    trees = EditTrees(strings)
    no_change = trees.add("xyz", "xyz")
    empty = trees.add("", "")
    assert no_change == empty
