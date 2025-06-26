from src.tokenizer import Alphabet

def test_alphabet_size():
    alphabet = Alphabet()
    assert len(alphabet) == 33

def test_alphabet_special_tokens():
    alphabet = Alphabet()
    assert alphabet.get_idx("$") == 28
    assert alphabet.get_idx("-") == 29
    assert alphabet.get_idx("!") == 30
    assert alphabet.get_idx("?") == 31
    assert alphabet.get_idx("#") == 32

def test_encode():
    alphabet = Alphabet()
    sequence = "PRTEINS!"
    encoded = alphabet.encode(sequence)
    assert encoded == [10, 6, 7, 5, 8, 13, 4, 30]

def test_encode_unknown_tokens():
    alphabet = Alphabet()
    bad_sequence = '@[]/`~'
    encoded = alphabet.encode(bad_sequence)
    assert encoded == [31, 31, 31, 31, 31, 31]