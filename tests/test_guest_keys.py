from guest_keys import generate_numeric_key, hash_guest_key


def test_guest_key_is_exactly_eight_digits():
    for _ in range(50):
        key = generate_numeric_key()
        assert len(key) == 8
        assert key.isdigit()


def test_guest_key_hash_does_not_equal_plaintext():
    key = "12345678"
    digest = hash_guest_key(key)
    assert digest != key
    assert len(digest) == 64


def test_generated_keys_are_not_constant():
    keys = {generate_numeric_key() for _ in range(20)}
    assert len(keys) > 1
