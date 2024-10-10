import pytest
from data.scripts.generate_annotated_barline_data import hex_to_rgb

@pytest.mark.parametrize(
    "hex_color, expected",
    [
        ("#7c180d", (124, 24, 13)),
    ]
)
def test_hex_to_rgb(hex_color: str, expected: tuple[int, int, int]):
    # Arange
    # Act
    answer = hex_to_rgb(hex_color)
    # Assert
    assert answer == expected