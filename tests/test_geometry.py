import pytest

from rsna_knee.audit.dicom import derive_geometry


@pytest.mark.parametrize(
    ("orientation", "position", "expected_plane", "expected_scalar"),
    [
        ([1, 0, 0, 0, 1, 0], [0, 0, 7], "Axial", 7.0),
        ([0, 1, 0, 0, 0, 1], [3, 0, 0], "Sagittal", 3.0),
        ([1, 0, 0, 0, 0, 1], [0, -4, 0], "Coronal", 4.0),
    ],
)
def test_derive_geometry(orientation, position, expected_plane, expected_scalar) -> None:
    plane, normal, scalar = derive_geometry(orientation, position)
    assert plane == expected_plane
    assert normal is not None
    assert scalar == pytest.approx(expected_scalar)


def test_invalid_orientation_is_unknown() -> None:
    assert derive_geometry([1, 2], [0, 0, 0]) == ("Unknown", None, None)

