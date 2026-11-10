import pytest

from core.validation import looks_like_role


@pytest.mark.parametrize("role", ["Data analyst", "QA", "HR", "C++ developer", "3D artist", "Analyst 2",
                                  "Data Scientist (ML)", "软件工程师", "  Nurse  ", "teddy bear", "x" * 39 + "y" * 41])
def test_real_or_unjudgeable_roles_pass(role):
    assert looks_like_role(role)


@pytest.mark.parametrize("role", ["", "   ", "123", "???", "a", "-- 5", "aaaa", "ZZzz", "C++", "R",
                                  "x" * 40 + "y" * 41])
def test_obvious_junk_roles_fail(role):
    assert not looks_like_role(role)
