from nose.tools import eq_
from bem.bem import BladeSection


def _assert_alpha(twist, inflow_angle, pitch, expected_alpha):
    class MockFoil:
        def lift_drag(self, alpha):
            eq_(alpha, expected_alpha)
            return [0, 0]
    section = BladeSection(1, twist, MockFoil())
    section.force_coefficients(inflow_angle, pitch)


class BladeSection_Tests:
    def test_holds_chord_twist_and_foil(self):
        section = BladeSection(1, 2, 3)
        eq_(section.chord, 1)
        eq_(section.twist, 2)
        eq_(section.foil, 3)

    def test_force_coefficients_alpha(self):
        # With zero twist and pitch, inflow angle equals alpha
        _assert_alpha(0, 0, 0, 0)
        _assert_alpha(0, 1.2, 0, 1.2)
        _assert_alpha(0, -.4, 0, -.4)

        # Twist and pitch both have the same effect
        _assert_alpha(0.23, 1.0, 0.34, 1.0 - 0.23 - 0.34)
