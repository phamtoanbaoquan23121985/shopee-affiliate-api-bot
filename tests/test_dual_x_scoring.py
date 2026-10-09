import unittest
from decimal import Decimal
from dual_x.scoring import Product, evaluate

class TestScoring(unittest.TestCase):
    def test_missing_conversion_is_not_fabricated(self):
        p = Product("shopee", "1", Decimal("100"), Decimal(".1"))
        self.assertIsNone(evaluate(p)["expected_net_per_1000_clicks"])

    def test_profit_calculation(self):
        p = Product("lazada", "2", Decimal("100"), Decimal(".1"),
                    Decimal(".02"), Decimal(".5"), 2, Decimal(".05"), Decimal(".1"))
        self.assertEqual(Decimal(evaluate(p)["expected_net_per_1000_clicks"]), Decimal("310"))

    def test_invalid_rate(self):
        p = Product("shopee", "3", Decimal("100"), Decimal("1.5"))
        with self.assertRaises(ValueError):
            evaluate(p)

if __name__ == "__main__":
    unittest.main()
