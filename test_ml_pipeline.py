import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "attendance": [40, 45, 50, 55, 60, 65, 70, 75, 80, 90],
            "result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        })

        self.X = self.data[["study_hours", "attendance"]]
        self.y = self.data["result"]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42,
                stratify=self.y
            )
        )

        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

        self.predictions = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(
            self.y_test, self.predictions
        )

    def test_model_training(self):
        self.assertIsNotNone(self.model)

    def test_predictions(self):
        self.assertEqual(len(self.predictions), len(self.y_test))

    def test_prediction_values(self):
        self.assertTrue(
            set(self.predictions).issubset({0, 1})
        )

    def test_accuracy_range(self):
        self.assertGreaterEqual(self.accuracy, 0.0)
        self.assertLessEqual(self.accuracy, 1.0)

    def test_metrics(self):
        metrics = {"accuracy": self.accuracy}
        accuracy = metrics["accuracy"]
        self.assertIsInstance(accuracy, float)


if __name__ == "__main__":
    unittest.main()
