import unittest
import random

import numpy as np
import torch

import benchmark_runner


class ToyTransE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_entities = 3
        self.p_norm = 1
        self.entity_embeddings = torch.nn.Embedding.from_pretrained(
            torch.tensor([[0.0], [1.0], [2.0]]), freeze=True
        )
        self.relation_embeddings = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0]]), freeze=True
        )

    def forward(self, heads, relations, tails):
        return torch.abs(
            self.entity_embeddings(heads)
            + self.relation_embeddings(relations)
            - self.entity_embeddings(tails)
        ).sum(dim=1)


class BenchmarkMetricTests(unittest.TestCase):
    def test_split_selection_matches_updated_thesis_protocol(self):
        triples = [(index, 0, index + 1) for index in range(20)]
        state = random.getstate()
        selected = benchmark_runner.select_split(triples, 0.5, 7, 42)
        self.assertEqual(len(selected), 7)
        self.assertEqual(selected, benchmark_runner.select_split(triples, 0.5, 7, 42))
        self.assertNotEqual(selected, benchmark_runner.select_split(triples, 0.5, 7, 43))
        self.assertEqual(state, random.getstate())
        self.assertEqual(triples, [(index, 0, index + 1) for index in range(20)])

    def test_filtered_ranking_calculates_real_hits(self):
        model = ToyTransE()
        triple = (0, 0, 1)
        metrics = benchmark_runner.ranking_metrics(
            model, [triple], {triple}, 3, torch.device("cpu"), chunk_size=2
        )
        self.assertEqual(metrics["mean_rank"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["hits_at_1"], 1.0)
        self.assertEqual(metrics["hits_at_3"], 1.0)
        self.assertEqual(metrics["hits_at_10"], 1.0)

    def test_threshold_separates_positive_and_negative_distances(self):
        threshold = benchmark_runner.best_threshold(
            np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 1.0])
        )
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLess(threshold, 0.8)

    def test_higher_is_better_adapter_is_ranked_in_correct_direction(self):
        class HigherScoreAdapter:
            score_direction = "higher"

            @staticmethod
            def score(model, heads, relations, tails):
                return -model(heads, relations, tails)

        model = ToyTransE()
        triple = (0, 0, 1)
        metrics = benchmark_runner.ranking_metrics(
            model, [triple], {triple}, 3, torch.device("cpu"), 2, HigherScoreAdapter()
        )
        self.assertEqual(metrics["mrr"], 1.0)

    def test_uncertainty_presets_are_explicit(self):
        config = {"uncertainty_level": "medium"}
        self.assertEqual(benchmark_runner.uncertainty_values(config), (0.8, 0.2, 0.15))


if __name__ == "__main__":
    unittest.main()
