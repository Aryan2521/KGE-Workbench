"""Small deterministic family graph used only as a local runner smoke-test dataset."""

from __future__ import annotations

import random


class SyntheticKG:
    """Generate family triples while keeping model code in the Thesis new folder."""

    def __init__(self, num_families: int = 15, seed: int = 42):
        self._rng = random.Random(seed)
        self.entities: list[str] = []
        self.entity_to_id: dict[str, int] = {}
        self.relations = ["spouse_of", "parent_of", "child_of", "sibling_of", "grandparent_of"]
        self.relation_to_id = {relation: index for index, relation in enumerate(self.relations)}
        self.triples: list[tuple[int, int, int]] = []
        for family in range(num_families):
            self._add_family(family)
        self.triples = sorted(set(self.triples))

    def _entity(self, name: str) -> int:
        if name not in self.entity_to_id:
            self.entity_to_id[name] = len(self.entities)
            self.entities.append(name)
        return self.entity_to_id[name]

    def _add(self, head: str, relation: str, tail: str) -> None:
        self.triples.append((self._entity(head), self.relation_to_id[relation], self._entity(tail)))

    def _add_family(self, family: int) -> None:
        grandfather, grandmother = f"G_Father_{family}", f"G_Mother_{family}"
        father, mother = f"Father_{family}", f"Mother_{family}"
        children = [f"Child_{family}_{index}" for index in range(self._rng.randint(2, 3))]
        self._add(grandfather, "spouse_of", grandmother)
        self._add(grandmother, "spouse_of", grandfather)
        self._add(father, "spouse_of", mother)
        self._add(mother, "spouse_of", father)
        for grandparent in (grandfather, grandmother):
            self._add(grandparent, "parent_of", father)
            self._add(father, "child_of", grandparent)
        for child in children:
            for parent in (father, mother):
                self._add(parent, "parent_of", child)
                self._add(child, "child_of", parent)
            for grandparent in (grandfather, grandmother):
                self._add(grandparent, "grandparent_of", child)
        for child in children:
            for sibling in children:
                if child != sibling:
                    self._add(child, "sibling_of", sibling)

    def get_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        del test_ratio
        rng = random.Random(seed)
        shuffled = list(self.triples)
        rng.shuffle(shuffled)
        train: list[tuple[int, int, int]] = []
        remaining: list[tuple[int, int, int]] = []
        entities: set[int] = set()
        relations: set[int] = set()
        for triple in shuffled:
            head, relation, tail = triple
            if head not in entities or tail not in entities or relation not in relations:
                train.append(triple)
                entities.update((head, tail))
                relations.add(relation)
            else:
                remaining.append(triple)
        train_target = int(len(shuffled) * train_ratio)
        validation_target = int(len(shuffled) * val_ratio)
        fill = max(0, train_target - len(train))
        train.extend(remaining[:fill])
        remaining = remaining[fill:]
        return train, remaining[:validation_target], remaining[validation_target:]
