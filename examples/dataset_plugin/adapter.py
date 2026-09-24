"""Generic text-triple dataset adapter example."""

import hashlib

from builtin_plugins import DatasetBundle


class DatasetAdapter:
    def __init__(self, definition, context):
        del context
        self.definition = definition

    def load(self, config):
        del config
        paths = [self.definition.root / name for name in ("train.tsv", "valid.tsv", "test.tsv")]
        raw_splits = []
        entities, relations = {}, {}
        for path in paths:
            triples = []
            for line in path.read_text(encoding="utf-8").splitlines():
                head, relation, tail = line.split("\t")[:3]
                head_id = entities.setdefault(head, len(entities))
                tail_id = entities.setdefault(tail, len(entities))
                relation_id = relations.setdefault(relation, len(relations))
                triples.append((head_id, relation_id, tail_id))
            raw_splits.append(triples)
        train, validation, test = raw_splits
        return DatasetBundle(
            splits={"train": train, "validation": validation, "test": test},
            features=set(self.definition.manifest.get("features", [])),
            metadata={"entities": len(entities), "relations": len(relations)},
            mappings={"entity_to_id": entities, "relation_to_id": relations},
            known_triples=set(train + validation + test),
            source_files=paths,
        )

    def fingerprint(self, config, bundle):
        del config
        digest = hashlib.sha256()
        for path in bundle.source_files:
            digest.update(path.read_bytes())
        return digest.hexdigest()
