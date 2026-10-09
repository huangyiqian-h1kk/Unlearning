"""Observe actual sampled rows without replacing or replaying the sampler."""

from collections import Counter, defaultdict

from .io import read, write, sha


class SamplingAudit:
    def __init__(self, data, root, identity, *, rank=0, resume=None, first_step=0):
        self.root, self.identity, self.rank = root, identity, rank
        self.objects = {group: {id(row): i for i, row in enumerate(rows)} for group, rows in data.items()}
        self.fingerprints = {}
        for group, rows in data.items():
            index = defaultdict(list)
            for i, row in enumerate(rows):
                index[sha(row)].append(i)
            self.fingerprints[group] = index
        self.counts = {group: [0] * len(rows) for group, rows in data.items()}
        if resume is not None:
            saved = read(resume / "sampling_state.json")
            if saved["identity"] != identity or saved["step"] != first_step:
                raise ValueError("Sampling audit checkpoint identity/step mismatch")
            for group, counts in self.counts.items():
                values = saved["counts"][group]
                if len(values) != len(counts) or any(type(x) is not int or x < 0 for x in values):
                    raise ValueError("Sampling audit dataset/count mismatch")
            self.counts = saved["counts"]

    def indices(self, group, rows):
        result = []
        for row in rows:
            index = self.objects[group].get(id(row))
            if index is None:
                matches = self.fingerprints[group].get(sha(row), [])
                if len(matches) != 1:
                    raise ValueError("Cannot identify sampled row uniquely in the frozen dataset")
                index = matches[0]
            result.append(index)
        return result

    def observe(self, groups, step, micro, *, gather_records=None):
        local = {group: self.indices(group, rows) for group, rows in groups.items()}
        if gather_records is None:
            import torch.distributed as dist
            if dist.is_initialized():
                ranks = [None] * dist.get_world_size()
                dist.all_gather_object(ranks, local)
            else:
                ranks = [local]
        else:
            ranks = gather_records(local)
        summary, sampled = {}, {}
        for group, counts in self.counts.items():
            sampled[group] = [index for rank in ranks for index in rank[group]]
            for index, times in Counter(sampled[group]).items():
                counts[index] += times
            total, unique = sum(counts), sum(x > 0 for x in counts)
            summary[group] = {"draws": total, "unique": unique, "rows": len(counts),
                "coverage": unique / len(counts), "min_count": min(counts),
                "max_count": max(counts), "mean_count": total / len(counts)}
        if self.rank == 0:
            write(self.root / "sampling" / f"step-{step:06d}-micro-{micro}.json",
                {"identity": self.identity, "step": step, "microbatch": micro,
                 "summary": summary, "row_indices": sampled,
                 "note": "Observed draws; use checkpoint sampling_state for committed coverage"})

    def snapshot(self, checkpoint, step):
        if self.rank == 0:
            write(checkpoint / "sampling_state.json", {"identity": self.identity,
                "step": step, "counts": self.counts})
