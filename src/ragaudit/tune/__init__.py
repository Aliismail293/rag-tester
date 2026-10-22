"""`ragaudit tune`: closed-loop config search on the dev split, scored once on the locked test split.

Nothing in this package may import `ragaudit.storage.lock` or read the
store directly; it sees questions only through `DevSet` and test results
only as aggregate scores from `ragaudit.runner.locked`. A unit test
enforces this.
"""
