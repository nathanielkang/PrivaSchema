"""Relational schema representation for multi-table datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Table:
    """A single table in a relational schema."""

    name: str
    attributes: list[str]
    dtypes: dict[str, str]  # attr -> "categorical" | "numerical"
    primary_key: str | None = None

    @property
    def num_attributes(self) -> int:
        return len(self.attributes)

    def categorical_columns(self) -> list[str]:
        return [a for a in self.attributes if self.dtypes.get(a) == "categorical"]

    def numerical_columns(self) -> list[str]:
        return [a for a in self.attributes if self.dtypes.get(a) == "numerical"]


@dataclass
class ForeignKey:
    """A foreign-key constraint between two tables."""

    child_table: str
    parent_table: str
    child_col: str
    parent_col: str


@dataclass
class RelationalSchema:
    """Full relational schema: tables + FK constraints + topological order."""

    tables: dict[str, Table] = field(default_factory=dict)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def add_table(self, table: Table) -> None:
        self.tables[table.name] = table

    def add_foreign_key(self, fk: ForeignKey) -> None:
        assert fk.child_table in self.tables, f"Child table '{fk.child_table}' not in schema"
        assert fk.parent_table in self.tables, f"Parent table '{fk.parent_table}' not in schema"
        self.foreign_keys.append(fk)

    def parent_tables(self, table_name: str) -> list[str]:
        """Return names of all parent tables for a given table."""
        return [fk.parent_table for fk in self.foreign_keys if fk.child_table == table_name]

    def child_tables(self, table_name: str) -> list[str]:
        """Return names of all child tables for a given table."""
        return [fk.child_table for fk in self.foreign_keys if fk.parent_table == table_name]

    def get_fk(self, child: str, parent: str) -> ForeignKey | None:
        """Get FK constraint between a specific child-parent pair."""
        for fk in self.foreign_keys:
            if fk.child_table == child and fk.parent_table == parent:
                return fk
        return None

    def topological_order(self) -> list[str]:
        """Kahn's algorithm: parents before children."""
        in_degree: dict[str, int] = {t: 0 for t in self.tables}
        for fk in self.foreign_keys:
            in_degree[fk.child_table] += 1

        queue = [t for t, d in in_degree.items() if d == 0]
        order: list[str] = []

        while queue:
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for child in self.child_tables(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.tables):
            raise ValueError("Cycle detected in foreign-key graph")
        return order

    @property
    def num_tables(self) -> int:
        return len(self.tables)


def parse_schema(metadata: dict[str, Any]) -> RelationalSchema:
    """Build a RelationalSchema from a metadata dictionary.

    Expected format::

        {
            "tables": {
                "accounts": {
                    "attributes": ["account_id", "district_id", "frequency", "date"],
                    "dtypes": {"account_id": "categorical", "district_id": "categorical",
                               "frequency": "categorical", "date": "numerical"},
                    "primary_key": "account_id"
                },
                ...
            },
            "foreign_keys": [
                {"child_table": "accounts", "parent_table": "districts",
                 "child_col": "district_id", "parent_col": "district_id"},
                ...
            ]
        }
    """
    schema = RelationalSchema()

    for tname, tinfo in metadata["tables"].items():
        table = Table(
            name=tname,
            attributes=tinfo["attributes"],
            dtypes=tinfo["dtypes"],
            primary_key=tinfo.get("primary_key"),
        )
        schema.add_table(table)

    for fk_info in metadata.get("foreign_keys", []):
        fk = ForeignKey(**fk_info)
        schema.add_foreign_key(fk)

    return schema
