"""Multi-table dataset loaders and synthetic data generators.

Supported datasets (7 total):
  - berka          8 tables   Czech financial dataset
  - rossmann       3 tables   Store sales
  - imdb           5 tables   Movie database
  - tpch           5 tables   TPC-H benchmark (simplified)
  - university     5 tables   Academic records
  - walmart        4 tables   Retail store sales
  - synthetic_star N tables   Configurable star schema
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.schema import ForeignKey, RelationalSchema, Table

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ======================================================================
# Berka Financial Dataset  (8 tables, realistic synthetic proxy)
# ======================================================================

def load_berka(data_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Load or generate a Berka-like financial dataset (8 tables).

    If CSV files exist under ``data_dir/berka/``, they are loaded directly.
    Otherwise a realistic synthetic proxy with matching schema is generated.
    """
    root = (data_dir or DATA_DIR) / "berka"

    table_files = {
        "district": root / "district.csv",
        "account": root / "account.csv",
        "client": root / "client.csv",
        "disp": root / "disp.csv",
        "order": root / "order.csv",
        "trans": root / "trans.csv",
        "loan": root / "loan.csv",
        "card": root / "card.csv",
    }

    if all(f.exists() for f in table_files.values()):
        logger.info("Loading Berka from %s", root)
        data = {name: pd.read_csv(path) for name, path in table_files.items()}
    else:
        logger.info("Generating synthetic Berka proxy (no CSV found at %s)", root)
        data = _generate_berka_proxy()

    schema = _berka_schema()
    data = _align_columns(data, schema)
    return data, schema


def _berka_schema() -> RelationalSchema:
    schema = RelationalSchema()

    schema.add_table(Table(
        name="district", primary_key="district_id",
        attributes=["district_id", "name", "region", "population", "avg_salary"],
        dtypes={"district_id": "categorical", "name": "categorical",
                "region": "categorical", "population": "numerical", "avg_salary": "numerical"},
    ))
    schema.add_table(Table(
        name="account", primary_key="account_id",
        attributes=["account_id", "district_id", "frequency", "date"],
        dtypes={"account_id": "categorical", "district_id": "categorical",
                "frequency": "categorical", "date": "numerical"},
    ))
    schema.add_table(Table(
        name="client", primary_key="client_id",
        attributes=["client_id", "district_id", "birth_date", "gender"],
        dtypes={"client_id": "categorical", "district_id": "categorical",
                "birth_date": "numerical", "gender": "categorical"},
    ))
    schema.add_table(Table(
        name="disp", primary_key="disp_id",
        attributes=["disp_id", "client_id", "account_id", "type"],
        dtypes={"disp_id": "categorical", "client_id": "categorical",
                "account_id": "categorical", "type": "categorical"},
    ))
    schema.add_table(Table(
        name="order", primary_key="order_id",
        attributes=["order_id", "account_id", "amount", "category"],
        dtypes={"order_id": "categorical", "account_id": "categorical",
                "amount": "numerical", "category": "categorical"},
    ))
    schema.add_table(Table(
        name="trans", primary_key="trans_id",
        attributes=["trans_id", "account_id", "date", "type", "amount", "balance"],
        dtypes={"trans_id": "categorical", "account_id": "categorical",
                "date": "numerical", "type": "categorical",
                "amount": "numerical", "balance": "numerical"},
    ))
    schema.add_table(Table(
        name="loan", primary_key="loan_id",
        attributes=["loan_id", "account_id", "date", "amount", "duration", "status"],
        dtypes={"loan_id": "categorical", "account_id": "categorical",
                "date": "numerical", "amount": "numerical",
                "duration": "numerical", "status": "categorical"},
    ))
    schema.add_table(Table(
        name="card", primary_key="card_id",
        attributes=["card_id", "disp_id", "type", "issued"],
        dtypes={"card_id": "categorical", "disp_id": "categorical",
                "type": "categorical", "issued": "numerical"},
    ))

    schema.add_foreign_key(ForeignKey("account", "district", "district_id", "district_id"))
    schema.add_foreign_key(ForeignKey("client", "district", "district_id", "district_id"))
    schema.add_foreign_key(ForeignKey("disp", "client", "client_id", "client_id"))
    schema.add_foreign_key(ForeignKey("disp", "account", "account_id", "account_id"))
    schema.add_foreign_key(ForeignKey("order", "account", "account_id", "account_id"))
    schema.add_foreign_key(ForeignKey("trans", "account", "account_id", "account_id"))
    schema.add_foreign_key(ForeignKey("loan", "account", "account_id", "account_id"))
    schema.add_foreign_key(ForeignKey("card", "disp", "disp_id", "disp_id"))

    return schema


def _generate_berka_proxy(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate a small realistic proxy of the Berka dataset."""
    rng = np.random.default_rng(seed)
    N_DISTRICTS, N_ACCOUNTS, N_CLIENTS = 77, 500, 600
    N_DISPS, N_ORDERS, N_TRANS, N_LOANS, N_CARDS = 700, 400, 5000, 200, 150

    regions = ["Prague", "Central Bohemia", "South Bohemia", "West Bohemia",
               "North Bohemia", "East Bohemia", "South Moravia", "North Moravia"]

    districts = pd.DataFrame({
        "district_id": np.arange(1, N_DISTRICTS + 1),
        "name": [f"District_{i}" for i in range(1, N_DISTRICTS + 1)],
        "region": rng.choice(regions, N_DISTRICTS),
        "population": rng.integers(5000, 500000, N_DISTRICTS),
        "avg_salary": rng.integers(6000, 15000, N_DISTRICTS),
    })

    accounts = pd.DataFrame({
        "account_id": np.arange(1, N_ACCOUNTS + 1),
        "district_id": rng.choice(districts["district_id"].values, N_ACCOUNTS),
        "frequency": rng.choice(["monthly", "weekly", "after_transaction"], N_ACCOUNTS),
        "date": rng.integers(930101, 981231, N_ACCOUNTS),
    })

    clients = pd.DataFrame({
        "client_id": np.arange(1, N_CLIENTS + 1),
        "district_id": rng.choice(districts["district_id"].values, N_CLIENTS),
        "birth_date": rng.integers(400101, 850101, N_CLIENTS),
        "gender": rng.choice(["M", "F"], N_CLIENTS),
    })

    disps = pd.DataFrame({
        "disp_id": np.arange(1, N_DISPS + 1),
        "client_id": rng.choice(clients["client_id"].values, N_DISPS),
        "account_id": rng.choice(accounts["account_id"].values, N_DISPS),
        "type": rng.choice(["OWNER", "DISPONENT"], N_DISPS, p=[0.75, 0.25]),
    })

    orders = pd.DataFrame({
        "order_id": np.arange(1, N_ORDERS + 1),
        "account_id": rng.choice(accounts["account_id"].values, N_ORDERS),
        "amount": rng.uniform(100, 50000, N_ORDERS).round(2),
        "category": rng.choice(["insurance", "loan_payment", "household", "other"], N_ORDERS),
    })

    trans = pd.DataFrame({
        "trans_id": np.arange(1, N_TRANS + 1),
        "account_id": rng.choice(accounts["account_id"].values, N_TRANS),
        "date": rng.integers(930101, 981231, N_TRANS),
        "type": rng.choice(["credit", "withdrawal"], N_TRANS),
        "amount": rng.uniform(10, 100000, N_TRANS).round(2),
        "balance": rng.uniform(-10000, 200000, N_TRANS).round(2),
    })

    loans = pd.DataFrame({
        "loan_id": np.arange(1, N_LOANS + 1),
        "account_id": rng.choice(accounts["account_id"].values, N_LOANS),
        "date": rng.integers(930101, 981231, N_LOANS),
        "amount": rng.uniform(1000, 500000, N_LOANS).round(2),
        "duration": rng.choice([12, 24, 36, 48, 60], N_LOANS),
        "status": rng.choice(["A", "B", "C", "D"], N_LOANS, p=[0.6, 0.15, 0.15, 0.1]),
    })

    cards = pd.DataFrame({
        "card_id": np.arange(1, N_CARDS + 1),
        "disp_id": rng.choice(disps["disp_id"].values, N_CARDS),
        "type": rng.choice(["classic", "junior", "gold"], N_CARDS, p=[0.6, 0.25, 0.15]),
        "issued": rng.integers(930101, 981231, N_CARDS),
    })

    return {
        "district": districts, "account": accounts, "client": clients,
        "disp": disps, "order": orders, "trans": trans,
        "loan": loans, "card": cards,
    }


# ======================================================================
# Rossmann Store Sales  (3 tables)
# ======================================================================

def load_rossmann(data_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Load or generate a Rossmann-like store sales dataset (3 tables)."""
    root = (data_dir or DATA_DIR) / "rossmann"

    table_files = {
        "store": root / "store.csv",
        "train": root / "train.csv",
        "store_states": root / "store_states.csv",
    }

    if all(f.exists() for f in table_files.values()):
        logger.info("Loading Rossmann from %s", root)
        data = {name: pd.read_csv(path) for name, path in table_files.items()}
    else:
        logger.info("Generating synthetic Rossmann proxy")
        data = _generate_rossmann_proxy()

    schema = _rossmann_schema()
    data = _align_columns(data, schema)
    return data, schema


def _rossmann_schema() -> RelationalSchema:
    schema = RelationalSchema()

    schema.add_table(Table(
        name="store_states", primary_key="state",
        attributes=["state", "state_name"],
        dtypes={"state": "categorical", "state_name": "categorical"},
    ))
    schema.add_table(Table(
        name="store", primary_key="store_id",
        attributes=["store_id", "store_type", "assortment", "competition_distance",
                     "promo2", "state"],
        dtypes={"store_id": "categorical", "store_type": "categorical",
                "assortment": "categorical", "competition_distance": "numerical",
                "promo2": "categorical", "state": "categorical"},
    ))
    schema.add_table(Table(
        name="train", primary_key=None,
        attributes=["store_id", "day_of_week", "sales", "customers", "open", "promo"],
        dtypes={"store_id": "categorical", "day_of_week": "categorical",
                "sales": "numerical", "customers": "numerical",
                "open": "categorical", "promo": "categorical"},
    ))

    schema.add_foreign_key(ForeignKey("store", "store_states", "state", "state"))
    schema.add_foreign_key(ForeignKey("train", "store", "store_id", "store_id"))

    return schema


def _generate_rossmann_proxy(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    N_STORES = 200
    N_TRAIN = 5000

    states = [f"ST_{i}" for i in range(1, 17)]
    store_states = pd.DataFrame({
        "state": states,
        "state_name": [f"State_{i}" for i in range(1, 17)],
    })

    stores = pd.DataFrame({
        "store_id": np.arange(1, N_STORES + 1),
        "store_type": rng.choice(["a", "b", "c", "d"], N_STORES),
        "assortment": rng.choice(["a", "b", "c"], N_STORES),
        "competition_distance": rng.uniform(100, 50000, N_STORES).round(0),
        "promo2": rng.choice([0, 1], N_STORES),
        "state": rng.choice(states, N_STORES),
    })

    train = pd.DataFrame({
        "store_id": rng.choice(stores["store_id"].values, N_TRAIN),
        "day_of_week": rng.integers(1, 8, N_TRAIN),
        "sales": rng.integers(0, 30000, N_TRAIN),
        "customers": rng.integers(0, 3000, N_TRAIN),
        "open": rng.choice([0, 1], N_TRAIN, p=[0.15, 0.85]),
        "promo": rng.choice([0, 1], N_TRAIN, p=[0.4, 0.6]),
    })

    return {"store_states": store_states, "store": stores, "train": train}


# ======================================================================
# IMDB Movie Database  (5 tables)
# ======================================================================

def load_imdb(data_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Load or generate an IMDB-like movie database (5 tables).

    Tables: genre, director, movie (FK→genre, director), actor, cast (FK→movie, actor).
    """
    root = (data_dir or DATA_DIR) / "imdb"

    table_files = {
        "genre": root / "genre.csv",
        "director": root / "director.csv",
        "movie": root / "movie.csv",
        "actor": root / "actor.csv",
        "cast": root / "cast.csv",
    }

    if all(f.exists() for f in table_files.values()):
        logger.info("Loading IMDB from %s", root)
        data = {name: pd.read_csv(path) for name, path in table_files.items()}
    else:
        logger.info("Generating synthetic IMDB proxy (no CSV found at %s)", root)
        data = _generate_imdb_proxy()

    schema = _imdb_schema()
    data = _align_columns(data, schema)
    return data, schema


def _imdb_schema() -> RelationalSchema:
    schema = RelationalSchema()

    schema.add_table(Table(
        name="genre", primary_key="genre_id",
        attributes=["genre_id", "name"],
        dtypes={"genre_id": "categorical", "name": "categorical"},
    ))
    schema.add_table(Table(
        name="director", primary_key="director_id",
        attributes=["director_id", "name", "nationality"],
        dtypes={"director_id": "categorical", "name": "categorical",
                "nationality": "categorical"},
    ))
    schema.add_table(Table(
        name="movie", primary_key="movie_id",
        attributes=["movie_id", "title", "year", "rating", "budget",
                     "genre_id", "director_id"],
        dtypes={"movie_id": "categorical", "title": "categorical",
                "year": "numerical", "rating": "numerical",
                "budget": "numerical", "genre_id": "categorical",
                "director_id": "categorical"},
    ))
    schema.add_table(Table(
        name="actor", primary_key="actor_id",
        attributes=["actor_id", "name", "gender", "birth_year"],
        dtypes={"actor_id": "categorical", "name": "categorical",
                "gender": "categorical", "birth_year": "numerical"},
    ))
    schema.add_table(Table(
        name="cast", primary_key="cast_id",
        attributes=["cast_id", "movie_id", "actor_id", "role_type"],
        dtypes={"cast_id": "categorical", "movie_id": "categorical",
                "actor_id": "categorical", "role_type": "categorical"},
    ))

    schema.add_foreign_key(ForeignKey("movie", "genre", "genre_id", "genre_id"))
    schema.add_foreign_key(ForeignKey("movie", "director", "director_id", "director_id"))
    schema.add_foreign_key(ForeignKey("cast", "movie", "movie_id", "movie_id"))
    schema.add_foreign_key(ForeignKey("cast", "actor", "actor_id", "actor_id"))

    return schema


def _generate_imdb_proxy(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate a small realistic proxy of an IMDB-like movie database."""
    rng = np.random.default_rng(seed)
    N_GENRES, N_DIRECTORS, N_MOVIES = 20, 300, 1000
    N_ACTORS, N_CAST = 500, 3000

    genre_names = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
        "Music", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller",
        "War", "Western",
    ]
    nationalities = [
        "American", "British", "French", "German", "Italian", "Japanese",
        "Korean", "Indian", "Canadian", "Australian", "Spanish", "Mexican",
    ]

    genres = pd.DataFrame({
        "genre_id": np.arange(1, N_GENRES + 1),
        "name": genre_names[:N_GENRES],
    })

    directors = pd.DataFrame({
        "director_id": np.arange(1, N_DIRECTORS + 1),
        "name": [f"Director_{i}" for i in range(1, N_DIRECTORS + 1)],
        "nationality": rng.choice(nationalities, N_DIRECTORS),
    })

    movies = pd.DataFrame({
        "movie_id": np.arange(1, N_MOVIES + 1),
        "title": [f"Movie_{i}" for i in range(1, N_MOVIES + 1)],
        "year": rng.integers(1970, 2025, N_MOVIES),
        "rating": rng.uniform(1.0, 10.0, N_MOVIES).round(1),
        "budget": (rng.uniform(0.5, 300, N_MOVIES) * 1e6).round(0),
        "genre_id": rng.choice(genres["genre_id"].values, N_MOVIES),
        "director_id": rng.choice(directors["director_id"].values, N_MOVIES),
    })

    actors = pd.DataFrame({
        "actor_id": np.arange(1, N_ACTORS + 1),
        "name": [f"Actor_{i}" for i in range(1, N_ACTORS + 1)],
        "gender": rng.choice(["M", "F"], N_ACTORS),
        "birth_year": rng.integers(1940, 2000, N_ACTORS),
    })

    role_types = ["lead", "supporting", "cameo", "voice", "extra"]
    casts = pd.DataFrame({
        "cast_id": np.arange(1, N_CAST + 1),
        "movie_id": rng.choice(movies["movie_id"].values, N_CAST),
        "actor_id": rng.choice(actors["actor_id"].values, N_CAST),
        "role_type": rng.choice(role_types, N_CAST, p=[0.2, 0.35, 0.15, 0.1, 0.2]),
    })

    return {
        "genre": genres, "director": directors, "movie": movies,
        "actor": actors, "cast": casts,
    }


# ======================================================================
# TPC-H Benchmark (simplified, 5 tables)
# ======================================================================

def load_tpch(data_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Load or generate a simplified TPC-H dataset (5 tables).

    Tables: region, nation (FK→region), supplier (FK→nation), part,
            partsupp (FK→part, supplier).
    """
    root = (data_dir or DATA_DIR) / "tpch"

    table_files = {
        "region": root / "region.csv",
        "nation": root / "nation.csv",
        "supplier": root / "supplier.csv",
        "part": root / "part.csv",
        "partsupp": root / "partsupp.csv",
    }

    if all(f.exists() for f in table_files.values()):
        logger.info("Loading TPC-H from %s", root)
        data = {name: pd.read_csv(path) for name, path in table_files.items()}
    else:
        logger.info("Generating synthetic TPC-H proxy (no CSV found at %s)", root)
        data = _generate_tpch_proxy()

    schema = _tpch_schema()
    data = _align_columns(data, schema)
    return data, schema


def _tpch_schema() -> RelationalSchema:
    schema = RelationalSchema()

    schema.add_table(Table(
        name="region", primary_key="region_id",
        attributes=["region_id", "name", "comment"],
        dtypes={"region_id": "categorical", "name": "categorical",
                "comment": "categorical"},
    ))
    schema.add_table(Table(
        name="nation", primary_key="nation_id",
        attributes=["nation_id", "name", "region_id", "comment"],
        dtypes={"nation_id": "categorical", "name": "categorical",
                "region_id": "categorical", "comment": "categorical"},
    ))
    schema.add_table(Table(
        name="supplier", primary_key="supplier_id",
        attributes=["supplier_id", "name", "address", "nation_id",
                     "phone", "acctbal"],
        dtypes={"supplier_id": "categorical", "name": "categorical",
                "address": "categorical", "nation_id": "categorical",
                "phone": "categorical", "acctbal": "numerical"},
    ))
    schema.add_table(Table(
        name="part", primary_key="part_id",
        attributes=["part_id", "name", "mfgr", "brand", "type",
                     "size", "retail_price"],
        dtypes={"part_id": "categorical", "name": "categorical",
                "mfgr": "categorical", "brand": "categorical",
                "type": "categorical", "size": "numerical",
                "retail_price": "numerical"},
    ))
    schema.add_table(Table(
        name="partsupp", primary_key=None,
        attributes=["part_id", "supplier_id", "availqty", "supplycost"],
        dtypes={"part_id": "categorical", "supplier_id": "categorical",
                "availqty": "numerical", "supplycost": "numerical"},
    ))

    schema.add_foreign_key(ForeignKey("nation", "region", "region_id", "region_id"))
    schema.add_foreign_key(ForeignKey("supplier", "nation", "nation_id", "nation_id"))
    schema.add_foreign_key(ForeignKey("partsupp", "part", "part_id", "part_id"))
    schema.add_foreign_key(ForeignKey("partsupp", "supplier", "supplier_id", "supplier_id"))

    return schema


def _generate_tpch_proxy(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate a small proxy of the TPC-H benchmark dataset."""
    rng = np.random.default_rng(seed)
    N_REGIONS, N_NATIONS, N_SUPPLIERS = 5, 25, 50
    N_PARTS, N_PARTSUPP = 100, 400

    region_names = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
    regions = pd.DataFrame({
        "region_id": np.arange(1, N_REGIONS + 1),
        "name": region_names,
        "comment": [f"Region comment {i}" for i in range(1, N_REGIONS + 1)],
    })

    nation_names = [
        "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA",
        "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ",
        "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU",
        "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA",
        "UNITED KINGDOM", "UNITED STATES",
    ]
    nations = pd.DataFrame({
        "nation_id": np.arange(1, N_NATIONS + 1),
        "name": nation_names[:N_NATIONS],
        "region_id": rng.choice(regions["region_id"].values, N_NATIONS),
        "comment": [f"Nation comment {i}" for i in range(1, N_NATIONS + 1)],
    })

    suppliers = pd.DataFrame({
        "supplier_id": np.arange(1, N_SUPPLIERS + 1),
        "name": [f"Supplier#{i:04d}" for i in range(1, N_SUPPLIERS + 1)],
        "address": [f"Addr_{i}" for i in range(1, N_SUPPLIERS + 1)],
        "nation_id": rng.choice(nations["nation_id"].values, N_SUPPLIERS),
        "phone": [f"{rng.integers(10,99)}-{rng.integers(100,999)}-{rng.integers(100,999)}-{rng.integers(1000,9999)}" for _ in range(N_SUPPLIERS)],
        "acctbal": rng.uniform(-1000, 10000, N_SUPPLIERS).round(2),
    })

    mfgrs = [f"Manufacturer#{i}" for i in range(1, 6)]
    brands = [f"Brand#{i}{j}" for i in range(1, 6) for j in range(1, 6)]
    part_types = ["ECONOMY ANODIZED STEEL", "PROMO BURNISHED COPPER",
                  "STANDARD POLISHED TIN", "LARGE BRUSHED BRASS",
                  "MEDIUM PLATED NICKEL", "SMALL BURNISHED STEEL"]
    parts = pd.DataFrame({
        "part_id": np.arange(1, N_PARTS + 1),
        "name": [f"Part_{i}" for i in range(1, N_PARTS + 1)],
        "mfgr": rng.choice(mfgrs, N_PARTS),
        "brand": rng.choice(brands, N_PARTS),
        "type": rng.choice(part_types, N_PARTS),
        "size": rng.integers(1, 50, N_PARTS),
        "retail_price": rng.uniform(900, 2100, N_PARTS).round(2),
    })

    partsupps = pd.DataFrame({
        "part_id": rng.choice(parts["part_id"].values, N_PARTSUPP),
        "supplier_id": rng.choice(suppliers["supplier_id"].values, N_PARTSUPP),
        "availqty": rng.integers(1, 10000, N_PARTSUPP),
        "supplycost": rng.uniform(1, 1000, N_PARTSUPP).round(2),
    })

    return {
        "region": regions, "nation": nations, "supplier": suppliers,
        "part": parts, "partsupp": partsupps,
    }


# ======================================================================
# University Database  (5 tables)
# ======================================================================

def load_university(data_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Load or generate a university database (5 tables).

    Tables: department, instructor (FK→department), course (FK→department),
            student (FK→department), enrollment (FK→student, course).
    """
    root = (data_dir or DATA_DIR) / "university"

    table_files = {
        "department": root / "department.csv",
        "instructor": root / "instructor.csv",
        "course": root / "course.csv",
        "student": root / "student.csv",
        "enrollment": root / "enrollment.csv",
    }

    if all(f.exists() for f in table_files.values()):
        logger.info("Loading University from %s", root)
        data = {name: pd.read_csv(path) for name, path in table_files.items()}
    else:
        logger.info("Generating synthetic University proxy (no CSV found at %s)", root)
        data = _generate_university_proxy()

    schema = _university_schema()
    data = _align_columns(data, schema)
    return data, schema


def _university_schema() -> RelationalSchema:
    schema = RelationalSchema()

    schema.add_table(Table(
        name="department", primary_key="dept_id",
        attributes=["dept_id", "name", "building", "budget"],
        dtypes={"dept_id": "categorical", "name": "categorical",
                "building": "categorical", "budget": "numerical"},
    ))
    schema.add_table(Table(
        name="instructor", primary_key="instructor_id",
        attributes=["instructor_id", "name", "dept_id", "salary", "rank"],
        dtypes={"instructor_id": "categorical", "name": "categorical",
                "dept_id": "categorical", "salary": "numerical",
                "rank": "categorical"},
    ))
    schema.add_table(Table(
        name="course", primary_key="course_id",
        attributes=["course_id", "title", "dept_id", "credits"],
        dtypes={"course_id": "categorical", "title": "categorical",
                "dept_id": "categorical", "credits": "numerical"},
    ))
    schema.add_table(Table(
        name="student", primary_key="student_id",
        attributes=["student_id", "name", "dept_id", "gpa", "year_level"],
        dtypes={"student_id": "categorical", "name": "categorical",
                "dept_id": "categorical", "gpa": "numerical",
                "year_level": "categorical"},
    ))
    schema.add_table(Table(
        name="enrollment", primary_key=None,
        attributes=["student_id", "course_id", "semester", "grade"],
        dtypes={"student_id": "categorical", "course_id": "categorical",
                "semester": "categorical", "grade": "categorical"},
    ))

    schema.add_foreign_key(ForeignKey("instructor", "department", "dept_id", "dept_id"))
    schema.add_foreign_key(ForeignKey("course", "department", "dept_id", "dept_id"))
    schema.add_foreign_key(ForeignKey("student", "department", "dept_id", "dept_id"))
    schema.add_foreign_key(ForeignKey("enrollment", "student", "student_id", "student_id"))
    schema.add_foreign_key(ForeignKey("enrollment", "course", "course_id", "course_id"))

    return schema


def _generate_university_proxy(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate a small realistic proxy of a university database."""
    rng = np.random.default_rng(seed)
    N_DEPTS, N_INSTRUCTORS, N_COURSES = 15, 80, 50
    N_STUDENTS, N_ENROLLMENTS = 200, 800

    dept_names = [
        "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
        "Economics", "History", "English", "Philosophy", "Psychology",
        "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
        "Statistics", "Music",
    ]
    buildings = ["Watson", "Taylor", "Painter", "Packard", "Mudd",
                 "Gates", "Sloan", "Green", "Baker", "Jordan"]
    ranks = ["Assistant Professor", "Associate Professor", "Professor", "Lecturer"]

    departments = pd.DataFrame({
        "dept_id": np.arange(1, N_DEPTS + 1),
        "name": dept_names[:N_DEPTS],
        "building": rng.choice(buildings, N_DEPTS),
        "budget": rng.uniform(50000, 1500000, N_DEPTS).round(0),
    })

    instructors = pd.DataFrame({
        "instructor_id": np.arange(1, N_INSTRUCTORS + 1),
        "name": [f"Prof_{i}" for i in range(1, N_INSTRUCTORS + 1)],
        "dept_id": rng.choice(departments["dept_id"].values, N_INSTRUCTORS),
        "salary": rng.uniform(50000, 200000, N_INSTRUCTORS).round(0),
        "rank": rng.choice(ranks, N_INSTRUCTORS, p=[0.3, 0.3, 0.25, 0.15]),
    })

    courses = pd.DataFrame({
        "course_id": np.arange(1, N_COURSES + 1),
        "title": [f"Course_{i}" for i in range(1, N_COURSES + 1)],
        "dept_id": rng.choice(departments["dept_id"].values, N_COURSES),
        "credits": rng.choice([1, 2, 3, 4], N_COURSES, p=[0.05, 0.15, 0.55, 0.25]),
    })

    students = pd.DataFrame({
        "student_id": np.arange(1, N_STUDENTS + 1),
        "name": [f"Student_{i}" for i in range(1, N_STUDENTS + 1)],
        "dept_id": rng.choice(departments["dept_id"].values, N_STUDENTS),
        "gpa": rng.uniform(1.5, 4.0, N_STUDENTS).round(2),
        "year_level": rng.choice(["Freshman", "Sophomore", "Junior", "Senior"],
                                 N_STUDENTS, p=[0.3, 0.28, 0.22, 0.2]),
    })

    semesters = ["2023_Fall", "2024_Spring", "2024_Fall", "2025_Spring"]
    grades = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
    grade_p = [0.05, 0.12, 0.10, 0.12, 0.15, 0.10, 0.10, 0.10, 0.06, 0.05, 0.05]
    enrollments = pd.DataFrame({
        "student_id": rng.choice(students["student_id"].values, N_ENROLLMENTS),
        "course_id": rng.choice(courses["course_id"].values, N_ENROLLMENTS),
        "semester": rng.choice(semesters, N_ENROLLMENTS),
        "grade": rng.choice(grades, N_ENROLLMENTS, p=grade_p),
    })

    return {
        "department": departments, "instructor": instructors,
        "course": courses, "student": students, "enrollment": enrollments,
    }


# ======================================================================
# Walmart Retail Sales  (4 tables)
# ======================================================================

def load_walmart(data_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Load or generate a Walmart-like retail dataset (4 tables).

    Tables: region, store (FK→region), item, sales (FK→store, item).
    """
    root = (data_dir or DATA_DIR) / "walmart"

    table_files = {
        "region": root / "region.csv",
        "store": root / "store.csv",
        "item": root / "item.csv",
        "sales": root / "sales.csv",
    }

    if all(f.exists() for f in table_files.values()):
        logger.info("Loading Walmart from %s", root)
        data = {name: pd.read_csv(path) for name, path in table_files.items()}
    else:
        logger.info("Generating synthetic Walmart proxy (no CSV found at %s)", root)
        data = _generate_walmart_proxy()

    schema = _walmart_schema()
    data = _align_columns(data, schema)
    return data, schema


def _walmart_schema() -> RelationalSchema:
    schema = RelationalSchema()

    schema.add_table(Table(
        name="region", primary_key="region_id",
        attributes=["region_id", "name", "country"],
        dtypes={"region_id": "categorical", "name": "categorical",
                "country": "categorical"},
    ))
    schema.add_table(Table(
        name="store", primary_key="store_id",
        attributes=["store_id", "region_id", "store_type", "area"],
        dtypes={"store_id": "categorical", "region_id": "categorical",
                "store_type": "categorical", "area": "numerical"},
    ))
    schema.add_table(Table(
        name="item", primary_key="item_id",
        attributes=["item_id", "item_name", "category", "price"],
        dtypes={"item_id": "categorical", "item_name": "categorical",
                "category": "categorical", "price": "numerical"},
    ))
    schema.add_table(Table(
        name="sales", primary_key=None,
        attributes=["store_id", "item_id", "date", "units_sold", "revenue"],
        dtypes={"store_id": "categorical", "item_id": "categorical",
                "date": "numerical", "units_sold": "numerical",
                "revenue": "numerical"},
    ))

    schema.add_foreign_key(ForeignKey("store", "region", "region_id", "region_id"))
    schema.add_foreign_key(ForeignKey("sales", "store", "store_id", "store_id"))
    schema.add_foreign_key(ForeignKey("sales", "item", "item_id", "item_id"))

    return schema


def _generate_walmart_proxy(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate a small realistic proxy of a Walmart-like retail dataset."""
    rng = np.random.default_rng(seed)
    N_REGIONS, N_STORES, N_ITEMS = 10, 100, 500
    N_SALES = 5000

    region_names = ["Northeast", "Southeast", "Midwest", "Southwest",
                    "West Coast", "Mountain", "Great Plains", "Mid-Atlantic",
                    "Pacific Northwest", "Gulf Coast"]
    regions = pd.DataFrame({
        "region_id": np.arange(1, N_REGIONS + 1),
        "name": region_names[:N_REGIONS],
        "country": ["US"] * N_REGIONS,
    })

    store_types = ["Supercenter", "Neighborhood Market", "Sam's Club", "Express"]
    stores = pd.DataFrame({
        "store_id": np.arange(1, N_STORES + 1),
        "region_id": rng.choice(regions["region_id"].values, N_STORES),
        "store_type": rng.choice(store_types, N_STORES, p=[0.5, 0.25, 0.15, 0.1]),
        "area": rng.uniform(5000, 250000, N_STORES).round(0),
    })

    categories = ["Grocery", "Electronics", "Clothing", "Home & Garden",
                  "Pharmacy", "Toys", "Automotive", "Sports", "Beauty",
                  "Office Supplies"]
    items = pd.DataFrame({
        "item_id": np.arange(1, N_ITEMS + 1),
        "item_name": [f"Item_{i}" for i in range(1, N_ITEMS + 1)],
        "category": rng.choice(categories, N_ITEMS),
        "price": rng.uniform(0.5, 500, N_ITEMS).round(2),
    })

    units = rng.integers(1, 50, N_SALES)
    chosen_items = rng.choice(items["item_id"].values, N_SALES)
    item_prices = items.set_index("item_id")["price"].reindex(chosen_items).values
    sales = pd.DataFrame({
        "store_id": rng.choice(stores["store_id"].values, N_SALES),
        "item_id": chosen_items,
        "date": rng.integers(20230101, 20251231, N_SALES),
        "units_sold": units,
        "revenue": (units * item_prices).round(2),
    })

    return {
        "region": regions, "store": stores, "item": items, "sales": sales,
    }


# ======================================================================
# Synthetic star schema generator
# ======================================================================

def generate_star_schema(
    num_dimension_tables: int = 4,
    dim_rows: int = 100,
    fact_rows: int = 5000,
    num_dim_attrs: int = 3,
    num_fact_measures: int = 2,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Generate a configurable synthetic star schema for scalability testing.

    Creates one fact table referencing ``num_dimension_tables`` dimension tables.
    """
    rng = np.random.default_rng(seed)
    schema = RelationalSchema()
    data: dict[str, pd.DataFrame] = {}

    dim_names: list[str] = []
    dim_pk_cols: list[str] = []

    for d in range(num_dimension_tables):
        tname = f"dim_{d}"
        pk_col = f"dim_{d}_id"
        attrs = [pk_col] + [f"dim_{d}_attr_{a}" for a in range(num_dim_attrs)]
        dtypes = {pk_col: "categorical"}
        for a in range(num_dim_attrs):
            col = f"dim_{d}_attr_{a}"
            dtypes[col] = rng.choice(["categorical", "numerical"])

        schema.add_table(Table(name=tname, attributes=attrs, dtypes=dtypes, primary_key=pk_col))

        rows: dict[str, Any] = {pk_col: np.arange(1, dim_rows + 1)}
        for a in range(num_dim_attrs):
            col = f"dim_{d}_attr_{a}"
            if dtypes[col] == "numerical":
                rows[col] = rng.uniform(0, 1000, dim_rows).round(2)
            else:
                rows[col] = rng.choice([f"cat_{v}" for v in range(10)], dim_rows)
        data[tname] = pd.DataFrame(rows)

        dim_names.append(tname)
        dim_pk_cols.append(pk_col)

    fact_attrs = list(dim_pk_cols) + [f"measure_{m}" for m in range(num_fact_measures)]
    fact_dtypes: dict[str, str] = {pk: "categorical" for pk in dim_pk_cols}
    for m in range(num_fact_measures):
        fact_dtypes[f"measure_{m}"] = "numerical"

    schema.add_table(Table(name="fact", attributes=fact_attrs, dtypes=fact_dtypes, primary_key=None))

    fact_rows_dict: dict[str, Any] = {}
    for d, pk_col in enumerate(dim_pk_cols):
        fact_rows_dict[pk_col] = rng.choice(data[dim_names[d]][pk_col].values, fact_rows)
        schema.add_foreign_key(ForeignKey("fact", dim_names[d], pk_col, pk_col))

    for m in range(num_fact_measures):
        fact_rows_dict[f"measure_{m}"] = rng.uniform(0, 10000, fact_rows).round(2)

    data["fact"] = pd.DataFrame(fact_rows_dict)

    return data, schema


# ======================================================================
# Dispatcher
# ======================================================================

def load_dataset(
    name: str, data_dir: Path | None = None, **kwargs: Any
) -> tuple[dict[str, pd.DataFrame], RelationalSchema]:
    """Load a named dataset.

    Supported names: berka, rossmann, imdb, tpch, university, walmart,
                     synthetic_star.
    """
    _LOADERS: dict[str, Any] = {
        "berka": load_berka,
        "rossmann": load_rossmann,
        "imdb": load_imdb,
        "tpch": load_tpch,
        "university": load_university,
        "walmart": load_walmart,
    }

    if name == "synthetic_star":
        return generate_star_schema(**kwargs)
    if name in _LOADERS:
        return _LOADERS[name](data_dir)
    raise ValueError(
        f"Unknown dataset: {name!r}. Choose from: "
        f"{', '.join(list(_LOADERS) + ['synthetic_star'])}"
    )


# ======================================================================
# Helpers
# ======================================================================

def _align_columns(
    data: dict[str, pd.DataFrame], schema: RelationalSchema
) -> dict[str, pd.DataFrame]:
    """Ensure each dataframe only has columns declared in the schema."""
    aligned: dict[str, pd.DataFrame] = {}
    for tname, table in schema.tables.items():
        df = data[tname]
        cols = [c for c in table.attributes if c in df.columns]
        aligned[tname] = df[cols].copy()
    return aligned
