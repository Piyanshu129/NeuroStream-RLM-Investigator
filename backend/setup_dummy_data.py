"""
setup_dummy_data.py — Initializes SQLite, FAISS, and local text files
with overlapping medical-supply-chain data.

Run once:  python -m backend.setup_dummy_data
"""

import os, sqlite3, json, pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "medical_supply.db")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
LOGS_PATH = os.path.join(DATA_DIR, "shipping_logs.txt")

# ── Suppliers (shared across all sources) ──────────────────────────────
SUPPLIERS = [
    (1, "MedCore Supplies",    72, "East"),
    (2, "HealthFirst Logistics", 91, "West"),
    (3, "BioRapid Inc.",        65, "South"),
    (4, "CareLink Distribution", 88, "North"),
    (5, "VitalEdge Partners",   54, "East"),
]

SHIPMENTS = [
    (101, 1, "Surgical Gloves",      "DELAYED",    4500,  "2025-11-02"),
    (102, 1, "N95 Masks",            "DELIVERED",  2200,  "2025-11-05"),
    (103, 2, "IV Drip Kits",         "DELIVERED",  8700,  "2025-10-28"),
    (104, 2, "Bandage Rolls",        "DELIVERED",  1300,  "2025-11-10"),
    (105, 3, "Syringes 10mL",        "DELAYED",    3900,  "2025-11-01"),
    (106, 3, "Sterilization Fluid",  "DELAYED",    6100,  "2025-11-08"),
    (107, 3, "Thermometers",         "CANCELLED",  2800,  "2025-11-15"),
    (108, 4, "Oxygen Regulators",    "DELIVERED",  12500, "2025-10-25"),
    (109, 4, "Pulse Oximeters",      "DELIVERED",  9400,  "2025-11-03"),
    (110, 5, "Wheelchairs",          "DELAYED",    15600, "2025-10-30"),
    (111, 5, "Crutches",             "DELAYED",    4200,  "2025-11-12"),
    (112, 5, "Bed Pans",             "CANCELLED",  1800,  "2025-11-18"),
    (113, 1, "Latex Gloves",         "DELIVERED",  3100,  "2025-11-20"),
    (114, 3, "Scalpel Blades",       "DELAYED",    5600,  "2025-11-22"),
    (115, 4, "Defibrillator Pads",   "DELIVERED",  7800,  "2025-11-25"),
]

# ── Supplier Agreement Documents (for FAISS) ──────────────────────────
AGREEMENT_DOCS = [
    {
        "id": "agreement_001",
        "supplier": "MedCore Supplies",
        "text": (
            "SUPPLIER AGREEMENT — MedCore Supplies (ID: 001)\n"
            "Section 4.2 — Late Delivery Penalties:\n"
            "If delivery is delayed by more than 5 business days, a penalty of 10% "
            "of the shipment cost will be applied. Delays exceeding 15 days incur "
            "an additional 5% surcharge.\n"
            "Section 6.1 — Quality Standards:\n"
            "All surgical supplies must meet ISO 13485 certification. Batch "
            "failure rate must not exceed 0.5%."
        ),
    },
    {
        "id": "agreement_002",
        "supplier": "HealthFirst Logistics",
        "text": (
            "SUPPLIER AGREEMENT — HealthFirst Logistics (ID: 002)\n"
            "Section 4.2 — Late Delivery Penalties:\n"
            "Delays beyond 3 business days result in a 7% penalty per shipment. "
            "HealthFirst maintains a 98% on-time guarantee.\n"
            "Section 6.1 — Quality Standards:\n"
            "IV and fluid-delivery products require FDA 510(k) clearance. "
            "Quarterly audits are mandatory."
        ),
    },
    {
        "id": "agreement_003",
        "supplier": "BioRapid Inc.",
        "text": (
            "SUPPLIER AGREEMENT — BioRapid Inc. (ID: 003)\n"
            "Section 4.2 — Late Delivery Penalties:\n"
            "Any delay beyond 7 calendar days triggers an automatic 12% penalty. "
            "Three consecutive delayed shipments allow contract termination.\n"
            "Section 6.1 — Quality Standards:\n"
            "Sterilization products must pass EPA registration. Syringes require "
            "lot-level traceability."
        ),
    },
    {
        "id": "agreement_004",
        "supplier": "CareLink Distribution",
        "text": (
            "SUPPLIER AGREEMENT — CareLink Distribution (ID: 004)\n"
            "Section 4.2 — Late Delivery Penalties:\n"
            "CareLink offers a 48-hour express guarantee. Penalties of 5% apply "
            "only if delay exceeds 10 business days.\n"
            "Section 6.1 — Quality Standards:\n"
            "Medical devices must carry CE marking. Annual compliance reports "
            "are required."
        ),
    },
    {
        "id": "agreement_005",
        "supplier": "VitalEdge Partners",
        "text": (
            "SUPPLIER AGREEMENT — VitalEdge Partners (ID: 005)\n"
            "Section 4.2 — Late Delivery Penalties:\n"
            "Delays beyond 4 business days incur a 15% penalty. VitalEdge has "
            "the highest penalty rate due to historically poor on-time performance.\n"
            "Section 6.1 — Quality Standards:\n"
            "Mobility aids must meet ADA compliance standards. Wheelchairs "
            "require crash-test certification."
        ),
    },
]

# ── Shipping Logs (local text file) ───────────────────────────────────
SHIPPING_LOGS = """\
[2025-10-25 08:12] Supplier 004 — Shipment #108 (Oxygen Regulators) — ON TIME — delivered in 2 days
[2025-10-28 09:45] Supplier 002 — Shipment #103 (IV Drip Kits) — ON TIME — delivered in 1 day
[2025-10-30 14:30] Supplier 005 — Shipment #110 (Wheelchairs) — DELAYED by 7 days — arrived 2025-11-06
[2025-11-01 11:00] Supplier 003 — Shipment #105 (Syringes 10mL) — DELAYED by 9 days — arrived 2025-11-10
[2025-11-02 07:22] Supplier 001 — Shipment #101 (Surgical Gloves) — DELAYED by 7 days — arrived 2025-11-09
[2025-11-03 16:50] Supplier 004 — Shipment #109 (Pulse Oximeters) — ON TIME — delivered in 2 days
[2025-11-05 10:15] Supplier 001 — Shipment #102 (N95 Masks) — ON TIME — delivered in 3 days
[2025-11-08 13:40] Supplier 003 — Shipment #106 (Sterilization Fluid) — DELAYED by 11 days — arrived 2025-11-19
[2025-11-10 09:00] Supplier 002 — Shipment #104 (Bandage Rolls) — ON TIME — delivered in 2 days
[2025-11-12 15:20] Supplier 005 — Shipment #111 (Crutches) — DELAYED by 6 days — arrived 2025-11-18
[2025-11-15 08:00] Supplier 003 — Shipment #107 (Thermometers) — CANCELLED — supplier out of stock
[2025-11-18 12:10] Supplier 005 — Shipment #112 (Bed Pans) — CANCELLED — damaged in transit
[2025-11-20 14:45] Supplier 001 — Shipment #113 (Latex Gloves) — ON TIME — delivered in 2 days
[2025-11-22 10:30] Supplier 003 — Shipment #114 (Scalpel Blades) — DELAYED by 8 days — arrived 2025-11-30
[2025-11-25 09:15] Supplier 004 — Shipment #115 (Defibrillator Pads) — ON TIME — delivered in 1 day
"""


def setup_sqlite():
    """Create SQLite database with suppliers and shipments tables."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE suppliers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE,
            reliability_score INTEGER NOT NULL,
            region TEXT NOT NULL COLLATE NOCASE
        )
    """)
    cur.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?)", SUPPLIERS)

    cur.execute("""
        CREATE TABLE shipments (
            id INTEGER PRIMARY KEY,
            supplier_id INTEGER NOT NULL,
            item TEXT NOT NULL COLLATE NOCASE,
            status TEXT NOT NULL COLLATE NOCASE,
            cost REAL NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
        )
    """)
    cur.executemany("INSERT INTO shipments VALUES (?, ?, ?, ?, ?, ?)", SHIPMENTS)

    conn.commit()
    conn.close()
    print(f"[✓] SQLite database created at {DB_PATH}")


def setup_faiss():
    """Create FAISS index from supplier agreement documents."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(FAISS_DIR, exist_ok=True)
    import faiss

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc["text"] for doc in AGREEMENT_DOCS]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product on normalized = cosine
    index.add(embeddings)

    faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))

    # Save metadata alongside
    metadata = [{"id": d["id"], "supplier": d["supplier"], "text": d["text"]} for d in AGREEMENT_DOCS]
    with open(os.path.join(FAISS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] FAISS index created at {FAISS_DIR}  ({len(texts)} documents)")


def setup_logs():
    """Write shipping logs text file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOGS_PATH, "w") as f:
        f.write(SHIPPING_LOGS.strip())
    print(f"[✓] Shipping logs written to {LOGS_PATH}")


def main():
    print("=== Setting up dummy data for RLM Data Investigator ===\n")
    setup_sqlite()
    setup_faiss()
    setup_logs()
    print("\n=== All data sources initialized successfully ===")


if __name__ == "__main__":
    main()
