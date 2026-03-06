"""
sandbox.py — Secure Python REPL sandbox for the RLM agent.

Exposes sql_query(), vector_search(), read_file() inside a restricted
exec() namespace.  Errors are captured and returned (not swallowed)
so the LLM can self-correct.
"""

import io, os, sys, json, sqlite3, traceback
from typing import Any, Callable, Optional
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "medical_supply.db")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
LOGS_PATH = os.path.join(DATA_DIR, "shipping_logs.txt")


class LogEmitter:
    """Collects execution log entries and optionally forwards them via callback."""

    def __init__(self, callback: Optional[Callable] = None):
        self.entries: list[dict] = []
        self._callback = callback

    def emit(self, entry_type: str, content: str, meta: dict | None = None):
        entry = {"type": entry_type, "content": content, "meta": meta or {}}
        self.entries.append(entry)
        if self._callback:
            self._callback(entry)


class GraphEmitter:
    """Emits graph nodes/edges at the exact moment the agent touches data."""

    def __init__(self, callback: Optional[Callable] = None):
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._callback = callback
        self._node_ids: set = set()

    def add_node(self, node_id: str, label: str, source: str, data: dict | None = None):
        if node_id not in self._node_ids:
            self._node_ids.add(node_id)
            node = {"id": node_id, "label": label, "source": source, "data": data or {}}
            self.nodes.append(node)
            if self._callback:
                self._callback({"action": "add_node", "node": node})

    def add_edge(self, source_id: str, target_id: str, label: str = ""):
        edge = {"source": source_id, "target": target_id, "label": label}
        self.edges.append(edge)
        if self._callback:
            self._callback({"action": "add_edge", "edge": edge})


# ── Sandbox helper functions (injected into exec namespace) ───────────

def _make_sql_query(log: LogEmitter, graph: GraphEmitter):
    """Factory for sql_query() with live logging + graph emission."""

    def sql_query(query: str) -> list[dict]:
        log.emit("code", f"sql_query(\"{query}\")")
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA case_sensitive_like = OFF")
            conn.create_function("UPPER", 1, lambda s: s.upper() if s else s)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()

            # Emit each row as a graph node at the moment of access
            for row in rows:
                row_id = f"sql_{row.get('id', id(row))}"
                label = str(row.get("name", row.get("item", row.get("id", "?"))))
                graph.add_node(row_id, label, "sql", row)

            log.emit("output", json.dumps(rows, indent=2, default=str))
            return rows
        except Exception as e:
            err = traceback.format_exc()
            log.emit("error", err)
            return {"error": str(e), "traceback": err}

    return sql_query


def _make_vector_search(log: LogEmitter, graph: GraphEmitter):
    """Factory for vector_search() with live logging + graph emission."""

    def vector_search(query: str, n: int = 3) -> list[dict]:
        log.emit("code", f"vector_search(\"{query}\", n={n})")
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_emb)

            index = faiss.read_index(os.path.join(FAISS_DIR, "index.faiss"))
            with open(os.path.join(FAISS_DIR, "metadata.json")) as f:
                metadata = json.load(f)

            scores, indices = index.search(q_emb, min(n, len(metadata)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = metadata[idx]
                doc_result = {**doc, "score": float(score)}
                results.append(doc_result)

                # Emit graph node the instant we retrieve it
                graph.add_node(f"vec_{doc['id']}", doc["supplier"], "vector", doc_result)

            log.emit("output", json.dumps(results, indent=2, default=str))
            return results
        except Exception as e:
            err = traceback.format_exc()
            log.emit("error", err)
            return {"error": str(e), "traceback": err}

    return vector_search


def _make_read_file(log: LogEmitter, graph: GraphEmitter):
    """Factory for read_file() with live logging + graph emission."""

    def read_file(path: str = "") -> str:
        resolved = LOGS_PATH if not path else os.path.join(DATA_DIR, path)
        log.emit("code", f"read_file(\"{path or 'shipping_logs.txt'}\")")
        try:
            with open(resolved, "r") as f:
                content = f.read()

            # Emit as a single file node
            graph.add_node(
                f"file_{os.path.basename(resolved)}",
                os.path.basename(resolved),
                "file",
                {"path": resolved, "lines": content.count("\n") + 1},
            )

            log.emit("output", content[:2000] + ("..." if len(content) > 2000 else ""))
            return content
        except Exception as e:
            err = traceback.format_exc()
            log.emit("error", err)
            return f"ERROR: {e}"

    return read_file


# ── Sandbox Executor ──────────────────────────────────────────────────

SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "bool": bool,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "isinstance": isinstance,
    "type": type,
    "True": True,
    "False": False,
    "None": None,
}


class Sandbox:
    """
    Restricted Python execution environment.

    Captures stdout, returns structured results including any errors
    so the LLM can self-correct.
    """

    def __init__(
        self,
        log_callback: Optional[Callable] = None,
        graph_callback: Optional[Callable] = None,
    ):
        self.log = LogEmitter(callback=log_callback)
        self.graph = GraphEmitter(callback=graph_callback)

        # Build the restricted namespace
        self._namespace: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "sql_query": _make_sql_query(self.log, self.graph),
            "vector_search": _make_vector_search(self.log, self.graph),
            "read_file": _make_read_file(self.log, self.graph),
            "json": json,  # handy for the LLM
        }

    def execute(self, code: str) -> dict:
        """
        Execute Python code in the sandbox.

        Returns:
            {
                "stdout": str,
                "result": Any | None,
                "error": str | None,
                "traceback": str | None,
            }
        """
        self.log.emit("code", code)

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        try:
            sys.stdout = stdout_capture
            exec(code, self._namespace)
            sys.stdout = old_stdout

            stdout_val = stdout_capture.getvalue()
            if stdout_val:
                self.log.emit("output", stdout_val)

            return {
                "stdout": stdout_val,
                "result": self._namespace.get("_result_"),
                "error": None,
                "traceback": None,
            }
        except Exception as e:
            sys.stdout = old_stdout
            tb = traceback.format_exc()
            self.log.emit("error", tb)
            # Error is returned so the LLM can self-correct
            return {
                "stdout": stdout_capture.getvalue(),
                "result": None,
                "error": str(e),
                "traceback": tb,
            }
