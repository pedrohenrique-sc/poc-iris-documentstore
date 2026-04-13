# SPDX-FileCopyrightText: 2026-present Pedro Henrique <pedro@s-c.ai>
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import time
from typing import Any

import iris as iris_driver
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.filters import document_matches_filter

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 3
_RETRY_BACKOFF: list[float] = [0.5, 1.0, 2.0]

# ---------------------------------------------------------------------------
# In-memory BM25 index
# ---------------------------------------------------------------------------

class _BM25Index:
    """
    In-memory Okapi BM25 index built over IRIS documents.

    The index is rebuilt from the current document set on every
    :meth:`IRISDocumentStore.bm25_retrieval` call to reflect the latest data.

    Parameters
    ----------
    k1:
        Term frequency saturation parameter (typical: 1.2–2.0).
    b:
        Length normalization parameter (0.0 = none, 1.0 = full).

    Reference
    ---------
    Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance
    Framework: BM25 and Beyond*. Foundations and Trends in IR, 3(4).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[tuple[str, str]] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._avg_dl: float = 1.0

    def build(self, docs: list[tuple[str, str]]) -> None:
        """Build the index from a list of ``(doc_id, content)`` tuples."""
        self._docs = docs
        self._tf, self._df = [], {}
        total = 0
        for _, content in docs:
            tokens = self._tokenize(content)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in set(tf):
                self._df[t] = self._df.get(t, 0) + 1
            total += len(tokens)
        self._avg_dl = total / len(docs) if docs else 1.0

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        """
        Rank documents by BM25 score.

        Returns
        -------
        List of ``(doc_id, score)`` in descending relevance order.
        Only documents with score > 0 are included.
        """
        if not self._docs:
            return []
        query_tokens = self._tokenize(text)
        N = len(self._docs)
        scores: list[float] = []
        for i, _ in enumerate(self._docs):
            tf_doc = self._tf[i]
            dl = sum(tf_doc.values())
            score = 0.0
            for token in query_tokens:
                tf = tf_doc.get(token, 0)
                df = self._df.get(token, 0)
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                score += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                )
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._docs[i][0], s) for i, s in ranked[:top_k] if s > 0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric tokenizer with accented character support."""
        return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower()) if text else []


# ---------------------------------------------------------------------------
# IRISDocumentStore
# ---------------------------------------------------------------------------

class IRISDocumentStore:
    """
    A DocumentStore backed by `InterSystems IRIS`_.

    Uses IRIS native ``VECTOR(DOUBLE, N)`` column for embedding storage
    and ``VECTOR_COSINE`` function for semantic similarity search.

    .. _InterSystems IRIS: https://www.intersystems.com/products/intersystems-iris/

    **Credentials**

    Pass the connection string and password as Haystack
    :class:`~haystack.utils.Secret` objects.  The recommended approach is to
    export environment variables and use :meth:`~haystack.utils.Secret.from_env_var`:

    .. code-block:: bash

        export IRIS_CONNECTION_STRING="localhost:1972/USER"
        export IRIS_USERNAME="_system"
        export IRIS_PASSWORD="SYS"

    .. code-block:: python

        from haystack_integrations.document_stores.iris import IRISDocumentStore
        store = IRISDocumentStore()

    **Retrievers**

    Use the companion retrievers for embedding-based and keyword-based search:

    .. code-block:: python

        from haystack_integrations.components.retrievers.iris import (
            IRISEmbeddingRetriever,
            IRISBm25Retriever,
        )

    **Table schema** (created automatically if it doesn't exist)

    .. code-block:: sql

        CREATE TABLE SQLUser.<table_name> (
            id        VARCHAR(128)  NOT NULL PRIMARY KEY,
            content   LONGVARCHAR,
            meta      LONGVARCHAR,   -- JSON with sort_keys=True
            score     DOUBLE,
            embedding VECTOR(DOUBLE, <embedding_dim>)
        )

    Parameters
    ----------
    connection_string:
        IRIS DB-API connection string in the format ``host:port/namespace``.
        Resolved from the ``IRIS_CONNECTION_STRING`` environment variable
        by default.
    username:
        IRIS username. Resolved from ``IRIS_USERNAME`` by default.
    password:
        IRIS password. Resolved from ``IRIS_PASSWORD`` by default.
    table_name:
        Name of the SQL table (without schema). The ``SQLUser`` schema is
        prepended automatically.
    embedding_dim:
        Number of dimensions of the embedding vectors.  Must match the
        embedding model used at indexing time.
    bm25_k1:
        BM25 term-frequency saturation parameter (typical: 1.2–2.0).
    bm25_b:
        BM25 length-normalization parameter (0.0–1.0).
    recreate_table:
        Drop and re-create the table on initialization.
        **Use with caution — all data will be lost.**

    Raises
    ------
    ConnectionError
        If unable to connect to IRIS after :data:`_MAX_RETRIES` attempts.
    """

    def __init__(
        self,
        *,
        connection_string: Secret = Secret.from_env_var("IRIS_CONNECTION_STRING"),
        username: Secret = Secret.from_env_var("IRIS_USERNAME"),
        password: Secret = Secret.from_env_var("IRIS_PASSWORD"),
        table_name: str = "HaystackDocuments",
        embedding_dim: int = 384,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        recreate_table: bool = False,
    ) -> None:
        self.connection_string = connection_string
        self.username = username
        self.password = password
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.recreate_table = recreate_table

        self._conn = None
        self._bm25 = _BM25Index(k1=bm25_k1, b=bm25_b)

        self._connect_with_retry()
        if recreate_table:
            self._drop_table()
        self._create_table_if_not_exists()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect_with_retry(self) -> None:
        """
        Connect to IRIS with exponential backoff.

        Attempts up to :data:`_MAX_RETRIES` times with waits of
        0.5 s, 1.0 s, and 2.0 s between attempts.

        Raises
        ------
        ConnectionError
            After all retries are exhausted.
        """
        conn_str = self.connection_string.resolve_value()
        user = self.username.resolve_value()
        pwd = self.password.resolve_value()
        last_exc: Exception | None = None

        for attempt, wait in enumerate(_RETRY_BACKOFF[:_MAX_RETRIES], start=1):
            try:
                self._conn = iris_driver.connect(conn_str, user, pwd)
                logger.info("Connected to IRIS at '%s' (attempt %d).", conn_str, attempt)
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "IRIS connection failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise ConnectionError(
            f"Could not connect to IRIS at '{conn_str}' after "
            f"{_MAX_RETRIES} attempts. Last error: {last_exc}"
        ) from last_exc

    def _ensure_connection(self) -> None:
        """Ping IRIS with ``SELECT 1`` and reconnect if the connection is lost."""
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        except Exception:
            logger.warning("IRIS connection lost — reconnecting...")
            self._connect_with_retry()

    def _cursor(self):
        self._ensure_connection()
        return self._conn.cursor()

    # ------------------------------------------------------------------
    # Schema DDL
    # ------------------------------------------------------------------

    def _drop_table(self) -> None:
        cur = self._cursor()
        try:
            cur.execute(f"DROP TABLE IF EXISTS SQLUser.{self.table_name}")
            self._conn.commit()
            logger.info("Dropped table 'SQLUser.%s'.", self.table_name)
        except Exception as exc:
            logger.warning("Could not drop table: %s", exc)
        finally:
            cur.close()

    def _create_table_if_not_exists(self) -> None:
        """
        Create the documents table in IRIS if it does not yet exist.

        The ``embedding`` column uses the native ``VECTOR(DOUBLE, N)`` type.
        The ``meta`` column stores JSON serialized with ``sort_keys=True``.
        """
        ddl = (
            f"CREATE TABLE IF NOT EXISTS SQLUser.{self.table_name} ("
            f"  id        VARCHAR(128)  NOT NULL PRIMARY KEY,"
            f"  content   LONGVARCHAR,"
            f"  meta      LONGVARCHAR,"
            f"  score     DOUBLE,"
            f"  embedding VECTOR(DOUBLE, {self.embedding_dim})"
            f")"
        )
        cur = self._cursor()
        try:
            cur.execute(ddl)
            self._conn.commit()
            logger.info("Table 'SQLUser.%s' ready.", self.table_name)
        except Exception as exc:
            logger.warning("Warning during table creation: %s", exc)
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Haystack 2.x DocumentStore protocol (mandatory)
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """
        Return the number of documents in the store.

        Returns
        -------
        int
            Total document count.

        Example
        -------
        >>> store.count_documents()
        5
        """
        cur = self._cursor()
        try:
            cur.execute(f"SELECT COUNT(*) FROM SQLUser.{self.table_name}")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            cur.close()

    def filter_documents(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Return documents matching the provided filters.

        Filtering is performed **in-memory** after loading all documents
        from IRIS.  This ensures compatibility with IRIS Community Edition
        (which lacks ``JSON_VALUE`` as a native SQL function) and supports
        all Python types and operators defined by the Haystack protocol.

        Parameters
        ----------
        filters:
            Filter in legacy format ``{"field": value}`` or in the
            official Haystack format ``{"operator": ..., "conditions": ...}``.
            ``None`` returns all documents.

        Returns
        -------
        list[Document]
            Documents satisfying the filter.

        Examples
        --------
        No filter — all documents::

            store.filter_documents()

        Legacy filter::

            store.filter_documents({"category": "db"})

        Integer filter::

            store.filter_documents({"year": 2024})

        Official Haystack filter with ``AND`` and ``>=``::

            store.filter_documents({
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "db"},
                    {"field": "meta.year",     "operator": ">=", "value": 2023},
                ],
            })
        """
        cur = self._cursor()
        try:
            cur.execute(
                f"SELECT id, content, meta, score "
                f"FROM SQLUser.{self.table_name}"
            )
            rows = cur.fetchall()
        finally:
            cur.close()

        docs = [self._row_to_document(row) for row in rows]
        if not filters:
            return docs
        #return [d for d in docs if _apply_filter(d.meta, filters)]
        return [d for d in docs if document_matches_filter(filters, d)]

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Persist documents to IRIS.

        Documents with embeddings are stored using ``TO_VECTOR(?, DOUBLE)``
        for native IRIS vector type conversion.  The ``meta`` field is
        serialized with ``json.dumps(sort_keys=True)`` to guarantee
        deterministic key ordering.

        Parameters
        ----------
        documents:
            List of :class:`~haystack.Document` objects to persist.
        policy:
            Duplicate handling policy:

            - ``FAIL`` *(default)*: raise :exc:`DuplicateDocumentError`.
            - ``SKIP``: silently ignore duplicate documents.
            - ``OVERWRITE``: replace the existing document.

        Returns
        -------
        int
            Number of documents written.

        Raises
        ------
        DuplicateDocumentError
            If ``policy=FAIL`` and a duplicate document ID is found.
        ValueError
            If ``documents`` contains non-:class:`~haystack.Document` objects.

        Example
        -------
        >>> from haystack import Document
        >>> from haystack.document_stores.types import DuplicatePolicy
        >>> store.write_documents(
        ...     [Document(content="Hello IRIS!", meta={"lang": "en"})],
        ...     policy=DuplicatePolicy.OVERWRITE,
        ... )
        1
        """
        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        written = 0
        cur = self._cursor()
        try:
            for doc in documents:
                if not isinstance(doc, Document):
                    raise ValueError(
                        f"Expected a Document object, got {type(doc).__name__!r}."
                    )
                existing = self._get_by_id(doc.id, cur)
                if existing:
                    if policy == DuplicatePolicy.FAIL:
                        raise DuplicateDocumentError(
                            f"Document with id '{doc.id}' already exists. "
                            "Use DuplicatePolicy.SKIP or OVERWRITE."
                        )
                    if policy == DuplicatePolicy.SKIP:
                        logger.debug("Skipping duplicate document: %s", doc.id)
                        continue
                    cur.execute(
                        f"DELETE FROM SQLUser.{self.table_name} WHERE id = ?",
                        [doc.id],
                    )

                meta_str = json.dumps(doc.meta or {}, ensure_ascii=False, sort_keys=True)
                emb_str = self._embedding_to_str(doc.embedding)

                if emb_str:
                    cur.execute(
                        f"INSERT INTO SQLUser.{self.table_name} "
                        f"(id, content, meta, score, embedding) "
                        f"VALUES (?, ?, ?, ?, TO_VECTOR(?, DOUBLE))",
                        [doc.id, doc.content or "", meta_str, doc.score, emb_str],
                    )
                else:
                    cur.execute(
                        f"INSERT INTO SQLUser.{self.table_name} "
                        f"(id, content, meta, score) VALUES (?, ?, ?, ?)",
                        [doc.id, doc.content or "", meta_str, doc.score],
                    )
                written += 1

            self._conn.commit()
            logger.debug("Wrote %d document(s) to IRIS.", written)
            return written
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by ID.

        Accepts an empty list without error (idempotent). Non-existent
        IDs are silently ignored by IRIS.

        Parameters
        ----------
        document_ids:
            List of document IDs to remove.

        Example
        -------
        >>> store.delete_documents(["id-1", "id-2"])
        """
        if not document_ids:
            return
        placeholders = ",".join(["?"] * len(document_ids))
        cur = self._cursor()
        try:
            cur.execute(
                f"DELETE FROM SQLUser.{self.table_name} "
                f"WHERE id IN ({placeholders})",
                document_ids,
            )
            self._conn.commit()
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Extra retrieval methods (used by companion Retrievers)
    # ------------------------------------------------------------------

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Return the top-K most similar documents using IRIS ``VECTOR_COSINE``.

        This method is **not** part of the Haystack DocumentStore protocol.
        It is called internally by :class:`IRISEmbeddingRetriever`.

        Parameters
        ----------
        query_embedding:
            Query vector with ``embedding_dim`` dimensions.
        filters:
            Post-search filters (applied in-memory).
        top_k:
            Maximum number of documents to return.

        Returns
        -------
        list[Document]
            Documents with ``score`` = ``VECTOR_COSINE`` value (range: -1 to 1),
            in descending similarity order.

        Raises
        ------
        DocumentStoreError
            If the vector search query fails.
        ValueError
            If ``query_embedding`` is empty or ``None``.
        """
        emb_str = self._embedding_to_str(query_embedding)
        if not emb_str:
            raise ValueError("query_embedding cannot be None or empty.")

        fetch_k = top_k * 4 if filters else top_k
        sql = (
            f"SELECT TOP ? id, content, meta, score, "
            f"VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE)) AS similarity "
            f"FROM SQLUser.{self.table_name} "
            f"WHERE embedding IS NOT NULL "
            f"ORDER BY similarity DESC"
        )
        cur = self._cursor()
        try:
            cur.execute(sql, [fetch_k, emb_str])
            docs: list[Document] = []
            for row in cur.fetchall():
                doc = self._row_to_document(row)
                similarity = float(row[4]) if row[4] is not None else None
                doc = dataclasses.replace(doc, score=similarity)
                
                #if filters and not _apply_filter(doc.meta, filters):
                if filters and not document_matches_filter(filters, doc):
                    continue
                docs.append(doc)
                if len(docs) >= top_k:
                    break
            return docs
        except Exception as exc:
            msg = f"IRIS vector search failed: {exc}"
            raise DocumentStoreError(msg) from exc
        finally:
            cur.close()

    def _bm25_retrieval(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Return the top-K most relevant documents using Okapi BM25.

        This method is **not** part of the Haystack DocumentStore protocol.
        It is called internally by :class:`IRISBm25Retriever`.

        Parameters
        ----------
        query:
            Natural language search text.
        filters:
            Filters applied before BM25 ranking.
        top_k:
            Maximum number of documents to return.

        Returns
        -------
        list[Document]
            Documents with ``score`` = BM25 value, in descending relevance
            order. Documents with score = 0 are excluded.
        """
        candidates = self.filter_documents(filters=filters)
        if not candidates:
            return []
        self._bm25.build([(d.id, d.content or "") for d in candidates])
        doc_map = {d.id: d for d in candidates}
        return [
            dataclasses.replace(doc_map[doc_id], score=score)
            for doc_id, score in self._bm25.query(query, top_k)
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the store for use in Haystack YAML/JSON pipelines.

        The password ``Secret`` is serialized according to the Haystack
        secret protocol — the resolved value is **never** included.

        Returns
        -------
        dict
            Serializable dictionary.

        Example
        -------
        >>> d = store.to_dict()
        >>> d["type"]
        'haystack_integrations.document_stores.iris.document_store.IRISDocumentStore'
        """
        return default_to_dict(
            self,
            connection_string=self.connection_string.to_dict(),
            username=self.username.to_dict(),
            password=self.password.to_dict(),
            table_name=self.table_name,
            embedding_dim=self.embedding_dim,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            recreate_table=False,  # never re-drop on restore
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IRISDocumentStore":
        """
        Deserialize the store from a dictionary.

        Called automatically by Haystack when loading a pipeline from a
        YAML file.

        Parameters
        ----------
        data:
            Dictionary in the format produced by :meth:`to_dict`.

        Returns
        -------
        IRISDocumentStore
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            keys=["connection_string", "username", "password"],
        )
        return default_from_dict(cls, data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_by_id(self, doc_id: str, cur=None):
        _cur = cur or self._cursor()
        _cur.execute(
            f"SELECT id FROM SQLUser.{self.table_name} WHERE id = ?", [doc_id]
        )
        return _cur.fetchone()

    @staticmethod
    def _embedding_to_str(embedding: list[float] | None) -> str | None:
        """
        Convert a list of floats to the string format expected by
        ``TO_VECTOR(?, DOUBLE)`` in IRIS: ``[v1,v2,...,vN]``.
        """
        if not embedding:
            return None
        return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"

    @staticmethod
    def _row_to_document(row) -> Document:
        """
        Convert a DB-API row (id, content, meta, score, ...) to a Document.

        The ``embedding`` field is **not** deserialized for performance.
        The returned ``Document.embedding`` will be ``None``.
        """
        meta = json.loads(row[2]) if row[2] else {}
        return Document(
            id=row[0],
            content=row[1],
            meta=meta,
            score=float(row[3]) if row[3] is not None else None,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the IRIS connection (idempotent)."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            logger.debug("IRIS connection closed.")

    def __enter__(self) -> "IRISDocumentStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"IRISDocumentStore("
            f"table={self.table_name!r}, "
            f"embedding_dim={self.embedding_dim})"
        )
