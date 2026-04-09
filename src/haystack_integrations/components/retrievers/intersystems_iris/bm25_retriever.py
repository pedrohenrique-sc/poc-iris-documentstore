# SPDX-FileCopyrightText: 2026-present Pedro Henrique <pedro@s-c.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore


@component
class IRISBm25Retriever:
    """
    Retrieve documents from :class:`IRISDocumentStore` using Okapi BM25.

    The BM25 index is rebuilt in-memory over the (optionally filtered)
    document set on every :meth:`run` call.  For large collections, apply
    ``filters`` to narrow the candidate set before ranking.

    **Usage in a pipeline**

    .. code-block:: python

        from haystack import Pipeline
        from haystack_integrations.document_stores.iris import IRISDocumentStore
        from haystack_integrations.components.retrievers.iris import IRISBm25Retriever

        store = IRISDocumentStore()
        pipeline = Pipeline()
        pipeline.add_component("retriever", IRISBm25Retriever(document_store=store))

        result = pipeline.run({"retriever": {"query": "multimodel database", "top_k": 5}})
        print(result["retriever"]["documents"])

    Parameters
    ----------
    document_store:
        An :class:`IRISDocumentStore` instance.
    filters:
        Filters applied before BM25 ranking at initialization time.
    top_k:
        Maximum number of documents to retrieve.
    filter_policy:
        Determines how init-time and runtime filters interact
        (``REPLACE`` or ``MERGE``).

    Raises
    ------
    ValueError
        If ``document_store`` is not an :class:`IRISDocumentStore`.
    """

    def __init__(
        self,
        *,
        document_store: IRISDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: FilterPolicy | str = FilterPolicy.REPLACE,
    ) -> None:
        if not isinstance(document_store, IRISDocumentStore):
            msg = "IRISBm25Retriever requires an IRISDocumentStore instance."
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            FilterPolicy.from_str(filter_policy)
            if isinstance(filter_policy, str)
            else filter_policy
        )

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve the most relevant documents for a keyword query.

        Parameters
        ----------
        query:
            Natural language search text.
        filters:
            Runtime filters merged with init-time filters per ``filter_policy``.
        top_k:
            Override the default ``top_k``.

        Returns
        -------
        dict
            ``{"documents": [Document, ...]}``.
        """
        resolved_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store._bm25_retrieval(
            query=query,
            filters=resolved_filters or None,
            top_k=top_k if top_k is not None else self.top_k,
        )
        return {"documents": docs}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IRISBm25Retriever":
        data["init_parameters"]["document_store"] = IRISDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        if "filter_policy" in data["init_parameters"]:
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(
                data["init_parameters"]["filter_policy"]
            )
        return default_from_dict(cls, data)
