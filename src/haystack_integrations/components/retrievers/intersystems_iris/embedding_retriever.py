# SPDX-FileCopyrightText: 2026-present Pedro Henrique <pedro@s-c.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore


@component
class IRISEmbeddingRetriever:
    """
    Retrieve documents from :class:`IRISDocumentStore` by embedding similarity.

    Uses IRIS native ``VECTOR_COSINE`` function for semantic similarity
    computation.  The similarity metric depends on the model used to generate
    the embeddings.

    **Usage in a pipeline**

    .. code-block:: python

        from haystack import Pipeline
        from haystack.components.embedders import SentenceTransformersTextEmbedder
        from haystack_integrations.document_stores.iris import IRISDocumentStore
        from haystack_integrations.components.retrievers.iris import (
            IRISEmbeddingRetriever,
        )

        store = IRISDocumentStore()
        pipeline = Pipeline()
        pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        pipeline.add_component(
            "retriever", IRISEmbeddingRetriever(document_store=store)
        )
        pipeline.connect("embedder.embedding", "retriever.query_embedding")

        result = pipeline.run({"embedder": {"text": "What is IRIS?"}})
        print(result["retriever"]["documents"])

    Parameters
    ----------
    document_store:
        An :class:`IRISDocumentStore` instance.
    filters:
        Filters applied to the retrieved documents at initialization time.
        Runtime filters are merged/replaced according to ``filter_policy``.
    top_k:
        Maximum number of documents to retrieve.
    filter_policy:
        Determines how ``filters`` set at init and ``filters`` passed at
        runtime interact.

        - ``REPLACE`` *(default)*: runtime filters override init filters.
        - ``MERGE``: runtime filters are combined with init filters.

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
            msg = "IRISEmbeddingRetriever requires an IRISDocumentStore instance."
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve the most similar documents for a query embedding.

        Parameters
        ----------
        query_embedding:
            Query vector (must have the same dimensions as stored embeddings).
        filters:
            Runtime filters. Combined with init-time filters according to
            ``filter_policy``.
        top_k:
            Override the default ``top_k`` set at initialization.

        Returns
        -------
        dict
            ``{"documents": [Document, ...]}``.
        """
        resolved_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=resolved_filters or None,
            top_k=top_k if top_k is not None else self.top_k,
        )
        return {"documents": docs}

    def to_dict(self) -> dict[str, Any]:
        """Serializes the component to a dictionary."""
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IRISEmbeddingRetriever":
        """Deserializes the component from a dictionary."""
        data["init_parameters"]["document_store"] = IRISDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        if "filter_policy" in data["init_parameters"]:
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(data["init_parameters"]["filter_policy"])
        return default_from_dict(cls, data)
