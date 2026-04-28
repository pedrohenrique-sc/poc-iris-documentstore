# SPDX-FileCopyrightText: 2026-present Pedro Henrique <pedro@s-c.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    WriteDocumentsTest,
)


@pytest.mark.integration
class TestCountDocuments(CountDocumentsTest):
    @pytest.fixture
    def document_store(self, document_store):
        return document_store

@pytest.mark.integration
class TestWriteDocuments(WriteDocumentsTest):
    @pytest.fixture
    def document_store(self, document_store):
        return document_store

    def test_write_documents(self, document_store):
        docs = [Document(id="1", content="teste")]
        assert document_store.write_documents(docs) == 1

        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs)

@pytest.mark.integration
class TestFilterDocuments(FilterDocumentsTest):
    """
    Official Haystack filter_documents() tests.
    """

    @pytest.fixture
    def document_store(self, document_store):
        return document_store

    def assert_documents_are_equal(self, received, expected):
        assert len(received) == len(expected)

        rec_sorted = sorted(received, key=lambda d: d.id)
        exp_sorted = sorted(expected, key=lambda d: d.id)

        for r, e in zip(rec_sorted, exp_sorted, strict=True):
            assert r.id == e.id
            assert r.content == e.content
            assert r.meta == e.meta

@pytest.mark.integration
class TestDeleteDocuments(DeleteDocumentsTest):
    """Official Haystack delete_documents() tests."""

    @pytest.fixture
    def document_store(self, document_store):
        return document_store
