# SPDX-FileCopyrightText: 2026-present Pedro Henrique <pedro@s-c.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore


class TestIRISDocumentStoreUnit:
    """Unit tests that do not require a running IRIS instance."""

    def test_embedding_to_str(self):
        result = IRISDocumentStore._embedding_to_str([0.1, 0.2, 0.3])
        assert result == "[0.10000000,0.20000000,0.30000000]"

    def test_embedding_to_str_none(self):
        assert IRISDocumentStore._embedding_to_str(None) is None
        assert IRISDocumentStore._embedding_to_str([]) is None

    def test_row_to_document(self):
        row = ("id-1", "hello world", '{"key": "val"}', 0.85)
        doc = IRISDocumentStore._row_to_document(row)
        assert doc.id == "id-1"
        assert doc.content == "hello world"
        assert doc.meta == {"key": "val"}
        assert doc.score == pytest.approx(0.85)

    def test_row_to_document_empty_meta(self):
        row = ("id-2", "content", None, None)
        doc = IRISDocumentStore._row_to_document(row)
        assert doc.meta == {}
        assert doc.score is None
