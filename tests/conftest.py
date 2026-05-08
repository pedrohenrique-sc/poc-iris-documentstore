# SPDX-FileCopyrightText: 2026-present Pedro Henrique <pedro@s-c.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from haystack.utils import Secret

from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


@pytest.fixture()
def document_store():
    store = IRISDocumentStore(
        connection_string=Secret.from_token(os.getenv("IRIS_CONNECTION_STRING", "localhost:1972/USER")),
        username=Secret.from_token(os.getenv("IRIS_USERNAME", "_system")),
        password=Secret.from_token(os.getenv("IRIS_PASSWORD", "SYS")),
        table_name="HaystackTest768",
        embedding_dim=768,
    )
    # Clean before test
    ids = [d.id for d in store.filter_documents()]
    store.delete_documents(ids)
    yield store
    # Clean after test
    ids = [d.id for d in store.filter_documents()]
    store.delete_documents(ids)
    store.close()
