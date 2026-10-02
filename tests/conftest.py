import pytest

from automl_agents.kb import KnowledgeBase


@pytest.fixture(scope="session")
def kb():
    return KnowledgeBase(backend="hash")
