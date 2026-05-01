"""
BaseService is the common base for all service classes.
It holds a reference to the repository so subclasses don't repeat the
constructor boilerplate and the dependency is consistently named.
"""


class BaseService:
    """Injects a repository dependency that subclasses access via self.repository."""

    def __init__(self, repository):
        self.repository = repository
