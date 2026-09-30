import uuid


def create_id(prefix: str = "comment") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"