"""zt-memory: local context engine for code LLMs (symbol-level hybrid retrieval + edit-energy prior, SSD store)."""
from .chunking import CodeChunk, chunk_source
from .config import KindPhysics, MemoryConfig, NodeKind, RouterConfig, default_physics
from .encoders import HashingEncoder, HiddenStateEncoder, TransformersHiddenStateEncoder
from .engine_client import ComputeEngineClient, GeneratedFile, parse_generated_files, write_generated_files
from .lexical import LexicalHit, LexicalIndex, build_query
from .manager import IncompatibleStoreError, IngestReport, NewtonianMemoryManager, Requirement
from .projector import SubspaceProjector
from .router import ContextBlock, GeometricRouter, RoutedContext, route_to_compute_engine
from .syntax import LanguageSpec, available_languages, register_language

__all__ = [
    "CodeChunk", "LanguageSpec", "LexicalHit", "LexicalIndex", "available_languages", "build_query",
    "chunk_source", "register_language",
    "ComputeEngineClient", "ContextBlock", "GeneratedFile", "GeometricRouter", "HashingEncoder",
    "HiddenStateEncoder", "IncompatibleStoreError", "IngestReport", "KindPhysics", "MemoryConfig",
    "NewtonianMemoryManager", "NodeKind", "Requirement", "RoutedContext", "RouterConfig",
    "SubspaceProjector", "TransformersHiddenStateEncoder", "default_physics", "parse_generated_files",
    "route_to_compute_engine", "write_generated_files",
]
