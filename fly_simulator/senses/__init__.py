"""Body senses that feed the connectome brain (taste patches: docs/TASTE.md)."""

from .taste import (PATCH_KINDS, Feed, FeedingRule, TasteConfig, TasteHandle, TastePatches,
                    TasteSense, install_taste)

__all__ = ["PATCH_KINDS", "Feed", "FeedingRule", "TasteConfig", "TasteHandle", "TastePatches",
           "TasteSense", "install_taste"]
