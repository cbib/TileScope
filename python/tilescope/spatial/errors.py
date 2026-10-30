"""Typed error hierarchy for bank / projection contract violations."""

from __future__ import annotations


class BankFingerprintMismatchError(ValueError):
    """Raised when a projection or pair result carries a fingerprint that
    disagrees with the spatial bank it is being attached to. Structural
    guarantee: never trigger under correct construction."""


class FeatureAttachmentError(ValueError):
    """Raised when a feature vector's length or dtype disagrees with the
    bank's SourceSpotTable at attachment time (bank-side length contract)."""
