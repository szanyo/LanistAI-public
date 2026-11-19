# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Route grouped-library modal effects through the model-scoped application service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from lanista.gui.services.configuration.evaluation_library_draft_service import (
    EvaluationLibraryDraft,
    EvaluationLibraryDraftService,
)
from lanista.reinforcengine.rl_model.evaluation.contracts import LibraryEvaluationScope
from lanista.reinforcengine.rl_model.evaluation.library_application_service import (
    EvaluationLibraryApplicationService,
    EvaluationLibraryCatalogItem,
)


@dataclass(frozen=True, slots=True)
class EvaluationLibrariesDialogAdapter:
    """Keep GUI widgets free of generation, catalog, and path decisions."""

    service: EvaluationLibraryApplicationService
    draft_service: EvaluationLibraryDraftService = EvaluationLibraryDraftService()

    def new_draft(self) -> EvaluationLibraryDraft:
        now = datetime.now(UTC)
        draft = EvaluationLibraryDraft(
            model_root=self.service.current_model_root(),
            library_id=f"library-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}",
            name="",
            description="",
            created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            master_seed="42",
        )
        return self.draft_service.add_group(draft)

    def refresh(
        self, draft: EvaluationLibraryDraft
    ) -> tuple[EvaluationLibraryDraft, tuple[EvaluationLibraryCatalogItem, ...]]:
        """Invalidate stale form/selection state before displaying another model."""

        if self.service.current_model_root() != draft.model_root:
            draft = self.new_draft()
        return draft, self.service.list_for_current_model()

    def generate(self, draft: EvaluationLibraryDraft) -> tuple[str, str]:
        """Publish one verified library from a draft tied to the selected model."""

        if self.service.current_model_root() != draft.model_root:
            raise RuntimeError("Selected model changed; reopen Evaluation Libraries.")
        parameters = self.draft_service.validate(draft)
        generated = self.service.generate_for_current_model(
            parameters, expected_model_root=draft.model_root
        )
        return generated.manifest.library_id, generated.manifest_sha256

    def select(
        self,
        draft: EvaluationLibraryDraft,
        item: EvaluationLibraryCatalogItem,
    ) -> tuple[str, str]:
        """Reject invalid or cross-model catalog rows before closing the modal."""

        if self.service.current_model_root() != draft.model_root:
            raise RuntimeError("Selected model changed; reopen Evaluation Libraries.")
        if not item.valid or item.manifest_sha256 is None:
            raise ValueError("This library failed integrity verification and cannot be selected.")
        scope = LibraryEvaluationScope(
            library_id=item.library_id,
            manifest_sha256=item.manifest_sha256,
        )
        self.service.resolve_selection(scope)
        return item.library_id, item.manifest_sha256


__all__ = ["EvaluationLibrariesDialogAdapter"]
