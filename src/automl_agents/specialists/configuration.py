"""Especialista puro de configuración; reutiliza el generador y no persiste."""
from .. import candidates
from ..config import RunConfig
from ..dataset import DatasetProfile
from ..kb import Doc
from ..schemas import DomainError
from .contracts import ConfigurationContext, ConfigurationResult


class ConfigurationSpecialist:
    def run(self, context: ConfigurationContext, llm=None) -> ConfigurationResult:
        context = ConfigurationContext.model_validate(context.to_dict())
        p, limits = context.profile, context.constraints
        profile = DatasetProfile("", p.classes, p.counts, p.n_images, p.imbalance_ratio,
                                 p.median_size, p.min_side, scope="train")
        cfg = RunConfig(dataset_dir="", **limits.to_dict(), allowed_architectures=list(context.catalog))
        docs = [Doc(**d.to_dict()) for d in context.evidence]
        mode, error, note = "reglas", None, ""
        if llm is not None and llm.available:
            try:
                cands, discarded = candidates.propose_llm(profile, docs, cfg, llm)
                return ConfigurationResult(candidates=cands, discarded=discarded, mode="llm").validate_context(context)
            except Exception as exc:
                error = DomainError(code="candidate_proposal_failed", message=str(exc) or type(exc).__name__, recoverable=True)
                note = f"Fallo del LLM ({type(exc).__name__}); se usó el generador por reglas."
        cands, discarded = candidates.propose_rules(profile, docs, cfg)
        return ConfigurationResult(candidates=cands, discarded=discarded, mode=mode,
                                   error=error, note=note).validate_context(context)
