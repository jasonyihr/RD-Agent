"""
Configuration for directed (direction-aware) hypothesis generation.

Provides a settings class that defaults to DirectedQlibQuantHypothesisGen
instead of the baseline QlibQuantHypothesisGen. All other settings remain
identical to QuantBasePropSetting.

Usage:
  - Via code: ``QuantRDLoop(DIRECTED_QUANT_PROP_SETTING)``
  - Via env var: ``QLIB_QUANT_QUANT_HYPOTHESIS_GEN=rdagent.scenarios.qlib.proposal.directed_quant_proposal.DirectedQlibQuantHypothesisGen``
"""

from rdagent.app.qlib_rd_loop.conf import QuantBasePropSetting


class DirectedQuantPropSetting(QuantBasePropSetting):
    quant_hypothesis_gen: str = (
        "rdagent.scenarios.qlib.proposal.directed_quant_proposal.DirectedQlibQuantHypothesisGen"
    )


DIRECTED_QUANT_PROP_SETTING = DirectedQuantPropSetting()
