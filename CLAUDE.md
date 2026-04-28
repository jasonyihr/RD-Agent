# 目标
学习rdagent。帮助我确定研究方向以及协助后续论文写作。

# 现状
## 已完成的学习
- 读了架构文档 `docs/architecture.md`，理解了整体结构和四种量化金融模式
- 读了核心抽象：
  - `core/evolving_framework.py`：EvolvingStrategy、RAGStrategy、EvoStep 等底层抽象
  - `core/proposal.py`：HypothesisGen、Trace（DAG结构）、Hypothesis 等核心接口
  - `components/workflow/rd_loop.py`：RDLoop 主循环（propose → exp_gen → coding → running → feedback）
- 读了 qlib 因子提案生成链路：
  - `components/proposal/__init__.py`：LLMHypothesisGen（通用 LLM 假设生成框架）
  - `scenarios/qlib/proposal/factor_proposal.py`：QlibFactorHypothesisGen
  - `components/proposal/prompts.yaml`：假设生成的 system/user prompt 模板
  - `scenarios/qlib/prompts.yaml`：qlib 场景的 prompt 模板

## 关键发现
- **方向1（提案方向控制）确认存在**：假设生成完全靠 LLM 阅读历史 trace 自由发挥，没有结构化的方向管理（如"已探索方向"、"方向深度"、"切换策略"）。方向控制仅靠 prompt 里一句话。
- **方向2（fin_factor_report 割裂）确认存在**：`fin_factor_report` 是独立的 `FactorReportLoop`，不走主 `RDLoop` 流程。

## 下一步
- 看 `quant_proposal.py` 和 `bandit.py`，理解联合场景下的动作选择逻辑
- 开始构思方向1的改进方案


# 可能的方向
- 其实我觉得rd有个没做好的地方，就是生成的提案没有做方向控制，这一条效果不好，马上跳一个提案差别很大的组合，就有点类似那种老式的扫地机器人，左边撞一下，右边撞一下
- 还有就是这个fin_factor_report(从报告提取因子)和主流程是割裂的

# 原则
- 如果有假设，我们直接验证，给出验证方案，等待批准
- 如果想查阅论文，可以自行搜索并下载，查阅之前告诉我
- 如果写代码，首先要遵循渐进式架构，对创建的文件进行收录而不是创建空文件夹，写代码前要给我讲解逻辑，等待批准
- 如果要写探索性代码，得到的知识要记录下来，记录之前告诉我。
- 如果从错误中获得知识，也要记录下来。记录之前告诉我。
- 如果写功能性代码，遵循TDD。
- 如果又产生新的idea，可以写在“可能的方向”那里，或者直接作为假设开始验证。
- 做好过程管理，比如日志。
- 如果有原项目文件，注意新建文件的时候做好区分。如果需要在源文件上改动，务必告知我，等待批准。
- 做好文件保存，比如及时提交github。
- 不能自主决定删除文件，必须提前告知我并等待批准。
- 用继承扩展而非源码修改。