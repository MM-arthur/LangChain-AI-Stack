# Skill: mock_interview

## 概述
多轮模拟面试技能。当用户要求"模拟面试"、"来面试"、"面试一下"时，触发此技能。

## 触发条件
```
intent_mode: mock_interview
```

## 实现
```
type: node
entry: src.nodes.career_intents.mock_interview
```

## 描述
模拟面试官角色，多轮问答，结束时输出结构化评估报告。

## 输入
- `optimized_text`: 用户当前回答（第一轮时为空）
- `intent_mode`: mock_interview
- `interview_history`: 累计的问答历史
- `current_round`: 当前轮次

## 输出
- `response`: 面试官的问题或反馈
- `mock_interview_mode`: 是否继续
- `current_round`: 轮次计数
- `interview_history`: 更新后的历史

## 配置
```
version: "1.0"
max_rounds: 5
requires_agents: ["intent_recognition"]
```

## 关联
- 依赖 skill: 无
- 触发 intent_mode: mock_interview