# Skill: career_planning

## 概述
职业规划技能。当用户询问"怎么发展"、"职业方向"、"要不要转"、"该学什么"时，触发此技能。

## 触发条件
```
intent_mode: career_planning
```

## 实现
```
type: node
entry: src.nodes.career_intents.career_planning
```

## 描述
结合 Arthur 的简历、历史对话，给出个性化职业发展路径建议。

## 输入
- `optimized_text`: 用户输入的职业问题/背景描述

## 输出
- `career_plan`: 个性化发展规划（含短期/中期/长期）
- `rag_sources`: 关联的简历文档

## 配置
```
version: "1.0"
requires_agents: ["intent_recognition", "rag"]
```

## 关联
- 依赖 skill: 无
- 触发 intent_mode: career_planning