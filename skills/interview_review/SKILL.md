# Skill: interview_review

## 概述
面试复盘技能。当用户说"复盘"、"面试表现怎么样"、"我今天面了"时，触发此技能。

## 触发条件
```
intent_mode: interview_review
```

## 实现
```
type: node
entry: src.nodes.career_intents.interview_review
```

## 描述
接收用户描述的面试内容，对照 JD 和简历，输出技术评分 + 改进建议。

## 输入
- `optimized_text`: 用户输入的面试内容描述

## 输出
- `review_report`: 结构化复盘报告
- `rag_sources`: 关联的简历/JD 文档

## 配置
```
version: "1.0"
requires_agents: ["intent_recognition", "rag"]
```

## 关联
- 依赖 skill: 无
- 触发 intent_mode: interview_review