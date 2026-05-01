# Evaluation Memory 安全红线

## 绝对禁止

1. ❌ 任何个人信息（手机/邮箱/地址/证件号/姓名）存入 memory
2. ❌ 原始简历内容存入 memory（只存 topic/skill 名 + 分数）
3. ❌ raw_report 对外暴露（只做内部聚合用）
4. ❌ profile.json 传给前端或 API 响应

## 允许的安全数据

```json
{
  "total_interviews": 5,
  "avg_overall_score": 3.8,
  "topic_stats": {
    "Redis": {"avg": 3.5, "trend": "up", "count": 3},
    "Vue": {"avg": 4.2, "trend": "stable", "count": 2}
  },
  "strengths": ["Vue3 原理", "项目架构"],
  "weaknesses": ["消息队列细节"]
}
```

## 数据流向

```
LLM 报告（含上下文）
  → parse_llm_report() 提取评分
  → EvaluationMemory.save_evaluation()
      ├── history/eval_YYYY-MM-DD_XXX.json  # 原始报告（内部）
      └── profile.json  # 聚合数据（AI 参考）
```

## 审计

每次保存后打印：
> `✅ 评估记忆已保存: {record_id} (session={session_id}, overall={overall_score})`

不含任何个人信息。