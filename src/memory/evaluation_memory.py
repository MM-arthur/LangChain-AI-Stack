"""
Evaluation Memory System - 面试评估历史记忆系统

安全原则：
- 只存储 topic/skill 名称和评分，不存储任何个人信息
- 历史记录存原始报告（包含上下文），profile 只存聚合数据
- 所有数据存在 app data 目录，不对外暴露
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class TopicScore:
    """单一 topic 的评分记录"""
    topic: str                      # e.g., "Redis", "系统设计", "Vue"
    score: float                     # 1-5 分
    weight: float = 1.0              # 权重（题目难度系数）
    notes: str = ""                  # 简短备注（无个人信息）


@dataclass
class EvaluationRecord:
    """单次面试评估记录"""
    id: str                          # UUID
    timestamp: str                  # ISO format
    session_id: str                  # 对应的 session
    duration_rounds: int             # 面试轮次
    overall_score: float             # 综合评分 1-5
    
    # 各维度评分
    technical_depth: float           # 技术深度
    communication: float             # 表达能力
    project_experience: float       # 项目经验
    problem_solving: float          # 临场反应
    
    # Topic 级评分（核心记忆）
    topic_scores: List[TopicScore]  # 各 topic 得分
    
    # 原始报告（供后续分析，不直接给 AI 看）
    raw_report: str                 # LLM 输出的完整 Markdown 报告
    summary: str                    # 一句话总结（用于快速检索）
    
    # 安全：不含任何个人信息


@dataclass
class ArthurProfile:
    """Arthur 的面试画像（聚合数据，无个人信息）"""
    # 统计信息
    total_interviews: int = 0
    last_updated: str = ""
    avg_overall_score: float = 0.0
    
    # Topic 级聚合（关键记忆）
    topic_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # {
    #   "Redis": {"avg": 3.5, "trend": "up", "count": 3, "sessions": ["s1","s2"]},
    #   "Vue": {"avg": 4.2, "trend": "stable", "count": 2, "sessions": ["s1"]}
    # }
    
    # 优势/劣势（AI 提炼）
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    # 最近重点练习方向
    recent_focus: List[str] = field(default_factory=list)
    
    # AI 给出的改进建议
    suggestions: List[str] = field(default_factory=list)


# ── Core Class ──────────────────────────────────────────────────────────────────

class EvaluationMemory:
    """
    面试评估记忆系统
    
    安全边界：
    - 不存储任何个人信息（手机/邮箱/地址/证件号）
    - topic_scores 只存 skill 名和分数
    - raw_report 存 LLM 输出（可能含上下文），但不对外暴露
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # 默认使用项目 data 目录
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "evaluation_memory"
        else:
            self.data_dir = Path(data_dir)
        
        self.history_dir = self.data_dir / "history"
        self.profile_path = self.data_dir / "profile.json"
        self.index_path = self.data_dir / "index.json"
        
        # 创建目录
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化/加载 profile
        self.profile = self._load_profile()
    
    # ── Persistence ──────────────────────────────────────────────────────────
    
    def _load_profile(self) -> ArthurProfile:
        """加载 profile，不存在则创建空 profile"""
        if self.profile_path.exists():
            try:
                data = json.loads(self.profile_path.read_text())
                return ArthurProfile(**data)
            except Exception:
                pass
        return ArthurProfile()
    
    def _save_profile(self):
        """持久化 profile"""
        self.profile_path.write_text(json.dumps(asdict(self.profile), indent=2, ensure_ascii=False))
    
    def _load_index(self) -> Dict[str, Any]:
        """加载索引：session_id → record_id"""
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {}
    
    def _save_index(self, index: Dict[str, Any]):
        self.index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    
    # ── Core Operations ─────────────────────────────────────────────────────
    
    def save_evaluation(
        self,
        session_id: str,
        rounds: int,
        overall_score: float,
        technical_depth: float,
        communication: float,
        project_experience: float,
        problem_solving: float,
        topic_scores: List[Dict[str, Any]],  # [{"topic": "Redis", "score": 3.5, "notes": ""}]
        raw_report: str,
        summary: str
    ) -> str:
        """
        保存一次面试评估
        
        Returns:
            record_id: 此次记录的 UUID
        """
        record_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # 构建记录
        record = EvaluationRecord(
            id=record_id,
            timestamp=timestamp,
            session_id=session_id,
            duration_rounds=rounds,
            overall_score=overall_score,
            technical_depth=technical_depth,
            communication=communication,
            project_experience=project_experience,
            problem_solving=problem_solving,
            topic_scores=[TopicScore(**t) for t in topic_scores],
            raw_report=raw_report,
            summary=summary
        )
        
        # 1. 保存历史记录
        record_path = self.history_dir / f"eval_{timestamp[:10]}_{record_id}.json"
        record_path.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False))
        
        # 2. 更新索引
        index = self._load_index()
        if session_id not in index:
            index[session_id] = []
        index[session_id].append(record_id)
        self._save_index(index)
        
        # 3. 更新 profile（聚合统计）
        self._update_profile(record)
        
        print(f"✅ 评估记忆已保存: {record_id} (session={session_id}, overall={overall_score})")
        return record_id
    
    def _update_profile(self, record: EvaluationRecord):
        """更新聚合 profile"""
        p = self.profile
        p.total_interviews += 1
        p.last_updated = record.timestamp
        
        # 更新 overall avg（简单移动平均）
        p.avg_overall_score = (p.avg_overall_score * (p.total_interviews - 1) + record.overall_score) / p.total_interviews
        
        # 更新各 topic 统计
        for ts in record.topic_scores:
            topic = ts.topic
            score = ts.score
            
            if topic not in p.topic_stats:
                p.topic_stats[topic] = {"avg": 0.0, "count": 0, "scores": [], "sessions": []}
            
            stats = p.topic_stats[topic]
            stats["count"] += 1
            stats["scores"].append(score)
            stats["sessions"].append(record.session_id)
            stats["avg"] = sum(stats["scores"]) / len(stats["scores"])
            
            # 计算 trend（最近 3 次 vs 再之前 3 次）
            if len(stats["scores"]) >= 2:
                recent = stats["scores"][-3:] if len(stats["scores"]) >= 3 else stats["scores"][-2:]
                older = stats["scores"][:-3] if len(stats["scores"]) > 3 else stats["scores"][:1]
                if older and sum(recent) / len(recent) > sum(older) / len(older):
                    stats["trend"] = "up"
                elif older and sum(recent) / len(recent) < sum(older) / len(older):
                    stats["trend"] = "down"
                else:
                    stats["trend"] = "stable"
            else:
                stats["trend"] = "stable"
        
        # 提炼 strengths/weaknesses（只看 topic_avg >= 4 为强项，<= 2.5 为弱项）
        all_topics = [(t, s["avg"]) for t, s in p.topic_stats.items()]
        strengths = [t for t, avg in all_topics if avg >= 4.0]
        weaknesses = [t for t, avg in all_topics if avg <= 2.5]
        p.strengths = strengths
        p.weaknesses = weaknesses
        
        # 最近 focus（出现最多的 topic，取 top 3）
        topic_counts = [(t, s["count"]) for t, s in p.topic_stats.items()]
        topic_counts.sort(key=lambda x: x[1], reverse=True)
        p.recent_focus = [t for t, _ in topic_counts[:3]]
        
        self._save_profile()
    
    # ── Query Interface ───────────────────────────────────────────────────────
    
    def get_profile(self) -> ArthurProfile:
        """获取 Arthur 的面试画像"""
        return self.profile
    
    def get_recent_topic_stats(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取最近练习的 topic 及统计（用于 AI 建议）"""
        stats = []
        for topic, data in self.profile.topic_stats.items():
            stats.append({
                "topic": topic,
                "avg_score": round(data["avg"], 1),
                "trend": data.get("trend", "stable"),
                "count": data["count"]
            })
        
        # 按 count 排序（最近练习最多的排前面）
        stats.sort(key=lambda x: x["count"], reverse=True)
        return stats[:limit]
    
    def get_topic_history(self, topic: str) -> List[Dict[str, Any]]:
        """获取某个 topic 的历史评分（用于趋势分析）"""
        if topic not in self.profile.topic_stats:
            return []
        
        stats = self.profile.topic_stats[topic]
        return [
            {"score": s, "session": sess}
            for s, sess in zip(stats["scores"][-10:], stats["sessions"][-10:])
        ]
    
    def suggest_improvement(self) -> str:
        """AI 生成改进建议（基于 profile）"""
        p = self.profile
        
        if p.total_interviews == 0:
            return "还没有面试记录，从模拟面试开始吧！"
        
        suggestions = []
        
        # 找出最弱的 topic（avg 最低且 count >= 2）
        weak_topics = [
            (t, d["avg"]) for t, d in p.topic_stats.items()
            if d["count"] >= 2 and d["avg"] < 3.5
        ]
        if weak_topics:
            weak_topics.sort(key=lambda x: x[1])
            suggestions.append(f"建议加强练习：{weak_topics[0][0]}（当前平均 {weak_topics[0][1]:.1f}/5）")
        
        # 找出进步中的 topic（trend = up）
        improving = [
            t for t, d in p.topic_stats.items()
            if d.get("trend") == "up" and d["count"] >= 2
        ]
        if improving:
            suggestions.append(f"继续保持：{improving[0]} 进步中！")
        
        # 一直没练的弱项
        never_practiced_weak = [
            t for t in p.weaknesses
            if t not in p.topic_stats or p.topic_stats[t]["count"] < 2
        ]
        if never_practiced_weak:
            suggestions.append(f"建议开始练习：{never_practiced_weak[0]}")
        
        return "；".join(suggestions) if suggestions else "继续保持当前练习节奏！"
    
    # ── Import from LLM Report ──────────────────────────────────────────────
    
    @staticmethod
    def parse_llm_report(report_text: str) -> Dict[str, Any]:
        """
        从 LLM 生成的报告文本中解析出结构化评分
        用于从 _generate_interview_report 的输出中提取数据
        
        这是防御性解析，如果解析失败返回默认值
        """
        result = {
            "overall_score": 3.0,
            "technical_depth": 3.0,
            "communication": 3.0,
            "project_experience": 3.0,
            "problem_solving": 3.0,
            "topic_scores": [],
            "summary": ""
        }
        
        try:
            # 尝试解析 overall score（找 "综合评分" 或 "Overall" 附近的数字）
            import re
            overall_match = re.search(r'(?:综合评分?|Overall|总评分?)[:：]?\s*(\d+\.?\d*)/?5', report_text, re.IGNORECASE)
            if overall_match:
                result["overall_score"] = float(overall_match.group(1))
            
            # 尝试解析各维度（简单关键词匹配）
            for dim, key in [
                ("technical_depth", ["技术深度", "技术能力"]),
                ("communication", ["表达", "沟通"]),
                ("project_experience", ["项目经验"]),
                ("problem_solving", ["临场反应", "问题解决"])
            ]:
                for kw in key:
                    if kw in report_text:
                        # 找附近数字
                        m = re.search(rf'{kw}[:：]?\s*(\d+\.?\d*)', report_text)
                        if m:
                            result[dim] = float(m.group(1))
                            break
            
            # 提 summary（取第一段非标题文字，最多100字）
            lines = [l.strip() for l in report_text.split("\n") if l.strip() and not l.startswith("#")]
            if lines:
                result["summary"] = lines[0][:100]
            
        except Exception:
            pass
        
        return result


# ── Singleton ──────────────────────────────────────────────────────────────────

_memory_instance: Optional[EvaluationMemory] = None

def get_evaluation_memory() -> EvaluationMemory:
    """获取单例"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = EvaluationMemory()
    return _memory_instance