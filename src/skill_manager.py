"""
SkillManager - 统一 Skill 加载管理器

功能：
- 统一的 Skill 访问接口
- 支持 workflow/calls/sub_agents 增强字段
- 通过 config/skill_config.json 配置

使用方式：
    from src.skill_manager import get_skill_manager
    
    sm = get_skill_manager()
    
    # 获取 Skill 函数
    skill_fn = sm.get_skill(intent_mode="mock_interview")
    
    # 获取 workflow（自然语言执行步骤）
    workflow = sm.get_workflow("mock_interview")
    
    # 获取 skill chaining 规则
    calls = sm.get_calls("mock_interview")
    
    # 获取 sub_agents 定义
    agents = sm.get_sub_agents("mock_interview")
"""

import json
import re
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List


class SkillCall:
    """Skill 调用规则"""
    def __init__(self, skill: str, trigger: str):
        self.skill = skill
        self.trigger = trigger


class SubAgent:
    """子 Agent 定义"""
    def __init__(self, name: str, role: str, tools: List[str] = None):
        self.name = name
        self.role = role
        self.tools = tools or []


class SkillInfo:
    """Skill 信息"""
    def __init__(self, skill_id: str, path: str):
        self.skill_id = skill_id
        self.path = path
        self.name: str = ""
        self.description: str = ""
        self.workflow: str = ""
        self.intent_mode: str = ""
        self.question_type: str = ""
        self.fallback_for: str = ""
        self.type: str = "node"
        self.entry: str = ""
        self.mcp_tool: str = ""
        self.calls: List[SkillCall] = []
        self.sub_agents: List[SubAgent] = []


class SkillLoader:
    """
    Skill 加载器 - 支持 DeerFlow 风格 workflow/calls/sub_agents
    """
    
    _instance: Optional["SkillLoader"] = None
    
    def __new__(cls, skills_dir: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(skills_dir)
        return cls._instance
    
    def _init(self, skills_dir: str = None):
        if skills_dir is None:
            skills_dir = Path(__file__).parent.parent / "skills"
        self.skills_dir = Path(skills_dir)
        self._registry: Dict[str, SkillInfo] = {}
        self._cache: Dict[str, Callable] = {}
        self._scanned = False
    
    def scan(self) -> Dict[str, SkillInfo]:
        self._registry = {}
        
        if not self.skills_dir.exists():
            print(f"[SkillLoader] skills 目录不存在: {self.skills_dir}")
            return self._registry
        
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            
            skill_id = skill_dir.name
            try:
                content = skill_md.read_text(encoding="utf-8")
                skill_info = self._parse(content, skill_id, str(skill_dir))
                self._registry[skill_id] = skill_info
                
                flags = []
                if skill_info.workflow: flags.append("workflow")
                if skill_info.calls: flags.append("calls")
                if skill_info.sub_agents: flags.append("sub_agents")
                flag_str = f" [+{', '.join(flags)}]" if flags else ""
                print(f"[SkillLoader] ✅ 注册 Skill: {skill_id}{flag_str}")
            except Exception as e:
                print(f"[SkillLoader] ❌ 解析 Skill 失败 {skill_id}: {e}")
        
        self._scanned = True
        print(f"[SkillLoader] 扫描完成，共 {len(self._registry)} 个 Skill")
        return self._registry
    
    def _parse(self, content: str, skill_id: str, path: str) -> SkillInfo:
        info = SkillInfo(skill_id, path)
        
        name_match = re.search(r"^#\s*Skill:\s*(\S+)", content, re.MULTILINE)
        if name_match:
            info.name = name_match.group(1)
        
        desc_match = re.search(r"##\s*描述\s*\n(.+?)(?=##|\Z)", content, re.DOTALL)
        if desc_match:
            info.description = desc_match.group(1).strip()
        
        # 工作流
        wf_match = re.search(r"##\s*工作流[（(]DeerFlow风格[)）]?\s*\n```?\n?(.*?)```?\n*(?=##|\Z)",
                             content, re.DOTALL)
        if not wf_match:
            wf_match = re.search(r"##\s*工作流\s*\n(.+?)(?=##|\Z)", content, re.DOTALL)
        if wf_match:
            info.workflow = wf_match.group(1).strip()
        
        # 触发条件
        intent_match = re.search(r"intent_mode:\s*(\S+)", content)
        if intent_match:
            info.intent_mode = intent_match.group(1).strip()
        
        question_match = re.search(r"question_type:\s*(\S+)", content)
        if question_match:
            info.question_type = question_match.group(1).strip()
        
        fallback_match = re.search(r"fallback_for:\s*(\S+)", content)
        if fallback_match:
            info.fallback_for = fallback_match.group(1).strip()
        
        # 实现
        type_match = re.search(r"type:\s*(\S+)", content)
        if type_match:
            info.type = type_match.group(1).strip()
        
        entry_match = re.search(r"entry:\s*(\S+)", content)
        if entry_match:
            info.entry = entry_match.group(1).strip()
        
        mcp_match = re.search(r"mcp_tool:\s*(\S+)", content)
        if mcp_match:
            info.mcp_tool = mcp_match.group(1).strip()
        
        # 可调用子技能
        calls_match = re.search(r"##\s*可调用子技能\s*\n```?\n?(.*?)\n?```?\n*(?=##|\Z)",
                                 content, re.DOTALL)
        if calls_match:
            info.calls = self._parse_calls(calls_match.group(1))
        
        # 子Agent定义
        agents_match = re.search(r"##\s*子Agent定义\s*\n```?\n?(.*?)\n?```?\n*(?=##|\Z)",
                                  content, re.DOTALL)
        if agents_match:
            info.sub_agents = self._parse_sub_agents(agents_match.group(1))
        
        return info
    
    def _parse_calls(self, content: str) -> List[SkillCall]:
        calls = []
        current_skill = None
        for line in content.split("\n"):
            line = line.strip()
            if not line or line == 'calls:' or line.startswith('```'):
                continue
            skill_match = re.search(r"^-?\s*skill:\s*(.+)", line)
            if skill_match:
                current_skill = skill_match.group(1).strip()
                continue
            trigger_match = re.search(r"^\s*trigger:\s*(.+)", line)
            if trigger_match and current_skill:
                calls.append(SkillCall(skill=current_skill, trigger=trigger_match.group(1).strip()))
                current_skill = None
        return calls
    
    def _parse_sub_agents(self, content: str) -> List[SubAgent]:
        agents = []
        current = {}
        for line in content.split("\n"):
            name_match = re.search(r"-?\s*name:\s*(.+)", line)
            role_match = re.search(r"role:\s*(.+)", line)
            tools_match = re.search(r"tools:\s*\[(.+)\]", line)
            if name_match:
                if current.get("name"):
                    agents.append(SubAgent(**current))
                current = {"name": name_match.group(1).strip(), "tools": []}
            elif role_match and current:
                current["role"] = role_match.group(1).strip()
            elif tools_match and current:
                current["tools"] = [t.strip() for t in tools_match.group(1).split(",")]
        if current.get("name"):
            agents.append(SubAgent(**current))
        return agents
    
    def match(self, intent_mode: str = None, question_type: str = None,
              fallback_for: str = None) -> Optional[str]:
        if not self._scanned:
            self.scan()
        for skill_id, info in self._registry.items():
            if intent_mode and info.intent_mode == intent_mode:
                return skill_id
            if question_type and info.question_type == question_type:
                return skill_id
            if fallback_for and info.fallback_for == fallback_for:
                return skill_id
        return None
    
    def get_skill_info(self, skill_id: str) -> Optional[SkillInfo]:
        return self._registry.get(skill_id)
    
    def load(self, skill_id: str) -> Optional[Callable]:
        if not self._scanned:
            self.scan()
        if skill_id in self._cache:
            return self._cache[skill_id]
        
        skill = self._registry.get(skill_id)
        if not skill:
            print(f"[SkillLoader] Skill 未注册: {skill_id}")
            return None
        
        if skill.type not in ("node", "tool"):
            print(f"[SkillLoader] 类型 {skill.type} 不支持代码加载")
            return None
        
        if not skill.entry:
            print(f"[SkillLoader] Skill {skill_id} 无 entry")
            return None
        
        try:
            module_path, func_name = skill.entry.rsplit(".", 1)
            module = importlib.import_module(module_path)
            fn = getattr(module, func_name)
            self._cache[skill_id] = fn
            print(f"[SkillLoader] ✅ 加载 Skill: {skill_id} → {skill.entry}")
            return fn
        except Exception as e:
            print(f"[SkillLoader] ❌ 加载 Skill 失败 {skill_id}: {e}")
            return None
    
    def list_skills(self) -> Dict[str, SkillInfo]:
        if not self._scanned:
            self.scan()
        return dict(self._registry)
    
    def reload(self):
        self._cache.clear()
        self._registry.clear()
        self._scanned = False
        self.scan()


# ── SkillManager ──────────────────────────────────────────────────────────────

class SkillManager:
    """
    统一 Skill 加载管理器
    
    所有 Skills 通过这个入口访问。
    支持 DeerFlow 风格的 workflow/calls/sub_agents 字段。
    """
    
    _instance: Optional["SkillManager"] = None
    
    def __init__(self, skills_dir: str = None):
        self._loader = SkillLoader(skills_dir)
        self._loader.scan()
    
    def get_skill(self, intent_mode: str = None, question_type: str = None,
                  fallback_for: str = None) -> Optional[Callable]:
        skill_id = self._loader.match(
            intent_mode=intent_mode,
            question_type=question_type,
            fallback_for=fallback_for
        )
        if skill_id:
            return self._loader.load(skill_id)
        return None
    
    def match(self, intent_mode: str = None, question_type: str = None,
              fallback_for: str = None) -> Optional[str]:
        return self._loader.match(
            intent_mode=intent_mode,
            question_type=question_type,
            fallback_for=fallback_for
        )
    
    def get_skill_info(self, skill_id: str) -> Optional[SkillInfo]:
        return self._loader.get_skill_info(skill_id)
    
    def list_skills(self) -> Dict[str, SkillInfo]:
        return self._loader.list_skills()
    
    def get_workflow(self, skill_id: str) -> str:
        info = self._loader.get_skill_info(skill_id)
        return info.workflow if info else ""
    
    def get_calls(self, skill_id: str) -> List[SkillCall]:
        info = self._loader.get_skill_info(skill_id)
        return info.calls if info else []
    
    def get_sub_agents(self, skill_id: str) -> List[SubAgent]:
        info = self._loader.get_skill_info(skill_id)
        return info.sub_agents if info else []
    
    def reload(self):
        self._loader.reload()


# 全局单例
_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager