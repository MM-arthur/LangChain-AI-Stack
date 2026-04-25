# Skill Loader - 动态技能加载系统
# 启动时扫描 skills/ 目录，按需加载 Skill

import json
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import re


class SkillLoader:
    """
    Skill 加载器 - 实现按需动态加载 Skill

    使用方式：
        loader = SkillLoader()
        loader.scan()                    # 启动时扫描，构建 registry
        skill_fn = loader.match(intent_mode="mock_interview")  # 匹配 skill
        result = skill_fn(state)         # 执行
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
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, Callable] = {}
        self._scanned = False

    def scan(self) -> Dict[str, Dict[str, Any]]:
        """
        扫描 skills/ 目录，构建 Skill Registry

        Returns:
            {skill_id: skill_info} 字典
        """
        self._registry = {}

        if not self.skills_dir.exists():
            print(f"[SkillLoader] skills 目录不存在: {self.skills_dir}")
            return self._registry

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md_path = skill_dir / "SKILL.md"
            if not skill_md_path.exists():
                continue

            skill_id = skill_dir.name
            try:
                skill_info = self._parse_skill_md(skill_md_path.read_text(encoding="utf-8"))
                skill_info["_skill_id"] = skill_id
                skill_info["_path"] = str(skill_dir)
                self._registry[skill_id] = skill_info
                print(f"[SkillLoader] ✅ 注册 Skill: {skill_id}")
            except Exception as e:
                print(f"[SkillLoader] ❌ 解析 Skill 失败 {skill_id}: {e}")

        self._scanned = True
        print(f"[SkillLoader] 扫描完成，共 {len(self._registry)} 个 Skill")
        return self._registry

    def _parse_skill_md(self, content: str) -> Dict[str, Any]:
        """
        解析 SKILL.md 文件，提取 skill 元信息

        格式：
            # Skill: mock_interview
            ## 概述
            ...
            ## 触发条件
            intent_mode: mock_interview
            ## 实现
            type: node
            entry: src.nodes.career_intents.mock_interview
        """
        info = {}

        # 提取 skill 名称（第一行 # Skill: xxx）
        name_match = re.search(r"^#\s*Skill:\s*(\S+)", content, re.MULTILINE)
        if name_match:
            info["name"] = name_match.group(1)

        # 提取描述（## 概述 到下一个 ## 之间的内容）
        desc_match = re.search(r"##\s*概述\s*\n(.+?)(?=##|\Z)", content, re.DOTALL)
        if desc_match:
            info["description"] = desc_match.group(1).strip()

        # 提取触发条件（intent_mode / question_type）
        intent_match = re.search(r"intent_mode:\s*(\S+)", content)
        if intent_match:
            info["intent_mode"] = intent_match.group(1).strip()

        question_type_match = re.search(r"question_type:\s*(\S+)", content)
        if question_type_match:
            info["question_type"] = question_type_match.group(1).strip()

        fallback_match = re.search(r"fallback_for:\s*(\S+)", content)
        if fallback_match:
            info["fallback_for"] = fallback_match.group(1).strip()

        # 提取实现信息（type / entry）
        type_match = re.search(r"type:\s*(\S+)", content)
        if type_match:
            info["type"] = type_match.group(1).strip()

        entry_match = re.search(r"entry:\s*(\S+)", content)
        if entry_match:
            info["entry"] = entry_match.group(1).strip()

        # 提取 MCP 工具名
        mcp_match = re.search(r"mcp_tool:\s*(\S+)", content)
        if mcp_match:
            info["mcp_tool"] = mcp_match.group(1).strip()

        return info

    def match(self, intent_mode: str = None, question_type: str = None,
             fallback_for: str = None) -> Optional[str]:
        """
        根据意图匹配 Skill ID

        Args:
            intent_mode: 意图模式，如 "mock_interview"
            question_type: 问题类型，如 "最新知识"
            fallback_for: 降级目标，如 "rag_processing"

        Returns:
            匹配的 skill_id，或 None
        """
        if not self._scanned:
            self.scan()

        for skill_id, info in self._registry.items():
            if intent_mode and info.get("intent_mode") == intent_mode:
                return skill_id
            if question_type and info.get("question_type") == question_type:
                return skill_id
            if fallback_for and info.get("fallback_for") == fallback_for:
                return skill_id

        return None

    def get_skill_info(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """获取 Skill 的元信息"""
        return self._registry.get(skill_id)

    def load(self, skill_id: str) -> Optional[Callable]:
        """
        按需加载 Skill，返回可调用对象

        Args:
            skill_id: Skill ID

        Returns:
            Skill 的入口函数 / Tool 对象
        """
        if not self._scanned:
            self.scan()

        if skill_id in self._cache:
            return self._cache[skill_id]

        skill_info = self._registry.get(skill_id)
        if not skill_info:
            print(f"[SkillLoader] Skill 未注册: {skill_id}")
            return None

        skill_type = skill_info.get("type", "node")
        entry = skill_info.get("entry", "")

        try:
            if skill_type == "node":
                # Node 类型：动态 import 函数
                module_path, func_name = entry.rsplit(".", 1)
                module = importlib.import_module(module_path)
                fn = getattr(module, func_name)
                self._cache[skill_id] = fn
                print(f"[SkillLoader] ✅ 加载 Node Skill: {skill_id} → {entry}")
                return fn

            elif skill_type == "mcp":
                # MCP 类型：调 MCPToolLoader 获取工具
                from src.mcp_client import get_mcp_tool_loader
                mcp_tool_name = skill_info.get("mcp_tool", skill_id)
                loader = get_mcp_tool_loader()
                tools = loader.load_tools([mcp_tool_name])
                if tools:
                    tool = next((t for t in tools if t.name == mcp_tool_name), tools[0])
                    self._cache[skill_id] = tool
                    print(f"[SkillLoader] ✅ 加载 MCP Skill: {skill_id} → {mcp_tool_name}")
                    return tool
                return None

            elif skill_type == "tool":
                # Tool 类型：动态 import
                module_path, func_name = entry.rsplit(".", 1)
                module = importlib.import_module(module_path)
                fn = getattr(module, func_name)
                self._cache[skill_id] = fn
                return fn

            else:
                print(f"[SkillLoader] ❓ 未知 Skill 类型: {skill_type}")
                return None

        except Exception as e:
            print(f"[SkillLoader] ❌ 加载 Skill 失败 {skill_id}: {e}")
            return None

    def list_skills(self) -> Dict[str, Dict[str, Any]]:
        """列出所有已注册的 Skill"""
        if not self._scanned:
            self.scan()
        return dict(self._registry)

    def reload(self):
        """重新扫描 skills/ 目录，清空缓存"""
        self._cache.clear()
        self._registry.clear()
        self._scanned = False
        self.scan()


# 全局单例
_skill_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    global _skill_loader
    if _skill_loader is None:
        _skill_loader = SkillLoader()
        _skill_loader.scan()
    return _skill_loader


# 便捷函数：按 intent_mode 快速获取 Skill 函数
def get_skill(intent_mode: str = None, question_type: str = None,
              fallback_for: str = None) -> Optional[Callable]:
    """
    快捷函数：根据意图获取对应的 Skill 函数

    用法：
        skill_fn = get_skill(intent_mode="mock_interview")
        if skill_fn:
            result = skill_fn(state)
    """
    loader = get_skill_loader()
    skill_id = loader.match(intent_mode=intent_mode,
                           question_type=question_type,
                           fallback_for=fallback_for)
    if skill_id:
        return loader.load(skill_id)
    return None