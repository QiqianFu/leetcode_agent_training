"""
Generate synthetic SFT training data for the LeetCode Agent.

Creates diverse tool-calling trajectories covering all 17 tools,
formatted as multi-turn conversations in Qwen's chat template format.

Each sample is a conversation: user request -> assistant think + tool_call -> tool result -> assistant response.
We generate 3 types of trajectories:
  1. Direct call (model knows what to do)
  2. Multi-step (search/check first, then act)
  3. Error recovery (wrong tool -> correct tool)
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

# ─── Tool schemas (from leetcode_agent) ───
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "check_problem", "description": "按题号查询题目信息。返回题目元信息和是否有记忆文件。", "parameters": {"type": "object", "properties": {"problem_id": {"type": "integer", "description": "题目编号"}}, "required": ["problem_id"]}}},
    {"type": "function", "function": {"name": "read_solution", "description": "读取用户的解题代码文件。可传 file_path 或 problem_id（二选一），传 problem_id 时自动查找文件。", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "解题文件路径"}, "problem_id": {"type": "integer", "description": "题目编号（与 file_path 二选一）"}}}},},
    {"type": "function", "function": {"name": "find_problem_file", "description": "在当前工作区内按题号查找本地解题文件。", "parameters": {"type": "object", "properties": {"problem_id": {"type": "integer", "description": "题目编号"}}, "required": ["problem_id"]}}},
    {"type": "function", "function": {"name": "search_workspace_files", "description": "在当前工作区内按文件名关键词搜索本地题目文件。", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "文件名或题目关键词"}}, "required": ["keyword"]}}},
    {"type": "function", "function": {"name": "list_category_problems", "description": "列出当前工作区内某个分类文件夹下的题目文件。", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "分类/文件夹名，如 dp、graph、tree、string"}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "append_solution", "description": "将参考解法追加到用户的解题文件末尾。", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "解题文件路径"}, "content": {"type": "string", "description": "参考解法代码"}}, "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "pick_problem", "description": "从 CodeTop 高频题库推荐题目供用户选择。", "parameters": {"type": "object", "properties": {"tag": {"type": "string", "description": "标签筛选，如 dp, graph"}, "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"], "description": "难度筛选"}}}}},
    {"type": "function", "function": {"name": "search_problem", "description": "用英文关键词搜索 LeetCode 题目。", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "英文搜索关键词"}}, "required": ["keyword"]}}},
    {"type": "function", "function": {"name": "start_problem", "description": "开始做指定题号的 LeetCode 题。", "parameters": {"type": "object", "properties": {"problem_id": {"type": "integer", "description": "题目编号"}}, "required": ["problem_id"]}}},
    {"type": "function", "function": {"name": "read_memory", "description": "读取某道题的记忆文件内容。", "parameters": {"type": "object", "properties": {"problem_id": {"type": "integer", "description": "题目编号"}}, "required": ["problem_id"]}}},
    {"type": "function", "function": {"name": "write_memory", "description": "写入或追加内容到某道题的记忆文件。", "parameters": {"type": "object", "properties": {"problem_id": {"type": "integer", "description": "题目编号"}, "content": {"type": "string", "description": "要写入的内容"}, "mode": {"type": "string", "enum": ["append", "overwrite"], "description": "写入模式"}}, "required": ["problem_id", "content"]}}},
    {"type": "function", "function": {"name": "get_daily_plan", "description": "生成今日刷题计划。", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_hot_problems", "description": "查看高频题列表。", "parameters": {"type": "object", "properties": {"company": {"type": "string", "description": "公司名"}, "tag": {"type": "string", "description": "标签名"}}}}},
    {"type": "function", "function": {"name": "web_search", "description": "搜索互联网获取信息。", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}, "max_results": {"type": "integer", "description": "返回结果数量"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "update_user_memory", "description": "更新用户偏好记忆。", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "find_similar_problems", "description": "查找与当前题目算法思路相似的已做题目。", "parameters": {"type": "object", "properties": {"problem_id": {"type": "integer", "description": "当前题目编号"}}, "required": ["problem_id"]}}},
    {"type": "function", "function": {"name": "analyze_and_memorize", "description": "将当前题目的分析写入记忆文件。", "parameters": {"type": "object", "properties": {"problem_id": {"type": "integer", "description": "题目编号"}}, "required": ["problem_id"]}}},
]

# ─── Problem bank for generating diverse samples ───
PROBLEMS = [
    (1, "Two Sum", "array", "Easy"),
    (2, "Add Two Numbers", "linked-list", "Medium"),
    (3, "Longest Substring Without Repeating Characters", "string", "Medium"),
    (5, "Longest Palindromic Substring", "dp", "Medium"),
    (15, "3Sum", "array", "Medium"),
    (20, "Valid Parentheses", "stack", "Easy"),
    (21, "Merge Two Sorted Lists", "linked-list", "Easy"),
    (23, "Merge k Sorted Lists", "heap", "Hard"),
    (33, "Search in Rotated Sorted Array", "binary-search", "Medium"),
    (42, "Trapping Rain Water", "stack", "Hard"),
    (46, "Permutations", "backtracking", "Medium"),
    (53, "Maximum Subarray", "dp", "Medium"),
    (56, "Merge Intervals", "array", "Medium"),
    (70, "Climbing Stairs", "dp", "Easy"),
    (72, "Edit Distance", "dp", "Medium"),
    (76, "Minimum Window Substring", "sliding-window", "Hard"),
    (78, "Subsets", "backtracking", "Medium"),
    (94, "Binary Tree Inorder Traversal", "tree", "Easy"),
    (98, "Validate Binary Search Tree", "tree", "Medium"),
    (101, "Symmetric Tree", "tree", "Easy"),
    (102, "Binary Tree Level Order Traversal", "tree", "Medium"),
    (104, "Maximum Depth of Binary Tree", "tree", "Easy"),
    (121, "Best Time to Buy and Sell Stock", "dp", "Easy"),
    (124, "Binary Tree Maximum Path Sum", "tree", "Hard"),
    (128, "Longest Consecutive Sequence", "array", "Medium"),
    (136, "Single Number", "bit-manipulation", "Easy"),
    (141, "Linked List Cycle", "linked-list", "Easy"),
    (146, "LRU Cache", "design", "Medium"),
    (148, "Sort List", "linked-list", "Medium"),
    (152, "Maximum Product Subarray", "dp", "Medium"),
    (155, "Min Stack", "stack", "Medium"),
    (160, "Intersection of Two Linked Lists", "linked-list", "Easy"),
    (169, "Majority Element", "array", "Easy"),
    (198, "House Robber", "dp", "Medium"),
    (200, "Number of Islands", "graph", "Medium"),
    (206, "Reverse Linked List", "linked-list", "Easy"),
    (207, "Course Schedule", "graph", "Medium"),
    (215, "Kth Largest Element in an Array", "heap", "Medium"),
    (221, "Maximal Square", "dp", "Medium"),
    (226, "Invert Binary Tree", "tree", "Easy"),
    (234, "Palindrome Linked List", "linked-list", "Easy"),
    (236, "Lowest Common Ancestor of a Binary Tree", "tree", "Medium"),
    (238, "Product of Array Except Self", "array", "Medium"),
    (239, "Sliding Window Maximum", "sliding-window", "Hard"),
    (240, "Search a 2D Matrix II", "binary-search", "Medium"),
    (279, "Perfect Squares", "dp", "Medium"),
    (283, "Move Zeroes", "array", "Easy"),
    (287, "Find the Duplicate Number", "binary-search", "Medium"),
    (297, "Serialize and Deserialize Binary Tree", "tree", "Hard"),
    (300, "Longest Increasing Subsequence", "dp", "Medium"),
    (301, "Remove Invalid Parentheses", "backtracking", "Hard"),
    (309, "Best Time to Buy and Sell Stock with Cooldown", "dp", "Medium"),
    (312, "Burst Balloons", "dp", "Hard"),
    (322, "Coin Change", "dp", "Medium"),
    (337, "House Robber III", "tree", "Medium"),
    (338, "Counting Bits", "dp", "Easy"),
    (347, "Top K Frequent Elements", "heap", "Medium"),
    (394, "Decode String", "stack", "Medium"),
    (399, "Evaluate Division", "graph", "Medium"),
    (406, "Queue Reconstruction by Height", "greedy", "Medium"),
    (416, "Partition Equal Subset Sum", "dp", "Medium"),
    (437, "Path Sum III", "tree", "Medium"),
    (438, "Find All Anagrams in a String", "sliding-window", "Medium"),
    (448, "Find All Numbers Disappeared in an Array", "array", "Easy"),
    (460, "LFU Cache", "design", "Hard"),
    (494, "Target Sum", "dp", "Medium"),
    (543, "Diameter of Binary Tree", "tree", "Easy"),
    (560, "Subarray Sum Equals K", "array", "Medium"),
    (581, "Shortest Unsorted Continuous Subarray", "array", "Medium"),
    (617, "Merge Two Binary Trees", "tree", "Easy"),
    (621, "Task Scheduler", "greedy", "Medium"),
    (647, "Palindromic Substrings", "dp", "Medium"),
    (739, "Daily Temperatures", "stack", "Medium"),
    (763, "Partition Labels", "greedy", "Medium"),
]

CATEGORIES = ["dp", "tree", "graph", "array", "linked-list", "stack", "heap",
              "binary-search", "sliding-window", "backtracking", "design",
              "greedy", "bit-manipulation", "string"]

COMPANIES = ["google", "meta", "amazon", "microsoft", "apple", "bytedance", "tencent"]

# ─── Trajectory templates ───

def _tool_call_id():
    return f"call_{random.randint(100000, 999999)}"


def make_direct_start_problem(prob):
    """User explicitly asks to do a problem by number -> direct start_problem."""
    pid, name, cat, diff = prob
    user_msgs = [
        f"我要做第 {pid} 题",
        f"开始做 {pid} 题吧",
        f"来做 LeetCode {pid}",
        f"帮我开始 {pid} 题 {name}",
        f"做一下 {pid}",
    ]
    user = random.choice(user_msgs)

    tc_id_start = _tool_call_id()
    tc_id_similar = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_start, "type": "function", "function": {"name": "start_problem", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_start,
            "content": json.dumps({"problem_id": pid, "title": name, "difficulty": diff, "file": f"{cat}/{pid}_{name.lower().replace(' ', '_')}.py", "memory_file": f".memories/{pid}_{name.lower().replace(' ', '_')}.md"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_similar, "type": "function", "function": {"name": "find_similar_problems", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_similar,
            "content": json.dumps({"similar_problems": [], "message": "暂无相似题目记录"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"已为你创建 {pid}. {name} 的解题文件。这是一道 {diff} 难度的{cat}题。\n\n你可以先想想思路，有问题随时问我！",
        },
    ]
    return messages


def make_check_then_start(prob):
    """User mentions a problem -> check_problem first, then start_problem."""
    pid, name, cat, diff = prob
    user_msgs = [
        f"146 题是什么来着？我想做一下",
        f"{pid} 题我之前做过吗？想再做一遍",
        f"帮我看看 {pid} 题的信息",
    ]
    # Replace 146 with actual pid
    user = random.choice(user_msgs).replace("146", str(pid))

    tc_id_check = _tool_call_id()
    tc_id_start = _tool_call_id()
    tc_id_similar = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_check, "type": "function", "function": {"name": "check_problem", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_check,
            "content": json.dumps({"problem_id": pid, "title": name, "difficulty": diff, "category": cat, "has_memory": random.choice([True, False])}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"{pid}. {name} 是一道 {diff} 难度的{cat}题。我来帮你开始。",
            "tool_calls": [{"id": tc_id_start, "type": "function", "function": {"name": "start_problem", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_start,
            "content": json.dumps({"problem_id": pid, "title": name, "difficulty": diff, "file": f"{cat}/{pid}_{name.lower().replace(' ', '_')}.py"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_similar, "type": "function", "function": {"name": "find_similar_problems", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_similar,
            "content": json.dumps({"similar_problems": [], "message": "暂无相似题目记录"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"已创建解题文件，你可以开始做了！这道题考察的是{cat}相关知识，想想怎么入手？",
        },
    ]
    return messages


def make_search_then_start(prob):
    """User asks for a type of problem -> search_problem, then start."""
    pid, name, cat, diff = prob
    tag_zh = {"dp": "动态规划", "tree": "树", "graph": "图", "array": "数组",
              "linked-list": "链表", "stack": "栈", "heap": "堆",
              "binary-search": "二分查找", "sliding-window": "滑动窗口",
              "backtracking": "回溯", "design": "设计", "greedy": "贪心",
              "bit-manipulation": "位运算", "string": "字符串"}.get(cat, cat)

    user_msgs = [
        f"来一道{tag_zh}的题",
        f"我想练习{tag_zh}，推荐一道",
        f"有什么{tag_zh}类型的题目吗",
    ]
    user = random.choice(user_msgs)
    tc_id_search = _tool_call_id()
    tc_id_start = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_search, "type": "function", "function": {"name": "search_problem", "arguments": json.dumps({"keyword": cat.replace("-", " ")})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_search,
            "content": json.dumps({"selected_id": pid, "selected_title": name, "difficulty": diff}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_start, "type": "function", "function": {"name": "start_problem", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_start,
            "content": json.dumps({"problem_id": pid, "title": name, "difficulty": diff, "file": f"{cat}/{pid}_{name.lower().replace(' ', '_')}.py"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"已选择 {pid}. {name}（{diff}），解题文件已创建。开始思考吧！",
        },
    ]
    return messages


def make_pick_problem():
    """User wants a random problem -> pick_problem."""
    prob = random.choice(PROBLEMS)
    pid, name, cat, diff = prob
    user_msgs = [
        "来一道题吧",
        "随便推荐一道题",
        "开始刷题",
        "今天做什么题好",
        "帮我选一道题",
    ]
    user = random.choice(user_msgs)
    tc_id_pick = _tool_call_id()
    tc_id_start = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_pick, "type": "function", "function": {"name": "pick_problem", "arguments": json.dumps({})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_pick,
            "content": json.dumps({"selected_id": pid, "selected_title": name, "difficulty": diff}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_start, "type": "function", "function": {"name": "start_problem", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_start,
            "content": json.dumps({"problem_id": pid, "title": name, "difficulty": diff, "file": f"{cat}/{pid}_{name.lower().replace(' ', '_')}.py"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"推荐你做 {pid}. {name}，{diff} 难度。文件已创建，开始吧！",
        },
    ]
    return messages


def make_pick_with_tag():
    """User wants a problem with specific tag/difficulty -> pick_problem with params."""
    cat = random.choice(CATEGORIES)
    diff = random.choice(["Easy", "Medium", "Hard"])
    prob = random.choice([p for p in PROBLEMS if p[2] == cat] or [random.choice(PROBLEMS)])
    pid, name, _, _ = prob
    tag_zh = {"dp": "动态规划", "tree": "树", "graph": "图", "array": "数组",
              "linked-list": "链表", "stack": "栈", "heap": "堆"}.get(cat, cat)

    user_msgs = [
        f"来一道 {diff} 难度的{tag_zh}题",
        f"推荐一道{tag_zh}的 {diff} 题",
    ]
    user = random.choice(user_msgs)
    tc_id = _tool_call_id()
    tc_id_start = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "pick_problem", "arguments": json.dumps({"tag": cat, "difficulty": diff})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"selected_id": pid, "selected_title": name, "difficulty": diff}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_start, "type": "function", "function": {"name": "start_problem", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_start,
            "content": json.dumps({"problem_id": pid, "title": name, "difficulty": diff, "file": f"{cat}/{pid}_{name.lower().replace(' ', '_')}.py"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"推荐 {pid}. {name}（{diff}），开始做吧！",
        },
    ]
    return messages


def make_read_and_analyze(prob):
    """User finished a problem -> read solution, then analyze_and_memorize."""
    pid, name, cat, diff = prob
    file_path = f"{cat}/{pid}_{name.lower().replace(' ', '_')}.py"
    user_msgs = [
        f"我 {pid} 题做完了，帮我看看",
        f"做完了，帮我总结一下 {pid} 题",
        f"我写好了，看看我的代码",
    ]
    user = random.choice(user_msgs)
    tc_id_read = _tool_call_id()
    tc_id_analyze = _tool_call_id()

    solution_code = f'''class Solution:
    def solve(self, nums):
        # Solution for {name}
        pass
'''

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_read, "type": "function", "function": {"name": "read_solution", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_read,
            "content": json.dumps({"file_path": file_path, "content": solution_code}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"看到你的代码了。思路不错！让我帮你记录一下总结。",
            "tool_calls": [{"id": tc_id_analyze, "type": "function", "function": {"name": "analyze_and_memorize", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_analyze,
            "content": json.dumps({"status": "ok", "message": f"已更新 {pid}. {name} 的记忆文件"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"已为 {pid}. {name} 生成做题总结并保存。下次复习时可以直接查看。还要继续做题吗？",
        },
    ]
    return messages


def make_review_memory(prob):
    """User wants to review a problem -> read_memory."""
    pid, name, cat, diff = prob
    user_msgs = [
        f"帮我回顾一下 {pid} 题",
        f"我之前 {pid} 题是怎么做的",
        f"看看 {pid} 题的笔记",
    ]
    user = random.choice(user_msgs)
    tc_id = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "read_memory", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"problem_id": pid, "title": name, "content": f"# {pid}. {name}\n\n## 思路\n使用{cat}方法解决\n\n## 要点\n- 关键点1\n- 关键点2\n\n## 复杂度\n- 时间: O(n)\n- 空间: O(n)"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"这是你之前做 {pid}. {name} 的笔记：\n\n这道题用的是{cat}方法。要不要再做一遍巩固一下？",
        },
    ]
    return messages


def make_daily_plan():
    """User asks for daily plan -> get_daily_plan."""
    user_msgs = [
        "今天做什么题",
        "给我安排一下今天的计划",
        "今日计划",
    ]
    user = random.choice(user_msgs)
    tc_id = _tool_call_id()

    probs = random.sample(PROBLEMS, 3)
    plan = [{"problem_id": p[0], "title": p[1], "difficulty": p[3], "reason": "高频题"} for p in probs]

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "get_daily_plan", "arguments": json.dumps({})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"plan": plan}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"今日推荐：\n\n1. {probs[0][0]}. {probs[0][1]}（{probs[0][3]}）\n2. {probs[1][0]}. {probs[1][1]}（{probs[1][3]}）\n3. {probs[2][0]}. {probs[2][1]}（{probs[2][3]}）\n\n要从哪道开始？",
        },
    ]
    return messages


def make_hot_problems():
    """User asks for hot problems -> get_hot_problems."""
    company = random.choice(COMPANIES)
    user_msgs = [
        f"看看 {company} 的高频题",
        f"{company} 面试常考什么题",
        "看看高频题列表",
    ]
    user = random.choice(user_msgs)
    tc_id = _tool_call_id()

    probs = random.sample(PROBLEMS, 5)
    hot = [{"problem_id": p[0], "title": p[1], "frequency": random.randint(50, 100)} for p in probs]

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "get_hot_problems", "arguments": json.dumps({"company": company})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"problems": hot}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"以下是 {company} 的高频面试题：\n\n" + "\n".join(f"- {p['problem_id']}. {p['title']}（频率 {p['frequency']}%）" for p in hot) + "\n\n想做哪一道？",
        },
    ]
    return messages


def make_web_search():
    """User asks about algorithm knowledge -> web_search."""
    topics = [
        ("什么是单调栈", "monotonic stack algorithm"),
        ("KMP 算法怎么理解", "KMP string matching algorithm"),
        ("拓扑排序怎么实现", "topological sort implementation"),
        ("并查集是什么", "union find data structure"),
        ("A* 算法原理", "A star pathfinding algorithm"),
        ("红黑树和 AVL 树区别", "red black tree vs AVL tree"),
        ("动态规划状态压缩", "dynamic programming bitmask"),
        ("Trie 树有什么用", "trie prefix tree applications"),
    ]
    topic_zh, query_en = random.choice(topics)
    tc_id = _tool_call_id()

    messages = [
        {"role": "user", "content": topic_zh},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "web_search", "arguments": json.dumps({"query": query_en})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"results": [{"title": f"Understanding {query_en}", "snippet": f"A {query_en} is a fundamental concept in computer science..."}]}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"关于{topic_zh}，简单来说：\n\n这是一种常用的数据结构/算法技巧。核心思想是通过特定的方式组织数据来优化查询或操作效率。\n\n要深入了解还是直接做相关的题？",
        },
    ]
    return messages


def make_workspace_search():
    """User searches workspace -> search_workspace_files or list_category_problems."""
    cat = random.choice(CATEGORIES)
    tag_zh = {"dp": "动态规划", "tree": "树", "graph": "图", "array": "数组",
              "linked-list": "链表"}.get(cat, cat)

    if random.random() < 0.5:
        # list_category_problems
        user = f"看看我 {tag_zh} 做了哪些题"
        tc_id = _tool_call_id()
        probs_in_cat = [p for p in PROBLEMS if p[2] == cat][:5]
        files = [f"{cat}/{p[0]}_{p[1].lower().replace(' ', '_')}.py" for p in probs_in_cat]

        messages = [
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "list_category_problems", "arguments": json.dumps({"category": cat})}}],
            },
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps({"category": cat, "files": files}, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": f"你在 {cat} 分类下做了 {len(files)} 道题：\n\n" + "\n".join(f"- {f}" for f in files) + "\n\n想复习哪一道？",
            },
        ]
    else:
        # search_workspace_files
        keyword = random.choice(["two sum", "palindrome", "tree", "sort", "substring"])
        user = f"我之前做过一道关于 {keyword} 的题，帮我找找"
        tc_id = _tool_call_id()
        matched = [p for p in PROBLEMS if keyword.lower() in p[1].lower()][:3]
        if not matched:
            matched = [random.choice(PROBLEMS)]
        files = [f"{p[2]}/{p[0]}_{p[1].lower().replace(' ', '_')}.py" for p in matched]

        messages = [
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "search_workspace_files", "arguments": json.dumps({"keyword": keyword})}}],
            },
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps({"keyword": keyword, "matches": files}, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": f"找到了以下相关文件：\n\n" + "\n".join(f"- {f}" for f in files) + "\n\n要看哪个？",
            },
        ]
    return messages


def make_append_solution(prob):
    """User asks for a reference solution -> find_problem_file + append_solution."""
    pid, name, cat, diff = prob
    file_path = f"{cat}/{pid}_{name.lower().replace(' ', '_')}.py"
    user_msgs = [
        f"给我 {pid} 题的参考答案",
        f"我想不出来了，给我看看 {pid} 的解法",
        f"直接告诉我 {pid} 怎么做吧",
    ]
    user = random.choice(user_msgs)
    tc_id_find = _tool_call_id()
    tc_id_append = _tool_call_id()

    solution = f'''# Reference solution for {pid}. {name}
class Solution:
    def solve(self, nums):
        # {cat} approach
        pass
'''

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id_find, "type": "function", "function": {"name": "find_problem_file", "arguments": json.dumps({"problem_id": pid})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_find,
            "content": json.dumps({"problem_id": pid, "file_path": file_path}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"好的，这道题可以用{cat}的思路来解决。我把参考解法加到你的文件里：",
            "tool_calls": [{"id": tc_id_append, "type": "function", "function": {"name": "append_solution", "arguments": json.dumps({"file_path": file_path, "content": solution})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id_append,
            "content": json.dumps({"status": "ok", "message": f"已追加到 {file_path}"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"参考解法已加到你的文件末尾。核心思路是利用{cat}的方法。有什么不理解的地方吗？",
        },
    ]
    return messages


def make_user_preference():
    """User expresses a preference -> update_user_memory."""
    preferences = [
        "我喜欢用迭代而不是递归",
        "我习惯先写暴力解再优化",
        "请不要给太多提示，让我自己想",
        "我比较擅长 DP，但链表不太熟",
        "我想重点练习 Hard 难度的题",
        "我更喜欢 Python，不要用其他语言",
        "请用中文注释，不要用英文",
        "每次做完题后自动帮我总结",
    ]
    user = random.choice(preferences)
    tc_id = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": "好的，我记住了。",
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "update_user_memory", "arguments": json.dumps({})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"status": "ok", "message": "用户偏好已更新"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"已记录你的偏好，之后会按照你的习惯来辅导。",
        },
    ]
    return messages


def make_write_memory(prob):
    """User asks to note something -> write_memory."""
    pid, name, cat, diff = prob
    notes = [
        "这道题的关键是用哈希表优化查找",
        "记住边界条件：空数组和单元素",
        "双指针从两端向中间移动",
        "需要用到前缀和的技巧",
    ]
    note = random.choice(notes)
    user = f"帮我记一下 {pid} 题：{note}"
    tc_id = _tool_call_id()

    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "write_memory", "arguments": json.dumps({"problem_id": pid, "content": note})}}],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"status": "ok", "message": f"已写入 {pid} 的记忆文件"}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"已记录到 {pid}. {name} 的笔记中。下次复习会看到。",
        },
    ]
    return messages


# ─── Generator dispatch ───

GENERATORS = [
    (make_direct_start_problem, 0.15, True),   # needs prob
    (make_check_then_start, 0.12, True),
    (make_search_then_start, 0.10, True),
    (make_pick_problem, 0.08, False),
    (make_pick_with_tag, 0.06, False),
    (make_read_and_analyze, 0.10, True),
    (make_review_memory, 0.08, True),
    (make_daily_plan, 0.04, False),
    (make_hot_problems, 0.04, False),
    (make_web_search, 0.05, False),
    (make_workspace_search, 0.06, False),
    (make_append_solution, 0.05, True),
    (make_user_preference, 0.03, False),
    (make_write_memory, 0.04, True),
]


def generate_sample():
    """Generate one training sample by weighted random selection."""
    funcs, weights, needs_prob = zip(*GENERATORS)
    idx = random.choices(range(len(funcs)), weights=weights, k=1)[0]
    fn = funcs[idx]
    if needs_prob[idx]:
        prob = random.choice(PROBLEMS)
        return fn(prob)
    else:
        return fn()


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = args.output or str(Path(__file__).parent / "sft_train.jsonl")

    samples = []
    for i in range(args.n_samples):
        messages = generate_sample()
        sample = {
            "id": f"sft_{i:05d}",
            "messages": messages,
            "tools": TOOL_SCHEMAS,
        }
        samples.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples -> {output_path}")

    # Print distribution stats
    from collections import Counter
    tool_counts = Counter()
    for s in samples:
        for msg in s["messages"]:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_counts[tc["function"]["name"]] += 1
    print("\nTool distribution:")
    for tool, count in tool_counts.most_common():
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
