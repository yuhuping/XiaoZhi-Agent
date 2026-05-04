from __future__ import annotations

import ast
import operator

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculateInput(BaseModel):
    expression: str = Field(description="四则运算表达式，支持 + - * / 与括号，如 '(123+456)*789/12'。")


def _safe_eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("仅允许数字常量。")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError("仅允许 + - * / 运算。")
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        return _ALLOWED_BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError("仅允许一元正负号。")
        return _ALLOWED_UNARY_OPS[op_type](_safe_eval_ast(node.operand))
    raise ValueError("表达式包含不被允许的语法。")


def safe_calculate(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression:
        return "计算错误: expression 不能为空。"
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval_ast(parsed)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "计算错误: 除数不能为 0。"
    except Exception as e:
        return f"计算错误: {e}"


calculate_tool = StructuredTool.from_function(
    func=safe_calculate,
    name="calculate",
    description="执行安全的四则运算表达式计算，支持 + - * / 与括号。",
    args_schema=CalculateInput,
)
