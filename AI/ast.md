


AST 抽象语法树

LLM 在对代码文本做处理的时候，需要这个来结构化理解代码，将 ast 节点作为语义块，构建知识索引；

**普通处理的暴力切分，会破坏代码的逻辑完整性**



## demo1. 展示 ast

```bash
conda install -c conda-forge graphviz
```


```python
import ast
from graphviz import Digraph

def visualize_ast_graphviz(code: str, output_file="ast_graph"):
    """
    用 Graphviz 将 Python 代码的 AST 可视化成一张 PNG 图
    """
    tree = ast.parse(code)
    dot = Digraph(comment="Python AST")
    dot.attr("node", shape="box", style="rounded,filled", color="lightblue", fontname="Helvetica")

    def add_nodes(node, parent=None):
        node_id = str(id(node))
        label = type(node).__name__

        # 展示节点名称（函数名、变量名等）
        if hasattr(node, "name"):
            label += f"\\n({node.name})"
        elif hasattr(node, "id"):
            label += f"\\n({node.id})"
        elif isinstance(node, ast.Constant):
            label += f"\\n(value={node.value})"

        dot.node(node_id, label)

        if parent:
            dot.edge(str(id(parent)), node_id)

        for child in ast.iter_child_nodes(node):
            add_nodes(child, node)

    add_nodes(tree)
    dot.render(output_file, format="png", cleanup=True)
    print(f"✅ AST 图已生成：{output_file}.png")

# 示例代码
if __name__ == "__main__":
    code = """
class User:
    def __init__(self, name):
        self.name = name

def greet(user):
    print("Hello", user.name)
"""
    visualize_ast_graphviz(code)
```


## demo2. llm 理解

将代码库作为 rag 中的知识库时，大致处理：
- 结合AST和静态分析工具，提取函数/类的调用关系、依赖关系等，构建代码图谱










