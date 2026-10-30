"""Imports stay in module headers, including in the tests themselves."""
import ast
from pathlib import Path


def test_imports_precede_definitions_and_do_not_live_inside_them():
    package = Path(__file__).resolve().parents[3]
    violations = []
    for path in sorted(package.rglob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        first_definition = min(
            (node.lineno for node in tree.body if isinstance(node, definitions)),
            default=float('inf'),
        )
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            ancestor = parents.get(node)
            nested = False
            while ancestor is not None:
                nested |= isinstance(ancestor, definitions)
                ancestor = parents.get(ancestor)
            if nested or node.lineno > first_definition:
                violations.append(f'{path.relative_to(package)}:{node.lineno}')
    assert not violations, 'Imports outside module headers: ' + ', '.join(violations)
