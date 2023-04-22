import enum
import pathlib
from dataclasses import dataclass, replace
from typing import Union, Any, Optional


class UpdateOperator(enum.Enum):
    ADD = enum.auto()


def get_indentation_prefix(indentation):
    indentation_prefix = ""
    if indentation:
        indentation_prefix = " " * indentation
    return indentation_prefix


def concatenate_as_string(sequence, delimiter, indentation=0):
    indentation_prefix = get_indentation_prefix(indentation)
    return delimiter.join([f"{indentation_prefix}{element}" for element in sequence])


@dataclass
class Expression:
    expression: Any

    @property
    def value(self):
        return self.expression.value

    def __repr__(self):
        return f"{self.expression}"

    def __lt__(self, other):
        return Expression(BinaryOperation(self, other, "<"))

    def __mul__(self, other):
        return Expression(BinaryOperation(self, other, "*"))

    def __add__(self, other):
        return Expression(BinaryOperation(self, other, "+"))

    def __sub__(self, other):
        return Expression(BinaryOperation(self, other, "-"))

    def __truediv__(self, other):
        return Expression(BinaryOperation(self, other, "/"))


def literal(value):
    @dataclass
    class Literal:
        value: Any

        def __repr__(self):
            return f"{self.value}"

    return Expression(Literal(value))


@dataclass
class BinaryOperation:
    left_side: Union["Variable", Expression]
    right_side: Union["Variable", Expression]
    operator: str

    def __repr__(self):
        left_side = self.left_side.value
        right_side = self.right_side.value
        return f"({left_side} {self.operator} {right_side})"

    @property
    def value(self):
        return str(self)


@dataclass
class Identifier:
    name: str

    def __repr__(self):
        return f"{self.name}"


@dataclass
class Type:
    name: str
    _pointer: bool = False
    _const: bool = False
    _constexpr: bool = False
    _reference: bool = False
    _restrict: bool = False
    _alignment: Optional[int] = None

    def __repr__(self):
        name = self.name
        if self._pointer:
            name = f"{name}*"
        if self._reference:
            name = f"{name}&"
        if self._restrict:
            name = f"{name} __restrict__"
        if self._alignment is not None:
            name = f"{name} __attribute__((aligned({self._alignment})))"
        if self._const:
            name = f"const {name}"
        if self._constexpr:
            name = f"constexpr {name}"
        return name

    def pointer(self):
        return replace(self, _pointer=True)

    def const(self):
        return replace(self, _const=True)

    def reference(self):
        return replace(self, _reference=True)

    def restrict(self):
        return replace(self, _restrict=True)

    def constexpr(self):
        return replace(self, _constexpr=True)

    def aligned(self, value):
        return replace(self, _alignment=value)


@dataclass
class Variable:
    type: Type
    identifier: Identifier

    @property
    def value(self):
        return self.identifier

    def __repr__(self):
        return f"{self.type} {self.identifier}"

    def __lshift__(self, value):
        return Statement(Declare(self, value))

    def __iadd__(self, value):
        return AssignUpdate(self, value, UpdateOperator.ADD)

    def __lt__(self, other):
        return Expression(BinaryOperation(self, other, "<"))

    def __getitem__(self, item):
        return Subscript(self, item)

    def __mul__(self, other):
        return Expression(BinaryOperation(self, other, "*"))

    def __add__(self, other):
        return Expression(BinaryOperation(self, other, "+"))

    def __sub__(self, other):
        return Expression(BinaryOperation(self, other, "-"))

    def __truediv__(self, other):
        return Expression(BinaryOperation(self, other, "/"))


def variable(type, name) -> Variable:
    return Variable(type, Identifier(name))


AUTO = Type("auto")


@dataclass
class Subscript:
    address: Variable
    offset: Expression

    def __repr__(self):
        address = self.address.value
        offset = self.offset.value
        return f"{address}[{offset}]"

    @property
    def value(self):
        return str(self)

    def __iadd__(self, value):
        return AssignUpdate(self, value, UpdateOperator.ADD)

    def __mul__(self, other):
        return Expression(BinaryOperation(self, other, "*"))

    def __add__(self, other):
        return Expression(BinaryOperation(self, other, "+"))

    def __sub__(self, other):
        return Expression(BinaryOperation(self, other, "-"))

    def __truediv__(self, other):
        return Expression(BinaryOperation(self, other, "/"))


@dataclass
class Statement:
    statement: Any

    def __repr__(self):
        statement = self.statement
        if isinstance(self.statement, Variable):
            statement = str(self.statement.identifier)
        return f"{statement};"


@dataclass
class Return:
    statement: Statement

    def __repr__(self):
        return f"return {self.statement}"


@dataclass
class Assign:
    left_side: Union[Variable, Subscript]
    right_side: Expression
    update_operator: Optional[UpdateOperator] = None

    def __repr__(self):
        return f"{self.left_side.value} = {self.right_side.value}"


def assign(*args):
    return Statement(Assign(*args))


@dataclass
class NotEquals:
    left_side: Union[Variable, Subscript]
    right_side: Expression

    def __repr__(self):
        return f"{self.left_side.value} != {self.right_side.value}"


def not_equals(*args):
    return NotEquals(*args)


@dataclass
class AssignUpdate:
    left_side: Union[Variable, Subscript]
    right_side: Expression
    update_operator: Optional[UpdateOperator] = None

    def __repr__(self):
        if self.update_operator == UpdateOperator.ADD and self.right_side.value == 1:
            return f"{self.left_side.value}++"

        operator_to_string = {
            UpdateOperator.ADD: "+=",
        }
        assign_operator = operator_to_string[self.update_operator]

        return f"{self.left_side.value} {assign_operator} {self.right_side.value}"


def add_in_place(variable, value):
    variable += value
    return variable


@dataclass
class Declare:
    left_side: Variable
    right_side: Expression

    def __repr__(self):
        return f"{self.left_side.type} {Assign(self.left_side, self.right_side)}"


@dataclass
class Block:
    statements: list[Union[Statement, Return, "If", "ForLoop"]]
    indentation_level: Optional[int] = None
    indentation: int = 4

    def set_indentation_level(self, indentation_level=0):
        if self.indentation_level is not None:
            return
        self.indentation_level = indentation_level
        for statement in self.statements:
            if isinstance(statement, (Block, If, ForLoop, Function)):
                statement.set_indentation_level(indentation_level + 1)

    def __add__(self, other: "Block") -> "Block":
        return Block(self.statements + other.statements)

    def __iadd__(self, other: "Block") -> "Block":
        return Block(self.statements + other.statements)

    def __repr__(self):
        indentation = self.indentation_level * self.indentation if self.indentation_level is not None else 0
        indentation_prefix = get_indentation_prefix(indentation)
        delimiter = "\n"
        return f"""\
{indentation_prefix}{{
{concatenate_as_string(self.statements, delimiter, indentation=indentation + self.indentation)}
{indentation_prefix}}}"""


def block(*statements) -> Block:
    return Block(statements)


@dataclass
class If:
    condition: Expression
    body: Block

    def set_indentation_level(self, indentation_level=0):
        self.body.set_indentation_level(indentation_level)

    def __repr__(self):
        self.set_indentation_level()
        return f"""\
if ({self.condition})
{self.body}"""


@dataclass
class ForLoop:
    initialization_statement: Any
    test_expression: Expression
    update_statement: Any
    body: Block

    def set_indentation_level(self, indentation_level=0):
        self.body.set_indentation_level(indentation_level)

    def __repr__(self):
        self.set_indentation_level()
        return f"""\
for ({self.initialization_statement}; {self.test_expression}; {self.update_statement})
{self.body}"""


@dataclass(kw_only=True)
class Function:
    return_type: Type
    name: Identifier
    arguments: list[Variable]
    body: Block

    _inline: bool = False
    _static: bool = False

    def set_indentation_level(self, indentation_level=0):
        self.body.set_indentation_level(indentation_level)

    def __repr__(self):
        self.set_indentation_level()
        delimiter = ", "
        string = f"""\
{self.return_type} {self.name}({concatenate_as_string(self.arguments, delimiter)})
{self.body}"""

        if self._inline:
            string = f"inline {string}"

        if self._static:
            string = f"static {string}"

        return string

    def inline(self):
        return replace(self, _inline=True)

    def static(self):
        return replace(self, _static=True)


@dataclass
class FunctionCall:
    function_name: Identifier
    arguments: list[Union[Identifier, Expression]]

    def __repr__(self):
        delimiter = ", "
        return f"{self.function_name}({concatenate_as_string(self.arguments, delimiter)})"

    @property
    def value(self):
        return str(self)


def invoke(function_name, *arguments):
    arguments = [argument.identifier if isinstance(argument, Variable) else argument for argument in arguments]
    return Expression(FunctionCall(function_name, arguments))


@dataclass
class Include:
    file_name: str

    def __repr__(self):
        return f"#include <{self.file_name}>"


@dataclass
class NewLine:
    amount: int = 1

    def __repr__(self):
        return "\n" * (self.amount - 1)


@dataclass
class Text:
    content: str

    def __repr__(self):
        return self.content


@dataclass
class File:
    name: Union[str, pathlib.Path]
    elements: list[Union[Include, Variable, Function]]

    def __repr__(self):
        delimiter = "\n"
        return f"{concatenate_as_string(self.elements, delimiter)}\n"

    def save(self):
        with open(self.name, "w") as f:
            f.write(str(self))


__all__ = [
    "Expression",
    "literal",
    "BinaryOperation",
    "Identifier",
    "Type",
    "Variable",
    "variable",
    "AUTO",
    "Subscript",
    "Statement",
    "Return",
    "Assign",
    "assign",
    "not_equals",
    "AssignUpdate",
    "add_in_place",
    "Declare",
    "Block",
    "block",
    "If",
    "ForLoop",
    "Function",
    "FunctionCall",
    "invoke",
    "Include",
    "NewLine",
    "File",
    "Text",
]
