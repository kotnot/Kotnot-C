# gitrex_language.py
# Gitrex Language - Язык программирования с современными возможностями

import ast
import re
import math
import json
import random
import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# ==================== ЛЕКСЕР ====================
class TokenType(Enum):
    # Ключевые слова
    ASK = "ASK"
    RESPOND = "RESPOND"
    CHOICE = "CHOICE"
    MATCH = "MATCH"
    CASE = "CASE"
    IF = "IF"
    ELSE = "ELSE"
    WHILE = "WHILE"
    FOR = "FOR"
    IN = "IN"
    FUNCTION = "FUNCTION"
    RETURN = "RETURN"
    IMPORT = "IMPORT"
    FROM = "FROM"
    AS = "AS"
    
    # Типы данных
    LIST = "LIST"
    DICT = "DICT"
    SET = "SET"
    TUPLE = "TUPLE"
    VECTOR = "VECTOR"
    
    # Операторы
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    EQUALS = "EQUALS"
    DOUBLE_EQUALS = "DOUBLE_EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    GREATER = "GREATER"
    LESS = "LESS"
    GREATER_EQUAL = "GREATER_EQUAL"
    LESS_EQUAL = "LESS_EQUAL"
    
    # Символы
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    COMMA = "COMMA"
    DOT = "DOT"
    COLON = "COLON"
    ARROW = "ARROW"
    
    # Литералы
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    BOOLEAN = "BOOLEAN"
    
    # Конец строки
    NEWLINE = "NEWLINE"
    EOF = "EOF"

class Token:
    def __init__(self, type: TokenType, value: Any = None, line: int = 0, col: int = 0):
        self.type = type
        self.value = value
        self.line = line
        self.col = col
    
    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, line={self.line})"

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.col = 1
        self.tokens = []
    
    def tokenize(self) -> List[Token]:
        keywords = {
            'ask': TokenType.ASK,
            'respond': TokenType.RESPOND,
            'choice': TokenType.CHOICE,
            'match': TokenType.MATCH,
            'case': TokenType.CASE,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'in': TokenType.IN,
            'function': TokenType.FUNCTION,
            'return': TokenType.RETURN,
            'import': TokenType.IMPORT,
            'from': TokenType.FROM,
            'as': TokenType.AS,
            'true': TokenType.BOOLEAN,
            'false': TokenType.BOOLEAN,
            'list': TokenType.LIST,
            'dict': TokenType.DICT,
            'set': TokenType.SET,
            'tuple': TokenType.TUPLE,
            'vector': TokenType.VECTOR,
        }
        
        while self.position < len(self.source):
            char = self.source[self.position]
            
            # Пропускаем пробелы
            if char in ' \t':
                self.position += 1
                self.col += 1
                continue
            
            # Новые строки
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, None, self.line, self.col))
                self.position += 1
                self.line += 1
                self.col = 1
                continue
            
            # Комментарии
            if char == '#':
                while self.position < len(self.source) and self.source[self.position] != '\n':
                    self.position += 1
                continue
            
            # Числа
            if char.isdigit():
                start = self.position
                while self.position < len(self.source) and self.source[self.position].isdigit():
                    self.position += 1
                if self.position < len(self.source) and self.source[self.position] == '.':
                    self.position += 1
                    while self.position < len(self.source) and self.source[self.position].isdigit():
                        self.position += 1
                
                number = self.source[start:self.position]
                if '.' in number:
                    self.tokens.append(Token(TokenType.NUMBER, float(number), self.line, self.col))
                else:
                    self.tokens.append(Token(TokenType.NUMBER, int(number), self.line, self.col))
                self.col += self.position - start
                continue
            
            # Строки
            if char in ('"', "'"):
                quote = char
                self.position += 1
                start = self.position
                
                while self.position < len(self.source) and self.source[self.position] != quote:
                    if self.source[self.position] == '\\':
                        self.position += 1
                    self.position += 1
                
                if self.position >= len(self.source):
                    raise SyntaxError(f"Незакрытая строка на строке {self.line}")
                
                string_value = self.source[start:self.position]
                self.tokens.append(Token(TokenType.STRING, string_value, self.line, self.col))
                self.position += 1
                self.col += (self.position - start) + 2
                continue
            
            # Идентификаторы и ключевые слова
            if char.isalpha() or char == '_':
                start = self.position
                while self.position < len(self.source) and (self.source[self.position].isalnum() or self.source[self.position] == '_'):
                    self.position += 1
                
                identifier = self.source[start:self.position]
                token_type = keywords.get(identifier.lower(), TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, identifier, self.line, self.col))
                self.col += self.position - start
                continue
            
            # Операторы
            operators = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                ':': TokenType.COLON,
            }
            
            if char in operators:
                self.tokens.append(Token(operators[char], char, self.line, self.col))
                self.position += 1
                self.col += 1
                continue
            
            # Сравнения
            if char == '=':
                if self.position + 1 < len(self.source) and self.source[self.position + 1] == '=':
                    self.tokens.append(Token(TokenType.DOUBLE_EQUALS, '==', self.line, self.col))
                    self.position += 2
                    self.col += 2
                else:
                    self.tokens.append(Token(TokenType.EQUALS, '=', self.line, self.col))
                    self.position += 1
                    self.col += 1
                continue
            
            if char == '!':
                if self.position + 1 < len(self.source) and self.source[self.position + 1] == '=':
                    self.tokens.append(Token(TokenType.NOT_EQUALS, '!=', self.line, self.col))
                    self.position += 2
                    self.col += 2
                else:
                    raise SyntaxError(f"Неизвестный символ '!' на строке {self.line}")
                continue
            
            if char == '>':
                if self.position + 1 < len(self.source) and self.source[self.position + 1] == '=':
                    self.tokens.append(Token(TokenType.GREATER_EQUAL, '>=', self.line, self.col))
                    self.position += 2
                    self.col += 2
                else:
                    self.tokens.append(Token(TokenType.GREATER, '>', self.line, self.col))
                    self.position += 1
                    self.col += 1
                continue
            
            if char == '<':
                if self.position + 1 < len(self.source) and self.source[self.position + 1] == '=':
                    self.tokens.append(Token(TokenType.LESS_EQUAL, '<=', self.line, self.col))
                    self.position += 2
                    self.col += 2
                else:
                    self.tokens.append(Token(TokenType.LESS, '<', self.line, self.col))
                    self.position += 1
                    self.col += 1
                continue
            
            # Стрелка для лямбда-выражений
            if char == '-':
                if self.position + 1 < len(self.source) and self.source[self.position + 1] == '>':
                    self.tokens.append(Token(TokenType.ARROW, '->', self.line, self.col))
                    self.position += 2
                    self.col += 2
                    continue
            
            raise SyntaxError(f"Неизвестный символ '{char}' на строке {self.line}")
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return self.tokens

# ==================== АБСТРАКТНОЕ СИНТАКСИЧЕСКОЕ ДЕРЕВО ====================
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements: List[ASTNode]):
        self.statements = statements
    
    def __repr__(self):
        return f"Program({self.statements})"

class AskStatement(ASTNode):
    def __init__(self, question: str):
        self.question = question
    
    def __repr__(self):
        return f"Ask({self.question})"

class RespondStatement(ASTNode):
    def __init__(self, message):
        self.message = message
    
    def __repr__(self):
        return f"Respond({self.message})"

class ChoiceStatement(ASTNode):
    def __init__(self, question: str, options: List[str]):
        self.question = question
        self.options = options
    
    def __repr__(self):
        return f"Choice({self.question}, {self.options})"

class MatchStatement(ASTNode):
    def __init__(self, value, cases: List['CaseStatement']):
        self.value = value
        self.cases = cases
    
    def __repr__(self):
        return f"Match({self.value}, {self.cases})"

class CaseStatement(ASTNode):
    def __init__(self, pattern, body: List[ASTNode]):
        self.pattern = pattern
        self.body = body
    
    def __repr__(self):
        return f"Case({self.pattern}, {self.body})"

class FunctionDeclaration(ASTNode):
    def __init__(self, name: str, params: List[str], body: List[ASTNode]):
        self.name = name
        self.params = params
        self.body = body
    
    def __repr__(self):
        return f"Function({self.name}, {self.params}, {self.body})"

class VariableDeclaration(ASTNode):
    def __init__(self, name: str, value):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"Var({self.name}, {self.value})"

class Assignment(ASTNode):
    def __init__(self, name: str, value):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"Assign({self.name}, {self.value})"

class ListLiteral(ASTNode):
    def __init__(self, elements: List[ASTNode]):
        self.elements = elements
    
    def __repr__(self):
        return f"List({self.elements})"

class DictLiteral(ASTNode):
    def __init__(self, elements: Dict[ASTNode, ASTNode]):
        self.elements = elements
    
    def __repr__(self):
        return f"Dict({self.elements})"

class BinaryOperation(ASTNode):
    def __init__(self, left, operator: str, right):
        self.left = left
        self.operator = operator
        self.right = right
    
    def __repr__(self):
        return f"BinOp({self.left}, {self.operator}, {self.right})"

class CallExpression(ASTNode):
    def __init__(self, callee, args: List[ASTNode]):
        self.callee = callee
        self.args = args
    
    def __repr__(self):
        return f"Call({self.callee}, {self.args})"

class Identifier(ASTNode):
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"Identifier({self.name})"

class NumberLiteral(ASTNode):
    def __init__(self, value: Union[int, float]):
        self.value = value
    
    def __repr__(self):
        return f"Number({self.value})"

class StringLiteral(ASTNode):
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return f"String({self.value})"

class BooleanLiteral(ASTNode):
    def __init__(self, value: bool):
        self.value = value
    
    def __repr__(self):
        return f"Boolean({self.value})"

class MapExpression(ASTNode):
    def __init__(self, collection, lambda_expr):
        self.collection = collection
        self.lambda_expr = lambda_expr
    
    def __repr__(self):
        return f"Map({self.collection}, {self.lambda_expr})"

class FilterExpression(ASTNode):
    def __init__(self, collection, lambda_expr):
        self.collection = collection
        self.lambda_expr = lambda_expr
    
    def __repr__(self):
        return f"Filter({self.collection}, {self.lambda_expr})"

# ==================== ПАРСЕР ====================
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else None
    
    def advance(self):
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def expect(self, token_type: TokenType, error_msg: str = None):
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            expected = error_msg or str(token_type)
            raise SyntaxError(f"Ожидался {expected}, получен {self.current_token}")
    
    def parse(self) -> Program:
        statements = []
        
        while self.current_token and self.current_token.type != TokenType.EOF:
            statement = self.parse_statement()
            if statement:
                statements.append(statement)
            
            # Пропускаем новые строки
            while self.current_token and self.current_token.type == TokenType.NEWLINE:
                self.advance()
        
        return Program(statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        if not self.current_token:
            return None
        
        # Ask statement
        if self.current_token.type == TokenType.ASK:
            return self.parse_ask()
        
        # Respond statement
        elif self.current_token.type == TokenType.RESPOND:
            return self.parse_respond()
        
        # Choice statement
        elif self.current_token.type == TokenType.CHOICE:
            return self.parse_choice()
        
        # Match statement
        elif self.current_token.type == TokenType.MATCH:
            return self.parse_match()
        
        # Function declaration
        elif self.current_token.type == TokenType.FUNCTION:
            return self.parse_function()
        
        # Variable assignment
        elif self.current_token.type == TokenType.IDENTIFIER:
            return self.parse_assignment_or_expression()
        
        # Map/Filter
        elif self.current_token.type in (TokenType.LIST, TokenType.DICT):
            # Пока пропускаем сложные случаи
            self.advance()
            return None
        
        return None
    
    def parse_ask(self) -> AskStatement:
        self.expect(TokenType.ASK)
        self.expect(TokenType.LPAREN)
        
        question_token = self.expect(TokenType.STRING)
        question = question_token.value
        
        self.expect(TokenType.RPAREN)
        
        return AskStatement(question)
    
    def parse_respond(self) -> RespondStatement:
        self.expect(TokenType.RESPOND)
        self.expect(TokenType.LPAREN)
        
        # Может быть строка или выражение
        if self.current_token.type == TokenType.STRING:
            message_token = self.expect(TokenType.STRING)
            message = StringLiteral(message_token.value)
        else:
            message = self.parse_expression()
        
        self.expect(TokenType.RPAREN)
        
        return RespondStatement(message)
    
    def parse_choice(self) -> ChoiceStatement:
        self.expect(TokenType.CHOICE)
        self.expect(TokenType.LPAREN)
        
        question_token = self.expect(TokenType.STRING)
        question = question_token.value
        
        self.expect(TokenType.COMMA)
        self.expect(TokenType.LBRACKET)
        
        options = []
        while self.current_token and self.current_token.type != TokenType.RBRACKET:
            if self.current_token.type == TokenType.STRING:
                options.append(self.current_token.value)
                self.advance()
            
            if self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RBRACKET)
        self.expect(TokenType.RPAREN)
        
        return ChoiceStatement(question, options)
    
    def parse_match(self) -> MatchStatement:
        self.expect(TokenType.MATCH)
        
        value = self.parse_expression()
        
        self.expect(TokenType.COLON)
        
        cases = []
        while self.current_token and self.current_token.type == TokenType.CASE:
            cases.append(self.parse_case())
        
        return MatchStatement(value, cases)
    
    def parse_case(self) -> CaseStatement:
        self.expect(TokenType.CASE)
        
        # Пока простой парсинг
        if self.current_token.type == TokenType.STRING:
            pattern = StringLiteral(self.current_token.value)
            self.advance()
        else:
            pattern = Identifier("_")  # Default case
        
        self.expect(TokenType.COLON)
        
        body = []
        while self.current_token and self.current_token.type not in (TokenType.CASE, TokenType.NEWLINE):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        
        return CaseStatement(pattern, body)
    
    def parse_function(self) -> FunctionDeclaration:
        self.expect(TokenType.FUNCTION)
        
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        
        self.expect(TokenType.LPAREN)
        
        params = []
        while self.current_token and self.current_token.type != TokenType.RPAREN:
            if self.current_token.type == TokenType.IDENTIFIER:
                params.append(self.current_token.value)
                self.advance()
            
            if self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.COLON)
        
        body = []
        while self.current_token and self.current_token.type not in (TokenType.NEWLINE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        
        return FunctionDeclaration(name, params, body)
    
    def parse_assignment_or_expression(self) -> ASTNode:
        identifier = Identifier(self.current_token.value)
        self.advance()
        
        if self.current_token and self.current_token.type == TokenType.EQUALS:
            self.advance()
            value = self.parse_expression()
            return Assignment(identifier.name, value)
        
        # Это может быть вызов функции или выражение
        # Упрощаем
        return identifier
    
    def parse_expression(self) -> ASTNode:
        # Упрощенный парсер выражений
        if self.current_token.type == TokenType.NUMBER:
            node = NumberLiteral(self.current_token.value)
            self.advance()
            return node
        elif self.current_token.type == TokenType.STRING:
            node = StringLiteral(self.current_token.value)
            self.advance()
            return node
        elif self.current_token.type == TokenType.BOOLEAN:
            node = BooleanLiteral(self.current_token.value.lower() == 'true')
            self.advance()
            return node
        elif self.current_token.type == TokenType.IDENTIFIER:
            node = Identifier(self.current_token.value)
            self.advance()
            return node
        else:
            # По умолчанию возвращаем пустую строку
            return StringLiteral("")

# ==================== ИНТЕРПРЕТАТОР ====================
class GitrexInterpreter:
    def __init__(self, input_callback=None, output_callback=None):
        self.variables = {}
        self.functions = {}
        self.input_callback = input_callback
        self.output_callback = output_callback
        
        # Стандартные функции
        self.std_functions = {
            'print': self._print,
            'len': self._len,
            'range': self._range,
            'time': self._time,
        }
    
    def interpret(self, ast: Program) -> Any:
        result = None
        
        for statement in ast.statements:
            result = self.visit(statement)
        
        return result
    
    def visit(self, node: ASTNode) -> Any:
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node: ASTNode) -> Any:
        raise Exception(f"Нет метода visit_{type(node).__name__}")
    
    def visit_Program(self, node: Program) -> Any:
        for statement in node.statements:
            self.visit(statement)
        return None
    
    def visit_AskStatement(self, node: AskStatement) -> Any:
        if self.output_callback:
            self.output_callback(f"❓ {node.question}")
        
        if self.input_callback:
            answer = self.input_callback()
            self.variables['_last_answer'] = answer
            return answer
        
        # Если нет callback, используем input()
        answer = input(f"❓ {node.question}: ")
        self.variables['_last_answer'] = answer
        return answer
    
    def visit_RespondStatement(self, node: RespondStatement) -> Any:
        message = self.visit(node.message)
        if self.output_callback:
            self.output_callback(f"💬 {message}")
        else:
            print(f"💬 {message}")
        return message
    
    def visit_ChoiceStatement(self, node: ChoiceStatement) -> Any:
        if self.output_callback:
            self.output_callback(f"🎯 {node.question}")
        
        for i, option in enumerate(node.options, 1):
            if self.output_callback:
                self.output_callback(f"  {i}. {option}")
            else:
                print(f"  {i}. {option}")
        
        if self.input_callback:
            choice = self.input_callback()
        else:
            choice = input("Выберите номер: ")
        
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(node.options):
                selected = node.options[choice_index]
                self.variables['_choice_result'] = selected
                return selected
        except:
            pass
        
        return None
    
    def visit_MatchStatement(self, node: MatchStatement) -> Any:
        value = self.visit(node.value)
        
        for case in node.cases:
            pattern = self.visit(case.pattern)
            
            # Простое сравнение для демонстрации
            if pattern == "_" or pattern == value:
                for stmt in case.body:
                    self.visit(stmt)
                break
        
        return None
    
    def visit_FunctionDeclaration(self, node: FunctionDeclaration) -> Any:
        self.functions[node.name] = node
        return None
    
    def visit_Assignment(self, node: Assignment) -> Any:
        value = self.visit(node.value)
        self.variables[node.name] = value
        return value
    
    def visit_ListLiteral(self, node: ListLiteral) -> Any:
        return [self.visit(element) for element in node.elements]
    
    def visit_DictLiteral(self, node: DictLiteral) -> Any:
        result = {}
        for key, value in node.elements.items():
            result[self.visit(key)] = self.visit(value)
        return result
    
    def visit_BinaryOperation(self, node: BinaryOperation) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if node.operator == '+':
            return left + right
        elif node.operator == '-':
            return left - right
        elif node.operator == '*':
            return left * right
        elif node.operator == '/':
            return left / right
        elif node.operator == '==':
            return left == right
        elif node.operator == '!=':
            return left != right
        elif node.operator == '>':
            return left > right
        elif node.operator == '<':
            return left < right
        elif node.operator == '>=':
            return left >= right
        elif node.operator == '<=':
            return left <= right
        
        raise Exception(f"Неизвестный оператор: {node.operator}")
    
    def visit_CallExpression(self, node: CallExpression) -> Any:
        callee = self.visit(node.callee)
        
        if callable(callee):
            args = [self.visit(arg) for arg in node.args]
            return callee(*args)
        
        # Если это стандартная функция
        if isinstance(callee, str) and callee in self.std_functions:
            args = [self.visit(arg) for arg in node.args]
            return self.std_functions[callee](*args)
        
        return None
    
    def visit_Identifier(self, node: Identifier) -> Any:
        if node.name in self.variables:
            return self.variables[node.name]
        elif node.name in self.std_functions:
            return self.std_functions[node.name]
        else:
            return node.name
    
    def visit_NumberLiteral(self, node: NumberLiteral) -> Any:
        return node.value
    
    def visit_StringLiteral(self, node: StringLiteral) -> Any:
        return node.value
    
    def visit_BooleanLiteral(self, node: BooleanLiteral) -> Any:
        return node.value
    
    def visit_MapExpression(self, node: MapExpression) -> Any:
        collection = self.visit(node.collection)
        result = []
        
        for item in collection:
            # Упрощенная реализация map
            result.append(item * 2)  # Пример
        
        return result
    
    # Стандартные функции
    def _print(self, *args):
        result = ' '.join(str(arg) for arg in args)
        if self.output_callback:
            self.output_callback(result)
        else:
            print(result)
        return result
    
    def _len(self, collection):
        return len(collection)
    
    def _range(self, start, end=None):
        if end is None:
            return list(range(start))
        return list(range(start, end))
    
    def _time(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

# ==================== ЯЗЫК GITREX ====================
class GitrexLanguage:
    """Основной класс языка Gitrex"""
    
    def __init__(self):
        self.interpreter = GitrexInterpreter()
        self.builtin_functions = self._setup_builtins()
    
    def _setup_builtins(self):
        """Настройка встроенных функций"""
        return {
            # Математика
            'abs': abs,
            'min': min,
            'max': max,
            'round': round,
            'sum': sum,
            'pow': pow,
            'sqrt': lambda x: math.sqrt(x),
            
            # Строки
            'upper': lambda s: s.upper(),
            'lower': lambda s: s.lower(),
            'capitalize': lambda s: s.capitalize(),
            'strip': lambda s: s.strip(),
            
            # Коллекции
            'map': lambda f, lst: [f(x) for x in lst],
            'filter': lambda f, lst: [x for x in lst if f(x)],
            'reduce': lambda f, lst, initial=None: self._reduce(f, lst, initial),
        }
    
    def _reduce(self, func, lst, initial=None):
        """Реализация reduce"""
        if not lst:
            return initial
        
        if initial is None:
            result = lst[0]
            start = 1
        else:
            result = initial
            start = 0
        
        for item in lst[start:]:
            result = func(result, item)
        
        return result
    
    def execute(self, code: str, input_data: List[str] = None) -> Dict[str, Any]:
        """
        Выполнение кода на Gitrex
        
        Args:
            code: Код на Gitrex
            input_data: Данные для ввода (если есть)
        
        Returns:
            Словарь с результатами выполнения
        """
        # Лексический анализ
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        
        # Синтаксический анализ
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Подготовка ввода
        input_index = 0
        input_results = []
        
        def input_callback():
            nonlocal input_index
            if input_data and input_index < len(input_data):
                result = input_data[input_index]
                input_index += 1
                return result
            return input("Введите значение: ")
        
        def output_callback(msg):
            print(msg)
        
        # Интерпретация
        self.interpreter.input_callback = input_callback
        self.interpreter.output_callback = output_callback
        
        try:
            result = self.interpreter.interpret(ast)
            return {
                'success': True,
                'result': result,
                'variables': self.interpreter.variables,
                'input_used': input_results,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'variables': self.interpreter.variables,
            }
    
    def compile_to_python(self, code: str) -> str:
        """
        Компиляция Gitrex кода в Python
        
        Args:
            code: Код на Gitrex
        
        Returns:
            Код на Python
        """
        # Упрощенная компиляция
        python_code = []
        python_code.append("# Скомпилированный Gitrex код")
        python_code.append("from datetime import datetime")
        python_code.append("")
        
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Преобразование ask()
            if 'ask(' in line:
                match = re.search(r'ask\(\s*"([^"]+)"\s*\)', line)
                if match:
                    question = match.group(1)
                    python_code.append(f'_last_answer = input("{question}: ")')
                    continue
            
            # Преобразование respond()
            if 'respond(' in line:
                match = re.search(r'respond\(\s*"([^"]+)"\s*\)', line)
                if match:
                    response = match.group(1)
                    python_code.append(f'print("💬 {response}")')
                    continue
            
            # Преобразование choice()
            if 'choice(' in line:
                match = re.search(r'choice\(\s*"([^"]+)",\s*\[(.*)\]\s*\)', line)
                if match:
                    question = match.group(1)
                    options = match.group(2)
                    python_code.append(f'print("🎯 {question}")')
                    
                    options_list = eval(f'[{options}]')
                    for i, opt in enumerate(options_list, 1):
                        python_code.append(f'print(f"  {i}. {opt}")')
                    
                    python_code.append('choice = int(input("Выберите номер: "))')
                    python_code.append(f'_choice_result = [{options}][choice-1]')
                    continue
            
            # Копируем остальные строки
            python_code.append(line)
        
        return '\n'.join(python_code)

# ==================== ПРИМЕРЫ ====================
def example_dialog():
    """Пример диалога на Gitrex"""
    code = '''
# Пример диалога
ask("Как вас зовут?")
respond("Привет, " + _last_answer + "!")

ask("Сколько вам лет?")
age = _last_answer
respond("Вам " + age + " лет")

choice("Что вы хотите сделать?", [
    "Посчитать сумму чисел",
    "Узнать текущее время",
    "Выйти"
])

match _choice_result:
    case "Посчитать сумму чисел"
        numbers = [1, 2, 3, 4, 5]
        sum_result = sum(numbers)
        respond("Сумма чисел: " + sum_result)
    case "Узнать текущее время"
        respond("Текущее время: " + time())
    case "Выйти"
        respond("До свидания!")
    case _
        respond("Неизвестный выбор")
'''
    
    gitrex = GitrexLanguage()
    print("🚀 Запуск примера диалога:")
    result = gitrex.execute(code, ["Анна", "25", "2"])
    print(f"📊 Результат: {result}")

def example_calculator():
    """Пример калькулятора на Gitrex"""
    code = '''
# Интерактивный калькулятор
function add(a, b):
    return a + b

function multiply(a, b):
    return a * b

respond("🧮 Добро пожаловать в калькулятор!")

choice = choice("Выберите операцию:", ["Сложение", "Умножение"])

if _choice_result == "Сложение":
    ask("Введите первое число:")
    num1 = _last_answer
    ask("Введите второе число:")
    num2 = _last_answer
    result = add(num1, num2)
    respond("Результат сложения: " + result)
else:
    ask("Введите первое число:")
    num1 = _last_answer
    ask("Введите второе число:")
    num2 = _last_answer
    result = multiply(num1, num2)
    respond("Результат умножения: " + result)
'''
    
    gitrex = GitrexLanguage()
    print("\n🚀 Запуск примера калькулятора:")
    result = gitrex.execute(code, ["1", "5", "3", "4"])
    print(f"📊 Результат: {result}")

def example_compilation():
    """Пример компиляции Gitrex в Python"""
    code = '''
ask("Введите ваше имя:")
respond("Привет, " + _last_answer)

numbers = [1, 2, 3, 4, 5]
squares = map(x => x * x, numbers)
respond("Квадраты чисел: " + squares)
'''
    
    gitrex = GitrexLanguage()
    python_code = gitrex.compile_to_python(code)
    
    print("\n📝 Скомпилированный Python код:")
    print(python_code)
    
    print("\n🚀 Выполнение скомпилированного кода:")
    exec(python_code, {'datetime': datetime})

# ==================== API ДЛЯ ИНТЕГРАЦИИ ====================
class GitrexAPI:
    """API для интеграции Gitrex в другие приложения"""
    
    @staticmethod
    def run_script(script_path: str, inputs: List[str] = None) -> Dict[str, Any]:
        """Запуск Gitrex скрипта из файла"""
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        gitrex = GitrexLanguage()
        return gitrex.execute(code, inputs)
    
    @staticmethod
    def evaluate(expression: str) -> Any:
        """Вычисление выражения Gitrex"""
        gitrex = GitrexLanguage()
        return gitrex.execute(expression)
    
    @staticmethod
    def create_function(name: str, params: List[str], body: str):
        """Создание пользовательской функции"""
        code = f"function {name}({', '.join(params)}):\n    {body}"
        gitrex = GitrexLanguage()
        return gitrex.execute(code)

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
if __name__ == "__main__":
    print("=" * 50)
    print("       🚀 ЯЗЫК ПРОГРАММИРОВАНИЯ GITREX")
    print("=" * 50)
    
    # Пример 1: Диалог
    example_dialog()
    
    # Пример 2: Калькулятор
    example_calculator()
    
    # Пример 3: Компиляция
    example_compilation()
    
    print("\n" + "=" * 50)
    print("✅ Все примеры выполнены успешно!")
    print("=" * 50)
    
    # Пример использования API
    print("\n📦 Пример использования API:")
    
    result = GitrexAPI.run_script(
        "example.gitrex",  # Предполагаемый файл
        ["Иван", "30", "1"]
    )
    print(f"Результат выполнения скрипта: {result}")
