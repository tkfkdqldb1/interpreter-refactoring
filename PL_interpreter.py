# -*- coding: utf-8 -*-
from string import letters, digits, whitespace

mydict = dict()

class CuteType:
    INT = 1
    ID = 4

    MINUS = 2
    PLUS = 3

    L_PAREN = 5
    R_PAREN = 6

    TRUE = 8
    FALSE = 9

    TIMES = 10
    DIV = 11

    LT = 12
    GT = 13
    EQ = 14
    APOSTROPHE = 15

    DEFINE = 20
    LAMBDA = 21
    COND = 22
    QUOTE = 23
    NOT = 24
    CAR = 25
    CDR = 26
    CONS = 27
    ATOM_Q = 28
    NULL_Q = 29
    EQ_Q = 30

    KEYWORD_LIST = ('define', 'lambda', 'cond', 'quote', 'not', 'car', 'cdr', 'cons',
                    'atom?', 'null?', 'eq?')

    BINARYOP_LIST = (DIV, TIMES, MINUS, PLUS, LT, GT, EQ)
    BOOLEAN_LIST = (TRUE, FALSE)


def check_keyword(token):
    """
    :type token:str
    :param token:
    :return:
    """
    if token.lower() in CuteType.KEYWORD_LIST:
        return True
    return False


def _get_keyword_type(token):
    return {
        'define': CuteType.DEFINE,
        'lambda': CuteType.LAMBDA,
        'cond': CuteType.COND,
        'quote': CuteType.QUOTE,
        'not': CuteType.NOT,
        'car': CuteType.CAR,
        'cdr': CuteType.CDR,
        'cons': CuteType.CONS,
        'atom?': CuteType.ATOM_Q,
        'null?': CuteType.NULL_Q,
        'eq?': CuteType.EQ_Q
    }[token]


CUTETYPE_NAMES = dict((eval(attr, globals(), CuteType.__dict__), attr) for attr in dir(
    CuteType()) if not callable(attr) and not attr.startswith('__'))


class Token(object):
    def __init__(self, type, lexeme):
        """
        :type type:CuteType
        :type lexeme: str
        :param type:
        :param lexeme:
        :return:
        """
        if check_keyword(lexeme):
            self.type = _get_keyword_type(lexeme)
            self.lexeme = lexeme
        else:
            self.type = type
            self.lexeme = lexeme
        # print type

    def __str__(self):
        # return self.lexeme
        return '[' + CUTETYPE_NAMES[self.type] + ': ' + self.lexeme + ']'

    def __repr__(self):
        return str(self)


class Scanner:

    def __init__(self, source_string=None):
        """
        :type self.__source_string: str
        :param source_string:
        """
        self.__source_string = source_string
        self.__pos = 0
        self.__length = len(source_string)
        self.__token_list = []

    def __make_token(self, transition_matrix, build_token_func=None):
        old_state = 0
        self.__skip_whitespace()
        temp_char = ''
        return_token = ''
        while not self.eos():
            temp_char = self.get()
            if old_state == 0 and temp_char in (')', '('):
                return_token = temp_char
                old_state = transition_matrix[(old_state, temp_char)]
                break

            return_token += temp_char
            old_state = transition_matrix[(old_state, temp_char)]
            next_char = self.peek()
            if next_char in whitespace or next_char in ('(', ')'):
                break

        return build_token_func(old_state, return_token)

    def scan(self, transition_matrix, build_token_func):
        while not self.eos():
            self.__token_list.append(self.__make_token(
                transition_matrix, build_token_func))
        return self.__token_list

    def pos(self):
        return self.__pos

    def eos(self):
        return self.__pos >= self.__length

    def skip(self, pattern):
        while not self.eos():
            temp_char = self.peek()
            if temp_char in pattern:
                temp_char = self.get()
            else:
                break

    def __skip_whitespace(self):
        self.skip(whitespace)

    def peek(self, length=1):
        return self.__source_string[self.__pos: self.__pos + length]

    def get(self, length=1):
        return_get_string = self.peek(length)
        self.__pos += len(return_get_string)
        return return_get_string


class CuteScanner(object):

    transM = {}

    def __init__(self, source):
        """
        :type source:str
        :param source:
        :return:
        """
        self.source = source
        self._init_TM()

    def _init_TM(self):
        for alpha in letters:
            self.transM[(0, alpha)] = 4
            self.transM[(4, alpha)] = 4

        for digit in digits:
            self.transM[(0, digit)] = 1
            self.transM[(1, digit)] = 1
            self.transM[(2, digit)] = 1
            self.transM[(4, digit)] = 4

        self.transM[(4, '?')] = 16
        self.transM[(0, '-')] = 2
        self.transM[(0, '+')] = 3
        self.transM[(0, '(')] = 5
        self.transM[(0, ')')] = 6

        self.transM[(0, '#')] = 7
        self.transM[(7, 'T')] = 8
        self.transM[(7, 'F')] = 9

        self.transM[(0, '/')] = 11
        self.transM[(0, '*')] = 10

        self.transM[(0, '<')] = 12
        self.transM[(0, '>')] = 13
        self.transM[(0, '=')] = 14
        self.transM[(0, "'")] = 15

    def tokenize(self):

        def build_token(type, lexeme): return Token(type, lexeme)
        cute_scanner = Scanner(self.source)
        return cute_scanner.scan(self.transM, build_token)


class TokenType():
    INT = 1
    ID = 4
    MINUS = 2
    PLUS = 3
    LIST = 5
    TRUE = 8
    FALSE = 9
    TIMES = 10
    DIV = 11
    LT = 12
    GT = 13
    EQ = 14
    APOSTROPHE = 15
    DEFINE = 20
    LAMBDA = 21
    COND = 22
    QUOTE = 23
    NOT = 24
    CAR = 25
    CDR = 26
    CONS = 27
    ATOM_Q = 28
    NULL_Q = 29
    EQ_Q = 30

NODETYPE_NAMES = dict((eval(attr, globals(), TokenType.__dict__), attr) for attr in dir(
    TokenType()) if not callable(attr) and not attr.startswith('__'))

class Node (object):

    def __init__(self, type, value=None):
        self.next = None
        self.value = value
        self.type = type

    def set_last_next(self, next_node):
        if self.next is not None:
            self.next.set_last_next(next_node)

        else:
            self.next = next_node

    def __str__(self):
        result = ''

        if self.type is TokenType.ID:
            result = '[' + NODETYPE_NAMES[self.type] + ':' + self.value + ']'
        elif self.type is TokenType.INT:
            result = '['+NODETYPE_NAMES[self.type]+':' + self.value + ']'
        elif self.type is TokenType.LIST:
            if self.value is not None:
                if self.value.type is TokenType.QUOTE:
                    result = str(self.value)
                else:
                    result = '(' + str(self.value) + ')'
            else:
                result = '(' + str(self.value) + ')'
        elif self.type is TokenType.QUOTE:
            result = "\'"
        else:
            result = '['+NODETYPE_NAMES[self.type]+']'

        # fill out
        if self.next is not None:
            return result + ' ' + str(self.next)
        else:
            return result


class BasicPaser(object):

    def __init__(self, token_list):
        """
        :type token_list:list
        :param token_list:
        :return:
        """
        self.token_iter = iter(token_list)

    def _get_next_token(self):
        """
        :rtype: Token
        :return:
        """
        next_token = next(self.token_iter, None)
        if next_token is None:
            return None
        return next_token

    def parse_expr(self):
        """
        :rtype : Node
        :return:
        """
        token = self._get_next_token()

        '"":type :Token""'
        if token is None:
            return None
        result = self._create_node(token)
        return result

    def _create_node(self, token):
        if token is None:
            return None
        elif token.type is CuteType.INT:
            return Node(TokenType.INT,  token.lexeme)
        elif token.type is CuteType.ID:
            return Node(TokenType.ID,   token.lexeme)
        elif token.type is CuteType.L_PAREN:
            return Node(TokenType.LIST, self._parse_expr_list())
        elif token.type is CuteType.R_PAREN:
            return None
        elif token.type in CuteType.BOOLEAN_LIST:
            return Node(token.type)
        elif token.type in CuteType.BINARYOP_LIST:
            return Node(token.type, token.lexeme)
        elif token.type is CuteType.QUOTE:
            return Node(TokenType.QUOTE, token.lexeme)
        elif token.type is CuteType.APOSTROPHE:
            node = Node(TokenType.LIST, Node(TokenType.QUOTE, token.lexeme))
            node.value.next = self.parse_expr()
            return node
        elif check_keyword(token.lexeme):
            return Node(token.type, token.lexeme)

    def _parse_expr_list(self):
        head = self.parse_expr()
        '"":type :Node""'
        if head is not None:
            head.next = self._parse_expr_list()
        return head


def run_list(root_node):
    """
    :type root_node: Node
    """
    op_code_node = root_node.value

    return run_func(op_code_node)(root_node)


def run_func(op_code_node):
    """
    :type op_code_node:Node/
    """
    def quote(node):
        return node

    def strip_quote(node):
        """
        :type node: Node
        """
        if node.type is TokenType.LIST:
            if node.value is TokenType.QUOTE or TokenType.APOSTROPHE:
                return node.value.next
        if node.type is TokenType.QUOTE:
            return node.next
        return node

    def cons(node):
        """
        :type node: Node
        """
        l_node = node.value.next
        r_node = l_node.next
        r_node = run_expr(r_node)
        l_node = run_expr(l_node)
        new_r_node = r_node
        new_l_node = l_node
        new_r_node = strip_quote(new_r_node)
        new_l_node = strip_quote(new_l_node)
        new_l_node.next = new_r_node.value

        return create_new_quote_list(new_l_node, True)

    def car(node):
        l_node = run_expr(node.value.next)
        result = strip_quote(l_node).value
        if result.type is not TokenType.LIST:
            return result
        return create_new_quote_list(result)

    def cdr(node):
        """
        :type node: Node
        """
        l_node = node.value.next
        l_node = run_expr(l_node)
        new_r_node = strip_quote(l_node)
        return create_new_quote_list(new_r_node.value.next, True)

    def null_q(node):
        l_node = run_expr(node.value.next)
        new_l_node = strip_quote(l_node).value
        if new_l_node is None:
            return Node(TokenType.TRUE)
        else:
            return Node(TokenType.FALSE)

    def atom_q(node):
        l_node = run_expr(node.value.next)
        new_l_node = strip_quote(l_node)

        if new_l_node.type is TokenType.LIST:
            if new_l_node.value is None:
                return Node(TokenType.TRUE)
            return Node(TokenType.FALSE)
        else:
            return Node(TokenType.TRUE)

    def eq_q(node):
        l_node = node.value.next
        r_node = l_node.next
        new_l_node = strip_quote(run_expr(l_node))
        new_r_node = strip_quote(run_expr(r_node))

        if (new_l_node.type or new_r_node.type) is not TokenType.INT:
            return Node(TokenType.FALSE)
        if new_l_node.value == new_r_node.value:
            return Node(TokenType.TRUE)
        return Node(TokenType.FALSE)

    def not_op(node):
        result = node.value.next

        if result.type is TokenType.TRUE:
            return Node(TokenType.FALSE)
        elif result.type is TokenType.FALSE:
            return Node(TokenType.TRUE)

        return 0

    # defin 연산 계산
    def define_result(one_node, two_node, opNum):
        # 만약 첫 번째 노드가 숫자이고 두 번째 노드에서 define str이 나올 경우
        if one_node.type is TokenType.INT:
            if two_node.type is TokenType.ID:
                if mydict.__contains__(two_node.value):
                    two_result = mydict.get(two_node.value)
                    two_result = two_result.value
                else:
                    return define_result(two_node, one_node, opNum)
            elif two_node.type is TokenType.LIST:
                two_result = two_node.value.value.value
            else:
                two_result = int(two_node.value)

            one_result = int(one_node.value)

            if opNum is TokenType.PLUS:
                result = one_result + two_result
            elif opNum is TokenType.MINUS:
                result = one_result - two_result
            elif opNum is TokenType.TIMES:
                result = one_result * two_result
            elif opNum is TokenType.DIV:
                result = one_result / two_result

            return Node(TokenType.INT, result)

        elif one_node.type is TokenType.ID:
            if two_node.type is TokenType.ID:
                two_result = int(print_node(two_node))
            else:
                two_result = int(two_node.value)

#            if mydict.__contains__(one_node.value):
#                one_result = int(mydict.get(one_node.value))
#            else:

            one_result = int(print_node(one_node))

            if opNum is TokenType.PLUS:
                result = one_result + two_result
            elif opNum is TokenType.MINUS:
                result = one_result - two_result
            elif opNum is TokenType.TIMES:
                result = one_result * two_result
            elif opNum is TokenType.DIV:
                result = one_result / two_result

            return Node(TokenType.INT, result)


        elif one_node.type is TokenType.LIST:
            if two_node.type is TokenType.ID:
                two_result = int(print_node(two_node))
            else:
                two_result = int(two_node.value)

            one_result = one_node.value.value.value

            if opNum is TokenType.PLUS:
                result = int(one_result) + two_result
            elif opNum is TokenType.MINUS:
                result = int(one_result) - two_result
            elif opNum is TokenType.TIMES:
                result = int(one_result) * two_result
            elif opNum is TokenType.DIV:
                result = int(one_result) / two_result

            return Node(TokenType.INT, result)

        return 0

    def plus(node):
        one_node = node.value.next
        two_node = one_node.next

        one_node = node_check(one_node)
        two_node = node_check(two_node)

        return define_result(one_node, two_node, TokenType.PLUS)

    def minus(node):
        one_node = node.value.next
        two_node = one_node.next

        one_node = node_check(one_node)
        two_node = node_check(two_node)

        return define_result(one_node, two_node, TokenType.MINUS)

    def multiple(node):
        one_node = node.value.next
        two_node = one_node.next

        one_node = node_check(one_node)
        two_node = node_check(two_node)

        return define_result(one_node, two_node, TokenType.TIMES)

    def divide(node):
        one_node = node.value.next
        two_node = one_node.next

        if one_node.type is TokenType.LIST:
            if one_node.value.type is TokenType.PLUS:
                one_node = plus(one_node)
            elif one_node.value.type is TokenType.MINUS:
                one_node = minus(one_node)
            elif one_node.value.type is TokenType.TIMES:
                one_node = multiple(one_node)
            elif one_node.value.type is TokenType.DIV:
                one_node = divide(one_node)

        return define_result(one_node, two_node, TokenType.DIV)

    def lt(node):
        one_node = node.value.next
        two_node = one_node.next

        one_node = node_check(one_node)
        two_node = node_check(two_node)

        one_result = int(one_node.value)
        two_result = int(two_node.value)

        if (one_result - two_result) < 0:
            return Node(TokenType.TRUE)
        else:
            return Node(TokenType.FALSE)

    def gt(node):
        one_node = node.value.next
        two_node = one_node.next

        one_node = node_check(one_node)
        two_node = node_check(two_node)

        one_result = int(one_node.value)
        two_result = int(two_node.value)

        if (one_result - two_result) > 0:
            return Node(TokenType.TRUE)
        else:
            return Node(TokenType.FALSE)

    def eq(node):
        one_node = node.value.next
        two_node = one_node.next

        one_node = node_check(one_node)
        two_node = node_check(two_node)

        one_result = int(one_node.value)
        two_result = int(two_node.value)

        if one_result == two_result:
            return Node(TokenType.TRUE)
        else :
            return Node(TokenType.FALSE)

    def define(node):
        l_node = node.value.next
        r_node = l_node.next

        if type(r_node.value) != str:#두번째가 str이 아니고
            if (r_node.value.next.next is not None) and (type(r_node.value.next.next.value)  is not str):
                if r_node.value.next.next.value.value == 'define':
                    return mydict.__setitem__(l_node.value, r_node.value.next.next )

            if r_node.type is TokenType.LIST and r_node.value.value == 'lambda':#그 문자열 뒤에 오는 것이 리스트이면 함수 선언이므로
                 return mydict.__setitem__(l_node.value,r_node.value.next.next.value)

        l_node = run_expr(l_node)
        r_node = run_expr(r_node)

        l_node = l_node.value
        r_result = r_node.value

        mydict.__setitem__(l_node, r_result)

    def redefine(node):
        l_node = node.value.next
        r_node = node.value.value.next.value

        if l_node.next != None:
            l2_node = l_node.next
            r2_node = r_node.next
            l2_node = l2_node.value
            r2_node = r2_node.value
            mydict.__setitem__(r2_node,l2_node)

        l_node = run_expr(l_node)
        r_node = run_expr(r_node)

        l_node = l_node.value
        r_result = r_node.value

        mydict.__setitem__(r_result,l_node)


    def LAMBDA(node):
        redefine(node)
        operater = node.value.value.next.next
        return run_expr(operater)

    def node_check(node):
        if (node.type is TokenType.LIST) or (node.type is TokenType.QUOTE) or (node.type is TokenType.APOSTROPHE):
            if node.value.type is TokenType.PLUS:
                node = plus(node)
            elif node.value.type is TokenType.MINUS:
                node = minus(node)
            elif node.value.type is TokenType.TIMES:
                node = multiple(node)
            elif node.value.type is TokenType.DIV:
                node = divide(node)
            elif node.value.type is TokenType.EQ:
                node = eq(node)
            elif node.value.type is TokenType.GT:
                node = gt(node)
            elif node.value.type is TokenType.LT:
                node = lt(node)
            elif node.value.type is TokenType.TRUE:
                node = node.value
            elif node.value.type is TokenType.FALSE:
                node = node.value
        return node

    def cond(node):
        l_node = node.value.next

        if l_node is not None:
            return run_cond(l_node)
        else:
            print('cond null error!')

    def run_cond(node):

        cond = None

        if node.value.type is TokenType.LIST:
            cond = run_expr(node.value)
        else:
            cond = node.value

        if not cond.type in CuteType.BOOLEAN_LIST:
            print "Type Error!"
            return None

        if cond.type is TokenType.TRUE:
            cond.value = True
        elif cond.type is TokenType.FALSE:
            cond.value = False

        if cond.value is False:
            node = node.next
            return run_cond(node)

        return run_expr(node.value.next)

    def create_new_quote_list(value_node, list_flag=False):
        """
        :type value_node: Node
        """
        quote_list = Node(TokenType.QUOTE, 'quote')
        wrapper_new_list = Node(TokenType.LIST, quote_list)
        if value_node is None:
            pass
        elif value_node.type is TokenType.LIST:
            if list_flag:
                inner_l_node = Node(TokenType.LIST, value_node)
                quote_list.next = inner_l_node
            else:
                quote_list.next = value_node
            return wrapper_new_list
        new_value_list = Node(TokenType.LIST, value_node)
        quote_list.next = new_value_list
        return wrapper_new_list

    table = {}
    table['cons'] = cons
    table["'"] = quote
    table['quote'] = quote
    table['cdr'] = cdr
    table['car'] = car
    table['eq?'] = eq_q
    table['null?'] = null_q
    table['atom?'] = atom_q
    table['not'] = not_op
    table['+'] = plus
    table['-'] = minus
    table['*'] = multiple
    table['/'] = divide
    table['<'] = lt
    table['>'] = gt
    table['='] = eq
    table['cond'] = cond
    table['define'] = define
    table['LAMBDA'] = LAMBDA

    if type(op_code_node.value) != str:
        op_code_node = op_code_node.value
        lam = op_code_node.value
        lam = lam.upper()
        op_code_node.value = lam

    if mydict.__contains__(op_code_node.value):
        para = mydict.__getitem__(op_code_node.value)
        nesteddict = dict()

        if type(para.value) is not str:
            if para.value.value == 'define':

                l_node = para.next #cube body
                r_node = para.value #sqrt1

                if r_node.value == 'define':
                    l_node2 = r_node.next.value #sqrt1
                    r_node2 = r_node.next.next

                    if r_node2.value.value == 'lambda':
                        nesteddict.__setitem__(l_node2, r_node2.value.next.next.value)

                getparm = nesteddict.get(l_node2)

                if type(getparm.next.value) == str:
                    getparm.next.value = op_code_node.next.value
                    if type(getparm.next.next.value) == str:
                        getparm.next.next.value = op_code_node.next.value

                    getparm = Node(TokenType.LIST, getparm)
                    nesteddict.__setitem__(l_node2, getparm)
                    getresult = run_expr(nesteddict.get(l_node2))
                    nesteddict.__setitem__(l_node2, getresult.value)


        if para.next.type is not TokenType.INT:
            if para.next.type is TokenType.ID:
                if op_code_node.next.next is not None: # 인자로 함수가 들어오는 경우의 처리를 위한 if문
                    if mydict.__contains__(op_code_node.next.next.value):#첫번째 함수의 값을 받아옴...
                      para = mydict.__getitem__(op_code_node.next.next.value)
                      define_para(para.next.value, op_code_node.next.value)
                    else:
                      define_para(para.next.value, op_code_node.next.value)
                else:
                    define_para(para.next.value, op_code_node.next.value)

            elif para.next.type is TokenType.LIST:#19번과 같이 함수의 내용에 list가 있는 경우
                if type(para.next.value.next.value) is not str:
                    if nesteddict.__contains__(para.next.value.next.value.value):
                        getresult2 = (str)(nesteddict.get(para.next.value.next.value.value))
                        l_node.value.next = Node(TokenType.INT, getresult2)
                        l_node.value.next.next = Node(TokenType.INT, op_code_node.next.value)
                        l_node = Node(TokenType.LIST, l_node.value)
                        mydict.__setitem__(op_code_node.value, l_node)
                        getresult3 = run_expr(mydict.get(op_code_node.value))
                        mydict.__setitem__(op_code_node.value, getresult3.value)
                        print getresult3.value
                        Test_All()

                if op_code_node.next.next is not None:
                    if mydict.__contains__(op_code_node.next.value):
                        para.next.value.value = op_code_node.next.value
                        opresult = result_function(para.next.value, op_code_node.next.next.next.value)
                        para.next = opresult;
                        para2 = mydict.get(op_code_node.next.next.value)
                        if para2.next.type is TokenType.ID:
                            define_para(para2.next.value, opresult)
                        if para2.next.next.type is TokenType.ID:
                            define_para(para2.next.next.value,opresult)
                        para = mydict.get(op_code_node.next.next.value)
                else:
                    opresult = result_function(para.next.value,op_code_node.next.value)
                    opnode = Node(TokenType.INT, opresult)
                    para.next.value = opnode
        else:
            if para.next.next.type is TokenType.ID:
                define_para(para.next.next.value, op_code_node.next.value)

            elif para.next.next.type is TokenType.LIST:
                opresult = result_function(para.next.next.value,op_code_node.next.value)
                opnode = Node(TokenType.INT, opresult)
                para.next.next.value = opnode

        op_code_node.value = para.value
        op_code_node.next = para.next

    return table[op_code_node.value]

def define_para(paraname, paravalue):
    node1 = paraname
    node2 = paravalue
    mydict.__setitem__(node1, node2)

def result_function(paranode, valuenode):
    #paranode 에는 (plus1 x)가, valuenode에는 (plus2 3)의 3이 들어와야함.
    if mydict.__contains__(paranode.value):
        op_node = mydict.__getitem__(paranode.value)
    mydict.__setitem__(paranode.next.value,valuenode)
    op_node_list = Node(TokenType.LIST,op_node)
    return run_list(op_node_list)


def run_expr(root_node):
    """
    :type root_node : Node
    """
    if root_node is None:
        return None

    if root_node.type is TokenType.ID:
        return root_node
    elif root_node.type is TokenType.INT:
        return root_node
    elif root_node.type is TokenType.TRUE:
        return root_node
    elif root_node.type is TokenType.FALSE:
        return root_node
    elif root_node.type is TokenType.LIST:
        return run_list(root_node)
    else:
        print 'Run Expr Error'
    return None


def print_node(node):
    """
    "Evaluation 후 결과를 출력하기 위한 함수"
    "입력은 List Node 또는 atom"
    :type node: Node
    """
    def print_list(node):
        """
        "List노드의 value에 대해서 출력"
        "( 2 3 )이 입력이면 2와 3에 대해서 모두 출력함"
        :type node: Node
        """
        def print_list_val(node):
            if node.next is not None:
                return print_node(node)+' '+print_list_val(node.next)
            return print_node(node)

        if node.type is TokenType.LIST:
            if node.value is None:
                return '( )'
            if node.value.type is TokenType.QUOTE:
                return print_node(node.value)
            return '('+print_list_val(node.value)+')'

    if node is None:
        return ''
    if node is None:
        return ''
    if node.type in [TokenType.ID, TokenType.INT]:
        if (mydict.__contains__(node.value)):
            myCheck = mydict.get(node.value)

            if type(myCheck) == str :
                if (mydict.__contains__(myCheck)):
                    myCheck = mydict.get(myCheck)
                    if type(myCheck) == str:
                        return myCheck
                    elif myCheck.type is TokenType.QUOTE:
                        return print_node(myCheck)
                return myCheck
            elif type(myCheck) == int:
                return myCheck
            elif myCheck.type is TokenType.QUOTE:
                return print_node(myCheck)
            else:
                return myCheck

        return node.value

    if node.type is TokenType.TRUE:
        return '#T'
    if node.type is TokenType.FALSE:
        return '#F'
    if node.type is TokenType.PLUS:
        return '+'
    if node.type is TokenType.MINUS:
        return '-'
    if node.type is TokenType.TIMES:
        return '*'
    if node.type is TokenType.DIV:
        return '/'
    if node.type is TokenType.GT:
        return '>'
    if node.type is TokenType.LT:
        return '<'
    if node.type is TokenType.EQ:
        return '='
    if node.type is TokenType.LIST:
        return print_list(node)
    if node.type is TokenType.ATOM_Q:
        return 'atom?'
    if node.type is TokenType.CAR:
        return 'car'
    if node.type is TokenType.CDR:
        return 'cdr'
    if node.type is TokenType.COND:
        return 'cond'
    if node.type is TokenType.CONS:
        return 'cons'
    if node.type is TokenType.LAMBDA:
        return 'lambda'
    if node.type is TokenType.NULL_Q:
        return 'null?'
    if node.type is TokenType.EQ_Q:
        return 'eq?'
    if node.type is TokenType.NOT:
        return 'not'
    if node.type is TokenType.QUOTE:
        return "'"+print_node(node.next)


def Test_method(input):
    test_cute = CuteScanner(input)
    test_tokens = test_cute.tokenize()
    test_basic_paser = BasicPaser(test_tokens)
    node = test_basic_paser.parse_expr()
    cute_inter = run_expr(node)
    print print_node(cute_inter)


def Test_All():
    while (1):
        a = raw_input("> ")
        print "... ",
        Test_method(a)

Test_All()
