from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Mapping,
    Sequence,
    overload,
)
from enum import IntEnum
from tvm import ir
from tvm.ir import BaseFunc, Range, Span
from tvm.runtime import (
    DataType,
    DataTypeCode,
    Object,
    ObjectGeneric,
)

"""
Redefine types
"""

class PrimExpr:
    def __init__(self: PrimExpr) -> None: ...
    @overload
    def __add__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __sub__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __mul__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __div__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __add__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __radd__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __sub__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __rsub__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __mul__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __rmul__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __div__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __rdiv__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...

"""
buffer
"""

class Buffer(Object):
    READ: int
    WRITE: int
    def access_ptr(
        self, access_mask, ptr_type: str = ..., content_lanes: int = ..., offset: int = ...
    ): ...
    def vload(self, begin, dtype: Any | None = ...): ...
    def vstore(self, begin, value): ...
    def scope(self): ...

def decl_buffer(
    shape,
    dtype: Any | None = ...,
    name: str = ...,
    data: Any | None = ...,
    strides: Any | None = ...,
    elem_offset: Any | None = ...,
    scope: str = ...,
    data_alignment: int = ...,
    offset_factor: int = ...,
    buffer_type: str = ...,
    span: Any | None = ...,
): ...

class DataProducer(Object): ...

"""
data layout
"""

class Layout(Object):
    def __len__(self): ...
    def __contains__(self, axis): ...
    def __getitem__(self, index): ...
    def index_of(self, axis): ...
    def factor_of(self, axis): ...

class BijectiveLayout(Object):
    def forward_index(self, index): ...
    def backward_index(self, index): ...
    def forward_shape(self, shape): ...
    def backward_shape(self, shape): ...

def layout(layout_str: str) -> Layout: ...
def bijective_layout(
    src_layout: Union[str, Layout], dst_layout: Union[str, Layout]
) -> BijectiveLayout: ...

"""
expr
"""
from typing import Any as _Any

def div_ambiguity_error(): ...

class ExprOp:
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __div__(self, other): ...
    def __rdiv__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __mod__(self, other): ...
    def __rmod__(self, other): ...
    def __neg__(self): ...
    def __lshift__(self, other): ...
    def __rlshift__(self, other): ...
    def __rshift__(self, other): ...
    def __rrshift__(self, other): ...
    def __and__(self, other): ...
    def __rand__(self, other): ...
    def __or__(self, other): ...
    def __ror__(self, other): ...
    def __xor__(self, other): ...
    def __rxor__(self, other): ...
    def __invert__(self): ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __nonzero__(self) -> None: ...
    def __bool__(self): ...
    def equal(self, other, span: _Any | None = ...): ...
    def astype(self, dtype: str, span: Optional[Span] = ...): ...

class EqualOp(ObjectGeneric, ExprOp):
    same_as: _Any
    a: _Any
    b: _Any
    span: _Any
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...
    def __nonzero__(self): ...
    def __bool__(self): ...
    def asobject(self): ...

class NotEqualOp(ObjectGeneric, ExprOp):
    same_as: _Any
    a: _Any
    b: _Any
    span: _Any
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...
    def __nonzero__(self): ...
    def __bool__(self): ...
    def asobject(self): ...

class IntImmEnum(ObjectGeneric):
    value: _Any
    span: _Any
    def __init__(self, value, span: _Any | None = ...) -> None: ...
    def asobject(self): ...

class PrimExprWithOp(ExprOp, PrimExpr):
    __hash__: _Any

class ConstExpr(PrimExprWithOp): ...
class BinaryOpExpr(PrimExprWithOp): ...
class CmpExpr(PrimExprWithOp): ...
class LogicalExpr(PrimExprWithOp): ...

class Var(PrimExprWithOp):
    def __init__(
        self, name: str, dtype: Union[str, ir.Type], span: Optional[Span] = ...
    ) -> None: ...

class SizeVar(Var):
    def __init__(self, name, dtype, span: _Any | None = ...) -> None: ...

class IterVar(Object, ExprOp):
    DataPar: int
    ThreadIndex: int
    CommReduce: int
    Ordered: int
    DimInfo: int
    Unrolled: int
    Vectorized: int
    Parallelized: int
    Tensorized: int
    def __init__(
        self, dom, var, iter_type, thread_tag: str = ..., span: _Any | None = ...
    ) -> None: ...

class CommReducer(Object):
    def __init__(self, lhs, rhs, result, identity_element, span: _Any | None = ...) -> None: ...

class Reduce(PrimExprWithOp):
    def __init__(
        self,
        combiner,
        src,
        rdom,
        condition,
        value_index,
        init: _Any | None = ...,
        span: _Any | None = ...,
    ) -> None: ...

class FloatImm(ConstExpr):
    def __init__(self, dtype, value, span: _Any | None = ...) -> None: ...
    def __float__(self): ...

class IntImm(ConstExpr):
    def __init__(self, dtype, value, span: _Any | None = ...) -> None: ...
    def __hash__(self): ...
    def __int__(self): ...
    def __nonzero__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __bool__(self): ...

class StringImm(ConstExpr):
    def __init__(self, value, span: _Any | None = ...) -> None: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __hash__(self): ...

class Cast(PrimExprWithOp):
    def __init__(self, dtype, value, span: _Any | None = ...) -> None: ...

class Add(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Sub(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Mul(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Div(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Mod(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class FloorDiv(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class FloorMod(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Min(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Max(BinaryOpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class EQ(CmpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class NE(CmpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class LT(CmpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class LE(CmpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class GT(CmpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class GE(CmpExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class And(LogicalExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Or(LogicalExpr):
    def __init__(self, a, b, span: _Any | None = ...) -> None: ...

class Not(LogicalExpr):
    def __init__(self, a, span: _Any | None = ...) -> None: ...

class Select(PrimExprWithOp):
    def __init__(self, condition, true_value, false_value, span: _Any | None = ...) -> None: ...

class Load(PrimExprWithOp):
    def __init__(
        self, dtype, buffer_var, index, predicate: _Any | None = ..., span: _Any | None = ...
    ) -> None: ...

class BufferLoad(PrimExprWithOp):
    def __init__(self, buffer, indices, span: _Any | None = ...) -> None: ...

class ProducerLoad(PrimExprWithOp):
    def __init__(self, producer, indices, span: _Any | None = ...) -> None: ...

class Ramp(PrimExprWithOp):
    def __init__(self, base, stride, lanes, span: _Any | None = ...) -> None: ...

class Broadcast(PrimExprWithOp):
    def __init__(self, value, lanes, span: _Any | None = ...) -> None: ...

class Shuffle(PrimExprWithOp):
    def __init__(self, vectors, indices, span: _Any | None = ...) -> None: ...

class CallEffectKind:
    ExprAnnotation: _Any
    Pure: _Any
    ReadState: _Any
    UpdateState: _Any
    Opaque: _Any

class Call(PrimExprWithOp):
    def __init__(self, dtype, op, args, span: _Any | None = ...) -> None: ...

class Let(PrimExprWithOp):
    def __init__(self, var, value, body, span: _Any | None = ...) -> None: ...

class Any(PrimExprWithOp):
    def __init__(self, span: _Any | None = ...) -> None: ...

"""
function
"""

class PrimFunc(BaseFunc):
    def __init__(
        self,
        params,
        body,
        ret_type: Any | None = ...,
        buffer_map: Any | None = ...,
        attrs: Any | None = ...,
        span: Any | None = ...,
    ) -> None: ...
    def with_body(self, new_body, span: Any | None = ...): ...
    def specialize(self, param_map: Mapping[Var, Union[PrimExpr, Buffer]]): ...
    def script(self, tir_prefix: str = ..., show_meta: bool = ...) -> str: ...

"""
generic
"""

def add(lhs, rhs, span: Any | None = ...): ...
def subtract(lhs, rhs, span: Any | None = ...): ...
def multiply(lhs, rhs, span: Any | None = ...): ...
def divide(lhs, rhs, span: Any | None = ...): ...
def floordiv(lhs, rhs, span: Any | None = ...): ...
def cast(src, dtype, span: Any | None = ...): ...

"""
ir_builder
"""

class WithScope:
    def __init__(self, enter_value, exit_cb) -> None: ...
    def __enter__(self): ...
    def __exit__(self, ptype, value, trace) -> None: ...

class BufferVar(ObjectGeneric):
    def __init__(self, builder, buffer_var, shape, content_type) -> None: ...
    def asobject(self): ...
    @property
    def dtype(self): ...
    def __getitem__(self, index): ...
    def __setitem__(self, index, value) -> None: ...

class IRBuilder:
    nidx: int
    def __init__(self) -> None: ...
    def emit(self, stmt) -> None: ...
    def scope_attr(self, node, attr_key, value): ...
    def for_range(self, begin, end, name: str = ..., dtype: str = ..., kind: str = ...): ...
    def while_loop(self, condition): ...
    def if_scope(self, cond): ...
    def else_scope(self): ...
    def new_scope(self): ...
    def let(self, var_name, value): ...
    def allocate(self, dtype, shape, name: str = ..., scope: str = ...): ...
    def pointer(self, content_type, name: str = ..., scope: str = ...): ...
    def buffer_ptr(self, buf, shape: Any | None = ...): ...
    def likely(self, expr): ...
    def get(self): ...

def create(): ...

"""
op
"""

def call_packed(*args, span: Any | None = ...): ...
def call_intrin(dtype, func_name, *args, span: Any | None = ...): ...
def call_pure_extern(dtype, func_name, *args, span: Any | None = ...): ...
def call_extern(dtype, func_name, *args, span: Any | None = ...): ...
def call_llvm_intrin(dtype, name, *args, span: Any | None = ...): ...
def call_llvm_pure_intrin(dtype, name, *args, span: Any | None = ...): ...
def ret(val): ...
def any(*args, span: Any | None = ...): ...
def all(*args, span: Any | None = ...): ...
def trace(args, trace_action: str = ...): ...
def min_value(dtype, span: Any | None = ...): ...
def max_value(dtype: str, span: Optional[Span] = ...) -> Any: ...
def exp(x): ...
def exp2(x): ...
def exp10(x): ...
def erf(x): ...
def tanh(x): ...
def sigmoid(x): ...
def log(x): ...
def log2(x): ...
def log10(x): ...
def log1p(x): ...
def tan(x): ...
def cos(x): ...
def cosh(x): ...
def acos(x): ...
def acosh(x): ...
def sin(x): ...
def sinh(x): ...
def asin(x): ...
def asinh(x): ...
def atan(x): ...
def atanh(x): ...
def atan2(x1, x2): ...
def sqrt(x): ...
def rsqrt(x): ...
def clz(x): ...
def floor(x: PrimExprWithOp, span: Any | None = ...): ...
def ceil(x, span: Any | None = ...): ...
def trunc(x, span: Any | None = ...): ...
def abs(x, span: Any | None = ...): ...
def round(x, span: Any | None = ...): ...
def nearbyint(x, span: Any | None = ...): ...
def nextafter(x1, x2): ...
def hypot(x1, x2): ...
def copysign(x1, x2): ...
def ldexp(x1, x2): ...
def isnan(x, span: Any | None = ...): ...
def isfinite(x, span: Any | None = ...): ...
def isinf(x, span: Any | None = ...): ...
def power(x, y, span: Any | None = ...): ...
def popcount(x): ...
def q_multiply_shift(x, y, q, s): ...
def fmod(x, y): ...
def if_then_else(cond, t, f, span: Any | None = ...): ...
def div(a, b, span: Any | None = ...): ...
def indexdiv(a, b, span: Any | None = ...): ...
def indexmod(a, b, span: Any | None = ...): ...
def truncdiv(a, b, span: Any | None = ...): ...
def truncmod(a, b, span: Any | None = ...): ...
def floordiv(a, b, span: Any | None = ...): ...
def floormod(a, b, span: Any | None = ...): ...
def comm_reducer(fcombine, fidentity, name: str = ...): ...

"""
stmt_functor
"""

def ir_transform(stmt, preorder, postorder, only_enable: Any | None = ...): ...
def post_order_visit(stmt, fvisit): ...
def substitute(node, vmap): ...

"""
stmt
"""

class Stmt(Object): ...

class LetStmt(Stmt):
    def __init__(self, var, value, body, span: Any | None = ...) -> None: ...

class AssertStmt(Stmt):
    def __init__(self, condition, message, body, span: Any | None = ...) -> None: ...

class ForKind(IntEnum):
    SERIAL: int
    PARALLEL: int
    VECTORIZED: int
    UNROLLED: int
    THREAD_BINDING: int

class For(Stmt):
    def __init__(
        self,
        loop_var,
        min_val,
        extent,
        kind,
        body,
        thread_binding: Any | None = ...,
        annotations: Any | None = ...,
        span: Any | None = ...,
    ) -> None: ...

class While(Stmt):
    def __init__(self, condition, body, span: Any | None = ...) -> None: ...

class Store(Stmt):
    def __init__(
        self, buffer_var, value, index, predicate: Any | None = ..., span: Any | None = ...
    ) -> None: ...

class BufferStore(Stmt):
    def __init__(self, buffer, value, indices, span: Any | None = ...) -> None: ...

class BufferRealize(Stmt):
    def __init__(self, buffer, bounds, condition, body, span: Any | None = ...) -> None: ...

class ProducerStore(Stmt):
    def __init__(self, producer, value, indices, span: Any | None = ...) -> None: ...

class Allocate(Stmt):
    def __init__(
        self,
        buffer_var,
        dtype,
        extents,
        condition,
        body,
        annotations: Any | None = ...,
        span: Any | None = ...,
    ) -> None: ...

class AttrStmt(Stmt):
    def __init__(self, node, attr_key, value, body, span: Any | None = ...) -> None: ...

class ProducerRealize(Stmt):
    def __init__(
        self, producer, bounds, condition, body, storage_scope: str = ..., span: Any | None = ...
    ) -> None: ...

class SeqStmt(Stmt):
    def __init__(self, seq, span: Any | None = ...) -> None: ...
    def __getitem__(self, i): ...
    def __len__(self): ...

class IfThenElse(Stmt):
    def __init__(self, condition, then_case, else_case, span: Any | None = ...) -> None: ...

class Evaluate(Stmt):
    def __init__(self, value, span: Any | None = ...) -> None: ...

class Prefetch(Stmt):
    def __init__(self, buffer, bounds, span: Any | None = ...) -> None: ...

class BufferRegion(Object):
    buffer: Buffer
    region: List[Range]
    def __init__(self, buffer: Buffer, region: List[Range]) -> None: ...

class MatchBufferRegion(Object):
    buffer: Buffer
    source: BufferRegion
    def __init__(self, buffer: Buffer, source: BufferRegion) -> None: ...

class Block(Stmt):
    iter_vars: List[IterVar]
    reads: List[BufferRegion]
    writes: List[BufferRegion]
    name_hint: str
    body: Stmt
    init: Optional[Stmt]
    alloc_buffers: Optional[List[Buffer]]
    match_buffers: Optional[List[MatchBufferRegion]]
    annotations: Optional[Mapping[str, Object]]
    span: Optional[Span]
    def __init__(
        self,
        iter_vars: List[IterVar],
        reads: List[BufferRegion],
        writes: List[BufferRegion],
        name_hint: str,
        body: Stmt,
        init: Optional[Stmt] = ...,
        alloc_buffers: Optional[List[Buffer]] = ...,
        match_buffers: Optional[List[MatchBufferRegion]] = ...,
        annotations: Optional[Mapping[str, Object]] = ...,
        span: Optional[Span] = ...,
    ) -> None: ...

class BlockRealize(Stmt):
    iter_values: List[PrimExpr]
    predicate: PrimExpr
    block: Block
    span: Optional[Span]
    def __init__(
        self,
        iter_values: List[PrimExpr],
        predicate: Union[PrimExpr, bool],
        block: Block,
        span: Optional[Span] = ...,
    ) -> None: ...

def stmt_seq(*args): ...
def stmt_list(stmt): ...