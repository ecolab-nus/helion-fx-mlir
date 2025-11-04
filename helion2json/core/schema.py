from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, Field, ValidationError, root_validator, validator

ScalarExpr = Union[int, str]
AllowedDtypes = {"f16", "bf16", "f32", "f64", "i32", "i64"}


class TensorTypeSpec(BaseModel):
    shape: List[ScalarExpr]
    elem: str

    @validator("elem")
    def validate_elem(cls, value: str) -> str:
        if value not in AllowedDtypes:
            raise ValueError(f"Unsupported element type '{value}'")
        return value


class TypeSpec(BaseModel):
    tensor: Optional[TensorTypeSpec] = None
    memref: Optional[TensorTypeSpec] = None

    @root_validator
    def ensure_single_variant(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        active = [key for key in ("tensor", "memref") if values.get(key) is not None]
        if len(active) != 1:
            raise ValueError("Type spec must define exactly one of {'tensor', 'memref'}")
        return values


class ArgumentSpec(BaseModel):
    id: str
    type: TypeSpec


class ReturnSpec(BaseModel):
    type: TypeSpec


class SliceSpec(BaseModel):
    base: str
    offsets: List[ScalarExpr]
    sizes: Optional[List[ScalarExpr]] = None

    @root_validator
    def check_sizes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        offsets: List[ScalarExpr] = values.get("offsets") or []
        sizes: Optional[List[ScalarExpr]] = values.get("sizes")
        if sizes is not None and len(offsets) != len(sizes):
            raise ValueError("Slice 'offsets' and 'sizes' must have the same length")
        return values


class SliceArg(BaseModel):
    slice: SliceSpec


class HlAllocOp(BaseModel):
    op: str
    shape: List[ScalarExpr]
    dtype: str
    result: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.alloc":
            raise ValueError("hl.alloc op expected")
        return value

    @validator("dtype")
    def validate_dtype(cls, value: str) -> str:
        if value not in AllowedDtypes:
            raise ValueError(f"Unsupported dtype '{value}'")
        return value


class HlTileBeginOp(BaseModel):
    op: str
    iters: List[str]
    sizes: List[ScalarExpr]
    result: List[str]

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.tile.begin":
            raise ValueError("hl.tile.begin op expected")
        return value

    @root_validator
    def check_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        iters = values.get("iters") or []
        sizes = values.get("sizes") or []
        results = values.get("result") or []
        if not (len(iters) == len(sizes) == len(results)):
            raise ValueError("hl.tile.begin requires matching iters/sizes/result lengths")
        return values


class HlTileEndOp(BaseModel):
    op: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.tile.end":
            raise ValueError("hl.tile.end op expected")
        return value


class HlZerosOp(BaseModel):
    op: str
    shape: List[ScalarExpr]
    dtype: str
    result: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.zeros":
            raise ValueError("hl.zeros op expected")
        return value

    @validator("dtype")
    def validate_dtype(cls, value: str) -> str:
        if value not in AllowedDtypes:
            raise ValueError(f"Unsupported dtype '{value}'")
        return value


class HlStoreSliceOp(BaseModel):
    op: str
    dst: str
    offsets: List[ScalarExpr]
    src: str
    sizes: Optional[List[ScalarExpr]] = None

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.store_slice":
            raise ValueError("hl.store_slice op expected")
        return value

    @root_validator
    def check_sizes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        offsets: List[ScalarExpr] = values.get("offsets") or []
        sizes: Optional[List[ScalarExpr]] = values.get("sizes")
        if sizes is not None and len(offsets) != len(sizes):
            raise ValueError("Offsets and sizes must have the same length")
        return values


class HlReturnOp(BaseModel):
    op: str
    values: List[str]

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.return":
            raise ValueError("hl.return op expected")
        return value


class HlAssertEqOp(BaseModel):
    op: str
    lhs: str
    rhs: str
    msg: str

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "hl.assert_eq":
            raise ValueError("hl.assert_eq op expected")
        return value


class TorchAddmmOp(BaseModel):
    op: str
    result: str
    args: List[Union[str, SliceArg]]
    attrs: Dict[str, float] = Field(default_factory=dict)

    @validator("op")
    def validate_op(cls, value: str) -> str:
        if value != "torch.addmm":
            raise ValueError("torch.addmm op expected")
        return value

    @validator("args")
    def validate_args(cls, value: List[Union[str, SliceArg]]) -> List[Union[str, SliceArg]]:
        if not value:
            raise ValueError("torch.addmm requires at least one argument")
        return value


Operation = Union[
    HlAllocOp,
    HlTileBeginOp,
    HlTileEndOp,
    HlZerosOp,
    HlStoreSliceOp,
    HlReturnOp,
    HlAssertEqOp,
    TorchAddmmOp,
]


class FunctionSpec(BaseModel):
    name: str
    args: List[ArgumentSpec]
    rets: List[ReturnSpec]
    body: List[Operation]


class ModuleSpec(BaseModel):
    name: str
    funcs: List[FunctionSpec]


class HelionJsonSpec(BaseModel):
    version: int
    module: ModuleSpec


def validate_spec(payload: Mapping[str, Any]) -> HelionJsonSpec:
    """Validate a Helion JSON payload against the interchange schema."""
    return HelionJsonSpec.parse_obj(payload)


__all__ = ["HelionJsonSpec", "validate_spec", "ValidationError"]
