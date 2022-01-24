/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/tir/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../support/nd_int_set.h"

namespace tvm {
namespace tir {

LinkedParam::LinkedParam(int64_t id, ::tvm::runtime::NDArray param) {
  auto n = make_object<LinkedParamNode>();
  n->id = id;
  n->param = param;
  data_ = std::move(n);
}

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(Array<tir::Var> params, Stmt body, Type ret_type,
                   Map<tir::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
  // Assume void-return type for now
  // TODO(tvm-team) consider type deduction from body.
  if (!ret_type.defined()) {
    ret_type = VoidType();
  }
  auto n = make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->buffer_map = std::move(buffer_map);
  n->attrs = std::move(attrs);
  n->checked_type_ = n->func_type_annotation();
  n->span = std::move(span);
  data_ = std::move(n);
}

FuncType PrimFuncNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(GetType(param));
  }
  return FuncType(param_types, ret_type, {}, {});
}

TVM_REGISTER_NODE_TYPE(PrimFuncNode);

Array<PrimExpr> IndexMapNode::Apply(const Array<PrimExpr>& inputs) const {
  CHECK_EQ(inputs.size(), this->src_iters.size());
  arith::Analyzer analyzer;
  int n = inputs.size();
  for (int i = 0; i < n; ++i) {
    analyzer.Bind(this->src_iters[i], inputs[i]);
  }
  Array<PrimExpr> results;
  results.reserve(this->tgt_iters.size());
  for (PrimExpr result : this->tgt_iters) {
    results.push_back(analyzer.Simplify(std::move(result)));
  }
  return results;
}

Array<PrimExpr> IndexMapNode::MapShape(const Array<PrimExpr>& shape) const {
  using namespace support;
  Array<PrimExpr> indices;
  std::unordered_map<const VarNode*, arith::IntSet> dom_map;
  for (const PrimExpr dim : shape) {
    Var var;
    indices.push_back(var);
    dom_map.emplace(var.get(), arith::IntSet::FromMinExtent(0, dim));
  }
  Array<PrimExpr> mapped_indices = Apply(indices);
  NDIntSet nd_int_set = NDIntSetFromPoint(mapped_indices);
  nd_int_set = NDIntSetEval(nd_int_set, dom_map);
  Array<PrimExpr> new_shape;
  for (const auto& int_set : nd_int_set) {
    ICHECK(is_zero(int_set.min()));
    new_shape.push_back(int_set.max() + 1);
  }
  auto fmul = [](PrimExpr a, PrimExpr b, Span span) { return a * b; };
  PrimExpr old_size = foldl(fmul, Integer(1), shape);
  PrimExpr new_size = foldl(fmul, Integer(1), new_shape);

  arith::Analyzer analyzer;
  CHECK(analyzer.CanProveEqual(old_size, new_size))
      << "ValueError: The size of the new shape after IndexMap " << new_shape
      << " doesn't match the size of the original shape " << shape;
  return new_shape;
}

String IndexMapNode::ToPythonString() const {
  std::unordered_set<std::string> used_names;
  Map<Var, PrimExpr> var_remap;
  for (const Var& src_iter : src_iters) {
    if (used_names.count(src_iter->name_hint)) {
      std::string new_name = src_iter->name_hint + std::to_string(used_names.size());
      used_names.insert(new_name);
      var_remap.Set(src_iter, Var(new_name));
    } else {
      used_names.insert(src_iter->name_hint);
    }
  }
  std::ostringstream oss;
  oss << "lambda ";
  for (size_t i = 0; i < src_iters.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    auto it = var_remap.find(src_iters[i]);
    if (it != var_remap.end()) {
      oss << (*it).second;
    } else {
      oss << src_iters[i];
    }
  }
  oss << ": (";
  for (size_t i = 0; i < tgt_iters.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << Substitute(tgt_iters[i], var_remap);
  }
  if (tgt_iters.size() == 1) {
    oss << ",";
  }
  oss << ")";
  return String(oss.str());
}

IndexMap::IndexMap(Array<Var> src_iters, Array<PrimExpr> tgt_iters) {
  ObjectPtr<IndexMapNode> n = make_object<IndexMapNode>();
  n->src_iters = std::move(src_iters);
  n->tgt_iters = std::move(tgt_iters);
  data_ = std::move(n);
}

IndexMap IndexMap::FromFunc(int ndim, runtime::TypedPackedFunc<Array<PrimExpr>(Array<Var>)> func) {
  Array<Var> src_iters;
  src_iters.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    src_iters.push_back(Var("i" + std::to_string(i), DataType::Int(32)));
  }
  return IndexMap(src_iters, func(src_iters));
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IndexMapNode>([](const ObjectRef& node, ReprPrinter* p) {
      const auto* n = node.as<IndexMapNode>();
      ICHECK(n);
      p->stream << "IndexMap: (";
      for (int i = 0, total = n->src_iters.size(); i < total; ++i) {
        if (i != 0) {
          p->stream << ", ";
        }
        p->stream << n->src_iters[i];
      }
      p->stream << ") => ";
      p->stream << "(";
      for (int i = 0, total = n->tgt_iters.size(); i < total; ++i) {
        if (i != 0) {
          p->stream << ", ";
        }
        p->stream << n->tgt_iters[i];
      }
      p->stream << ")";
    });

TVM_REGISTER_NODE_TYPE(IndexMapNode);
TVM_REGISTER_GLOBAL("tir.IndexMap")
    .set_body_typed([](Array<Var> src_iters, Array<PrimExpr> tgt_iters) {
      return IndexMap(src_iters, tgt_iters);
    });
TVM_REGISTER_GLOBAL("tir.IndexMapFromFunc").set_body_typed(IndexMap::FromFunc);
TVM_REGISTER_GLOBAL("tir.IndexMapApply").set_body_method<IndexMap>(&IndexMapNode::Apply);

class TensorIntrinManager {
 public:
  Map<String, tir::TensorIntrin> reg;

  static TensorIntrinManager* Global() {
    static TensorIntrinManager* inst = new TensorIntrinManager();
    return inst;
  }
};

TensorIntrin::TensorIntrin(PrimFunc desc, PrimFunc impl) {
  // Check the number of func var is equal
  CHECK_EQ(desc->params.size(), impl->params.size())
      << "ValueError: The number of parameters of the description and the implementation of the "
         "tensor intrinsic doesn't match.";
  for (size_t i = 0; i < desc->params.size(); i++) {
    CHECK(desc->params[i]->dtype.is_handle()) << "ValueError: Parameters of the description of the "
                                                 "tensor intrinsic should be handle only.";
    CHECK(impl->params[i]->dtype.is_handle()) << "ValueError: Parameters of the implementation of "
                                                 "the tensor intrinsic should be handle only.";
  }
  ICHECK_EQ(desc->buffer_map.size(), impl->buffer_map.size());

  ObjectPtr<TensorIntrinNode> n = make_object<TensorIntrinNode>();
  n->desc = std::move(desc);
  n->impl = std::move(impl);
  data_ = std::move(n);
}

void TensorIntrin::Register(String name, TensorIntrin intrin) {
  TensorIntrinManager* manager = TensorIntrinManager::Global();
  CHECK_EQ(manager->reg.count(name), 0)
      << "ValueError: TensorIntrin '" << name << "' has already been registered";
  manager->reg.Set(name, intrin);
}

TensorIntrin TensorIntrin::Get(String name) {
  const TensorIntrinManager* manager = TensorIntrinManager::Global();
  auto it = manager->reg.find(name);
  CHECK(it != manager->reg.end()) << "ValueError: TensorIntrin '" << name << "' is not registered";
  return manager->reg.at(name);
}

TVM_REGISTER_NODE_TYPE(TensorIntrinNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& ref, ReprPrinter* p) {
      // TODO(tvm-team) redirect to Text printer once we have a good text format.
      auto* node = static_cast<const PrimFuncNode*>(ref.get());
      p->stream << "PrimFunc(" << node->params << ") ";
      if (node->attrs.defined()) {
        p->stream << "attrs=" << node->attrs;
      }
      p->stream << " {\n";
      p->indent += 2;
      p->Print(node->body);
      p->indent -= 2;
      p->stream << "}\n";
    });

TVM_REGISTER_GLOBAL("tir.PrimFunc")
    .set_body_typed([](Array<tir::Var> params, Stmt body, Type ret_type,
                       Map<tir::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
      return PrimFunc(params, body, ret_type, buffer_map, attrs, span);
    });

TVM_REGISTER_GLOBAL("tir.TensorIntrin")
    .set_body_typed([](PrimFunc desc_func, PrimFunc intrin_func) {
      return TensorIntrin(desc_func, intrin_func);
    });

TVM_REGISTER_GLOBAL("tir.TensorIntrinRegister").set_body_typed(TensorIntrin::Register);
TVM_REGISTER_GLOBAL("tir.TensorIntrinGet").set_body_typed(TensorIntrin::Get);

}  // namespace tir
}  // namespace tvm
