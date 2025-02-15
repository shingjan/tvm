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
 * \file bound_deducer.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace arith {

using namespace tir;

namespace {

using BufferTouches = std::vector<std::vector<IntSet>>;

struct LoadAccess {
  BufferTouches set;
};

struct StoreAccess {
  BufferTouches set;
};

struct CombinedAccess {
  BufferTouches set;
};

using BufferDomainAccess = std::tuple<LoadAccess, StoreAccess, CombinedAccess>;

}  // namespace

// Find Read region of the tensor in the stmt.
class BufferTouchedDomain final : public StmtExprVisitor {
 public:
  BufferTouchedDomain(const Stmt& stmt) { operator()(stmt); }

  std::unordered_map<const BufferNode*, BufferDomainAccess>& GetAccessedBufferRegions() {
    return buffer_access_map_;
  }

  Region FindUnion(const Buffer& buffer, bool consider_loads, bool consider_stores) {
    auto kv = buffer_access_map_.find(buffer.get());
    CHECK(kv != buffer_access_map_.end())
        << "The requested buffer is not contained in the provided stmt body.";

    Region ret;
    Range none;
    BufferTouches bounds;
    if (consider_loads && consider_stores) {
      bounds = std::get<CombinedAccess>(kv->second).set;
    } else if (consider_loads) {
      bounds = std::get<LoadAccess>(kv->second).set;
    } else if (consider_stores) {
      bounds = std::get<StoreAccess>(kv->second).set;
    } else {
      CHECK(false) << "Must consider at least on of either loads and stores, but both are false";
    }
    for (size_t i = 0; i < bounds.size(); ++i) {
      ret.push_back(arith::Union(bounds[i]).CoverRange(none));
    }
    return ret;
  }

  void VisitStmt_(const ForNode* op) final {
    const VarNode* var = op->loop_var.get();
    dom_map_[var] = IntSet::FromRange(Range::FromMinExtent(op->min, op->extent));
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(var);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    dom_map_[op->var.get()] = arith::EvalSet(op->value, dom_map_);
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(op->var.get());
  }

  /* TODO: Thread extent unitest not generated.*/
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      const IterVarNode* thread_axis = op->node.as<IterVarNode>();
      ICHECK(thread_axis);
      const VarNode* var = thread_axis->var.get();
      dom_map_[var] = IntSet::FromRange(Range(make_zero(op->value.dtype()), op->value));
      StmtExprVisitor::VisitStmt_(op);
      dom_map_.erase(var);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    // Record load-exclusive buffer access
    Touch(&std::get<LoadAccess>(buffer_access_map_[op->buffer.get()]).set, op->indices);
    // Record load-store inclusive buffer access
    Touch(&std::get<CombinedAccess>(buffer_access_map_[op->buffer.get()]).set, op->indices);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    // Record store-exclusive buffer access
    Touch(&std::get<StoreAccess>(buffer_access_map_[op->buffer.get()]).set, op->indices);
    // Record load-store inclusive buffer access
    Touch(&std::get<CombinedAccess>(buffer_access_map_[op->buffer.get()]).set, op->indices);
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  template <typename ArrayType>
  void Touch(BufferTouches* bounds, const ArrayType& args) const {
    if (args.size() > bounds->size()) {
      bounds->resize(args.size());
    }
    for (size_t i = 0; i < args.size(); ++i) {
      (*bounds)[i].emplace_back(EvalSet(args[i], dom_map_));
    }
  }

  std::unordered_map<const BufferNode*, BufferDomainAccess> buffer_access_map_;
  std::unordered_map<const VarNode*, IntSet> dom_map_;
};

Region DomainTouched(const Stmt& stmt, const Buffer& buffer, bool consider_loads,
                     bool consider_stores) {
  return BufferTouchedDomain(stmt).FindUnion(buffer, consider_loads, consider_stores);
}

Map<Buffer, runtime::ADT> DomainTouchedAccessMap(const PrimFunc& func) {
  auto buffer_access_map = BufferTouchedDomain(func->body).GetAccessedBufferRegions();
  Map<Buffer, runtime::ADT> ret;
  auto& buffer_map = func->buffer_map;
  for (auto& var : func->params) {
    auto& buffer = buffer_map[var];
    auto& access = buffer_access_map[buffer.get()];
    Array<Array<IntSet>> loads, stores, combined;
    for (std::vector<IntSet>& touch : std::get<LoadAccess>(access).set) {
      loads.push_back(Array<IntSet>(touch));
    }
    for (std::vector<IntSet>& touch : std::get<StoreAccess>(access).set) {
      stores.push_back(Array<IntSet>(touch));
    }
    for (std::vector<IntSet>& touch : std::get<CombinedAccess>(access).set) {
      combined.push_back(Array<IntSet>(touch));
    }

    std::vector<ObjectRef> fields;
    fields.push_back(loads);
    fields.push_back(stores);
    fields.push_back(combined);
    ret.Set(buffer, runtime::ADT::Tuple(fields));
  }
  return ret;
}

TVM_REGISTER_GLOBAL("arith.DomainTouched").set_body_typed(DomainTouched);
TVM_REGISTER_GLOBAL("arith.DomainTouchedAccessMap").set_body_typed(DomainTouchedAccessMap);

}  // namespace arith
}  // namespace tvm
