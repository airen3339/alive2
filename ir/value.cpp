// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "ir/value.h"
#include "ir/function.h"
#include "ir/instr.h"
#include "ir/globals.h"
#include "smt/expr.h"
#include "util/compiler.h"
#include "util/config.h"
#include <sstream>

using namespace smt;
using namespace std;
using namespace util;

namespace IR {

VoidValue Value::voidVal;

bool Value::isVoid() const {
  return &getType() == &Type::voidTy;
}

void Value::rauw(const Value &what, Value &with) {
  UNREACHABLE();
}

expr Value::getTypeConstraints() const {
  return getType().getTypeConstraints();
}

void Value::fixupTypes(const Model &m) {
  type.fixup(m);
}

ostream& operator<<(ostream &os, const Value &val) {
  auto t = val.getType().toString();
  os << t;
  if (!val.isVoid()) {
    if (!t.empty()) os << ' ';
    os << val.getName();
  }
  return os;
}


void UndefValue::print(ostream &os) const {
  UNREACHABLE();
}

StateValue UndefValue::toSMT(State &s) const {
  return getType().mkUndef(s);
}


void PoisonValue::print(ostream &os) const {
  UNREACHABLE();
}

StateValue PoisonValue::toSMT(State &s) const {
  return getType().getDummyValue(false);
}


void VoidValue::print(ostream &os) const {
  UNREACHABLE();
}

StateValue VoidValue::toSMT(State &s) const {
  return { false, true };
}


void NullPointerValue::print(ostream &os) const {
  UNREACHABLE();
}

StateValue NullPointerValue::toSMT(State &s) const {
  return { Pointer::mkNullPointer(s.getMemory()).release(), true };
}


void GlobalVariable::print(ostream &os) const {
  os << getName() << " = " << (isconst ? "constant " : "global ");
  if (arbitrary_size)
    os << '?';
  else
    os << allocsize;
  os << " bytes, align " << align;
}

static expr get_global(State &s, const string &name, const expr *size,
                       unsigned align, bool isconst, unsigned &bid) {
  expr ptr;
  bool allocated;
  auto blkkind = isconst ? Memory::CONSTGLOBAL : Memory::GLOBAL;

  if (s.hasGlobalVarBid(name, bid, allocated)) {
    if (!allocated) {
      // Use the same block id that is used by src
      assert(!s.isSource());
      ptr = s.getMemory().alloc(size, align, blkkind, true, true, bid).first;
      s.markGlobalAsAllocated(name);
    } else {
      ptr = Pointer(s.getMemory(), bid, false).release();
    }
  } else {
    ptr = s.getMemory().alloc(size, align, blkkind, true, true, nullopt,
                              &bid).first;
    s.addGlobalVarBid(name, bid);
  }
  return ptr;
}

StateValue GlobalVariable::toSMT(State &s) const {
  unsigned bid;
  expr size = expr::mkUInt(allocsize, bits_size_t);
  return { get_global(s, getName(), arbitrary_size ? nullptr : &size, align,
                      isconst, bid),
           true };
}


static string agg_str(const Type &ty, vector<Value*> &vals) {
  auto agg = ty.getAsAggregateType();
  string r = "{ ";
  unsigned j = 0;
  for (unsigned i = 0, e = agg->numElementsConst(); i != e; ++i) {
    if (i != 0)
      r += ", ";
    if (agg->isPadding(i))
      r += "[padding]";
    else
      r += vals[j++]->getName();
  }
  return r + " }";
}

AggregateValue::AggregateValue(Type &type, vector<Value*> &&vals)
  : Value(type, agg_str(type, vals)), vals(std::move(vals)) {}

StateValue AggregateValue::toSMT(State &s) const {
  vector<StateValue> state_vals;
  for (auto *val : vals) {
    state_vals.emplace_back(val->toSMT(s));
  }
  return getType().getAsAggregateType()->aggregateVals(state_vals);
}

void AggregateValue::rauw(const Value &what, Value &with) {
  for (auto &val : vals) {
    if (val == &what)
      val = &with;
  }
  setName(agg_str(getType(), vals));
}

expr AggregateValue::getTypeConstraints() const {
  expr r = Value::getTypeConstraints();
  vector<Type*> types;
  for (auto *val : vals) {
    types.emplace_back(&val->getType());
    if (dynamic_cast<const Instr*>(val))
      // Instr's type constraints are already generated by BasicBlock's
      // getTypeConstraints()
      continue;
    r &= val->getTypeConstraints();
  }
  return r && getType().enforceAggregateType(&types);
}

void AggregateValue::print(std::ostream &os) const {
  UNREACHABLE();
}


static string attr_str(const ParamAttrs &attr) {
  stringstream ss;
  ss << attr;
  return std::move(ss).str();
}

Input::Input(Type &type, string &&name, ParamAttrs &&attributes)
  : Value(type, attr_str(attributes) + name), smt_name(std::move(name)),
    attrs(std::move(attributes)) {}

void Input::copySMTName(const Input &other) {
  smt_name = other.smt_name;
}

void Input::print(ostream &os) const {
  UNREACHABLE();
}

string Input::getSMTName(unsigned child) const {
  if (getType().isAggregateType())
    return smt_name + '#' + to_string(child);
  assert(child == 0);
  return smt_name;
}

StateValue Input::mkInput(State &s, const Type &ty, unsigned child) const {
  if (auto agg = ty.getAsAggregateType()) {
    vector<StateValue> vals;
    for (unsigned i = 0, e = agg->numElementsConst(); i < e; ++i) {
      if (agg->isPadding(i))
        continue;
      auto name = getSMTName(child + i);
      vals.emplace_back(mkInput(s, agg->getChild(i), child + i));
    }
    return agg->aggregateVals(vals);
  }

  expr val;
  if (hasAttribute(ParamAttrs::ByVal)) {
    unsigned bid;
    expr size = expr::mkUInt(attrs.blockSize, bits_size_t);
    val = get_global(s, smt_name, &size, attrs.align, false, bid);
    bool is_const = hasAttribute(ParamAttrs::NoWrite) ||
                    !s.getFn().getFnAttrs().mem.canWrite(MemoryAccess::Args);
    s.getMemory().markByVal(bid, is_const);
  } else {
    auto name = getSMTName(child);
    val = ty.mkInput(s, name.c_str(), attrs);
  }

  auto undef_mask = getUndefVar(ty, child);
  if (config::disable_undef_input || attrs.poisonImpliesUB()) {
    s.addUB(undef_mask == 0);
  } else {
    auto [undef, var] = ty.mkUndefInput(s, attrs);
    if (undef_mask.bits() == 1)
      val = expr::mkIf(undef_mask == 0, val, undef);
    else
      val = (~undef_mask & val) | (undef_mask & undef);
    s.addUndefVar(std::move(var));
  }

  auto state_val = attrs.encode(s, {std::move(val), expr(true)}, ty);

  bool never_poison = config::disable_poison_input || attrs.poisonImpliesUB();
  expr np = expr::mkBoolVar(("np_" + getSMTName(child)).c_str());
  if (never_poison) {
    s.addUB(std::move(np));
    np = true;
  }

  return { std::move(state_val.value), std::move(state_val.non_poison) && np };
}

bool Input::isUndefMask(const expr &e, const expr &var) {
  auto ty_name = e.fn_name();
  auto var_name = var.fn_name();
  return string_view(ty_name).substr(0, 8) == "isundef_" &&
         string_view(ty_name).substr(8, var_name.size()) == var_name;
}

StateValue Input::toSMT(State &s) const {
  return mkInput(s, getType(), 0);
}

void Input::merge(const ParamAttrs &other) {
  attrs.merge(other);
  setName(attr_str(attrs) + smt_name);
}

expr Input::getUndefVar(const Type &ty, unsigned child) const {
  string tyname = "isundef_" + getSMTName(child);
  //return expr::mkVar(tyname.c_str(), ty.getDummyValue(false).value);
  // FIXME: only whole value undef or non-undef for now
  return expr::mkVar(tyname.c_str(), expr::mkUInt(0, 1));
}

}
