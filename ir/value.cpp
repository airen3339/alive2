// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "ir/instr.h"
#include "ir/globals.h"
#include "ir/value.h"
#include "smt/expr.h"
#include "util/compiler.h"
#include "util/config.h"

using namespace smt;
using namespace std;
using namespace util;

namespace IR {

VoidValue Value::voidVal;

bool Value::isVoid() const {
  return &getType() == &Type::voidTy;
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
  auto val = getType().getDummyValue(true);
  expr var = expr::mkFreshVar("undef", val.value);
  s.addUndefVar(expr(var));
  StateValue undefv(move(var), move(val.non_poison));
  return undefv;
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
  return { false, false };
}


void NullPointerValue::print(ostream &os) const {
  UNREACHABLE();
}

StateValue NullPointerValue::toSMT(State &s) const {
  auto nullp = Pointer::mkNullPointer(s.getMemory());
  return { nullp.release(), true };
}


void GlobalVariable::print(ostream &os) const {
  os << getName() << " = " << (isconst ? "constant " : "global ") << allocsize
     << " bytes, align " << align;
}

static expr get_global(State &s, const string &name, const expr &size,
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
  return { get_global(s, getName(), size, align, isconst, bid), true };
}


static string agg_str(vector<Value*> &vals) {
  string r = "{ ";
  bool first = true;
  for (auto val : vals) {
    if (!first)
      r += ", ";
    r += val->getName();
    first = false;
  }
  return r + " }";
}

AggregateValue::AggregateValue(Type &type, vector<Value*> &&vals)
  : Value(type, agg_str(vals)), vals(move(vals)) {}

StateValue AggregateValue::toSMT(State &s) const {
  vector<StateValue> state_vals;
  for (auto val : vals) {
    state_vals.emplace_back(val->toSMT(s));
  }
  return getType().getAsAggregateType()->aggregateVals(state_vals);
}

expr AggregateValue::getTypeConstraints() const {
  expr r = Value::getTypeConstraints();
  vector<Type*> types;
  for (auto val : vals) {
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


static string attr_str(unsigned attributes) {
  string ret;
  if (attributes & Input::NonNull)
    ret += "nonnull ";
  if (attributes & Input::ByVal)
    ret += "byval ";
  if (attributes & Input::NoCapture)
    ret += "nocapture ";
  if (attributes & Input::ReadOnly)
    ret += "readonly ";
  return ret;
}

Input::Input(Type &type, string &&name, unsigned attributes)
  : Value(type, attr_str(attributes) + name), smt_name(move(name)),
    attributes(attributes) {}

void Input::copySMTName(const Input &other) {
  smt_name = other.smt_name;
}

void Input::print(ostream &os) const {
  UNREACHABLE();
}

StateValue Input::toSMT(State &s) const {
  // 00: normal, 01: undef, else: poison
  expr type = getTyVar();

  expr val;
  if (attributes & ByVal) {
    unsigned bid;
    string sz_name = getName() + "#size";
    expr size = expr::mkVar(sz_name.c_str(), bits_size_t-1).zext(1);
    val = get_global(s, getName(), size, 1, false, bid);
    s.getMemory().markByVal(bid);
  } else {
    val = getType().mkInput(s, smt_name.c_str(), attributes);
  }

  if (!config::disable_undef_input) {
    auto [undef, vars] = getType().mkUndefInput(s, attributes);
    for (auto &v : vars) {
      s.addUndefVar(move(v));
    }
    val = expr::mkIf(type.extract(0, 0) == 0, val, undef);
  }

  expr poison = getType().getDummyValue(false).non_poison;
  expr non_poison = getType().getDummyValue(true).non_poison;

  return { move(val),
           config::disable_poison_input
             ? move(non_poison)
             : expr::mkIf(type.extract(1, 1) == 0, non_poison, poison) };
}

expr Input::getTyVar() const {
  string tyname = "ty_" + smt_name;
  return expr::mkVar(tyname.c_str(), 2);
}

}
