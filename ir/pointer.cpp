// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

#include "ir/pointer.h"
#include "ir/function.h"
#include "ir/memory.h"
#include "ir/globals.h"
#include "ir/state.h"

using namespace IR;
using namespace smt;
using namespace std;
using namespace util;

static bool hasLogicalBit() {
  return has_int2ptr;
}

static unsigned total_bits_logical() {
  return hasLogicalBit() + bits_for_ptrattrs + bits_for_bid + bits_for_offset;
}

static unsigned total_bits_physical() {
  return hasLogicalBit() * (1 + bits_for_ptrattrs + bits_ptr_address);
}

static unsigned padding_logical() {
  auto l = total_bits_logical();
  auto p = total_bits_physical();
  return l > p ? 0 : p - l;
}

static unsigned padding_physical() {
  auto l = total_bits_logical();
  auto p = total_bits_physical();
  return p > l ? 0 : l - p;
}

static expr prepend_if(const expr &pre, expr &&e, bool prepend) {
  return prepend ? pre.concat(e) : std::move(e);
}

static string local_name(const State *s, const char *name) {
  return string(name) + (s->isSource() ? "_src" : "_tgt");
}

// Assumes that both begin + len don't overflow
static expr disjoint(const expr &begin1, const expr &len1, const expr &begin2,
                     const expr &len2) {
  return begin1.uge(begin2 + len2) || begin2.uge(begin1 + len1);
}

static expr attr_to_bitvec(const ParamAttrs &attrs) {
  if (!bits_for_ptrattrs)
    return {};

  uint64_t bits = 0;
  auto idx = 0;
  auto add = [&](bool b, const ParamAttrs::Attribute &a) {
    bits |= b ? (attrs.has(a) << idx++) : 0;
  };
  add(has_nocapture, ParamAttrs::NoCapture);
  add(has_noread, ParamAttrs::NoRead);
  add(has_nowrite, ParamAttrs::NoWrite);
  add(has_ptr_arg, ParamAttrs::IsArg);
  return expr::mkUInt(bits, bits_for_ptrattrs);
}

namespace IR {

Pointer::Pointer(const Memory &m, const expr &bid, const expr &offset,
                 const expr &attr) : m(m),
  p(prepend_if(expr::mkUInt(0, 1 + padding_logical()),
               bid.concat(offset), hasLogicalBit())) {
  if (bits_for_ptrattrs)
    p = p.concat(attr);
  assert(!bid.isValid() || !offset.isValid() || p.bits() == totalBits());
}

Pointer::Pointer(const Memory &m, const char *var_name,
                 const ParamAttrs &attr) : m(m) {
  unsigned bits = bitsShortBid() + bits_for_offset;
  p = prepend_if(expr::mkUInt(0, 1 + padding_logical()),
                 expr::mkVar(var_name, bits, false), hasLocalBit());
  if (bits_for_ptrattrs)
    p = p.concat(attr_to_bitvec(attr));
  assert(p.bits() == totalBits());
}

Pointer::Pointer(const Memory &m, expr repr) : m(m), p(std::move(repr)) {
  assert(!p.isValid() || p.bits() == totalBits());
}

Pointer::Pointer(const Memory &m, unsigned bid, bool local)
  : m(m), p(
    prepend_if(expr::mkUInt(0, 1 + padding_logical()),
      prepend_if(expr::mkUInt(local, 1),
                 expr::mkUInt(bid, bitsShortBid())
                   .concat_zeros(bits_for_offset + bits_for_ptrattrs),
                 hasLocalBit()),
              hasLogicalBit())) {
  assert((local && bid < m.numLocals()) || (!local && bid < num_nonlocals));
  assert(p.bits() == totalBits());
}

Pointer::Pointer(const Memory &m, const expr &bid, const expr &offset,
                 const ParamAttrs &attr)
  : Pointer(m, bid, offset, attr_to_bitvec(attr)) {}

Pointer Pointer::mkPhysical(const Memory &m, const expr &addr) {
  return mkPhysical(m, addr, expr::mkUInt(0, bits_for_ptrattrs));
}

Pointer Pointer::mkPhysical(const Memory &m, const expr &addr,
                            const expr &attr) {
  assert(hasLogicalBit());
  assert(addr.bits() == bits_ptr_address);
  auto p = expr::mkUInt(1, 1)
           .concat_zeros(padding_physical())
           .concat(addr);
  if (bits_for_ptrattrs)
    p = p.concat(attr);
  return { m, std::move(p) };
}

expr Pointer::mkLongBid(const expr &short_bid, bool local) {
  assert((local  && (num_locals_src || num_locals_tgt)) ||
         (!local && num_nonlocals));
  if (!hasLocalBit()) {
    return short_bid;
  }
  return expr::mkUInt(local, 1).concat(short_bid);
}

expr Pointer::mkUndef(State &s) {
  auto &m = s.getMemory();
  bool force_local = false, force_nonlocal = false;
  if (hasLocalBit()) {
    force_nonlocal = m.numLocals() == 0;
    force_local    = !force_nonlocal && m.numNonlocals() == 0;
  }

  unsigned var_bits = bits_for_bid + bits_for_offset
                        - (force_local | force_nonlocal);
  expr var = expr::mkFreshVar("undef", expr::mkUInt(0, var_bits));
  s.addUndefVar(expr(var));

  if (force_local || force_nonlocal)
    var = mkLongBid(var, force_local);

  // TODO: support undef phy pointers
  return expr::mkUInt(0, 1 + padding_logical())
           .concat(var).concat_zeros(bits_for_ptrattrs);
}

unsigned Pointer::totalBits() {
  return max(total_bits_logical(), total_bits_physical());
}

unsigned Pointer::bitsShortBid() {
  return bits_for_bid - hasLocalBit();
}

unsigned Pointer::bitsShortOffset() {
  return bits_for_offset - zeroBitsShortOffset();
}

unsigned Pointer::zeroBitsShortOffset() {
  assert(is_power2(bits_byte));
  return ilog2(bits_byte / 8);
}

bool Pointer::hasLocalBit() {
  return (num_locals_src || num_locals_tgt) && num_nonlocals;
}

expr Pointer::isLogical() const {
  return hasLogicalBit() ? p.sign() == 0 : true;
}

expr Pointer::isLocal(bool simplify) const {
  if (m.numLocals() == 0)
    return false;
  if (m.numNonlocals() == 0)
    return true;

  auto bit = bits_for_bid - 1 + bits_for_offset + bits_for_ptrattrs;
  expr local = p.extract(bit, bit);

  if (simplify) {
    switch (m.isInitialMemBlock(local, true)) {
      case 0:  break;
      case 1:  return false;
      case 2:
        // If no local escaped, pointers written to memory by a callee can't
        // alias with a local pointer.
        if (!m.hasEscapedLocals())
          return false;
        break;
      default: UNREACHABLE();
    }
  }

  return local == 1;
}

expr Pointer::isConstGlobal() const {
  auto bid = getShortBid();
  auto generic = bid.uge(has_null_block) &&
                 expr(num_consts_src > 0) &&
                 bid.ule(num_consts_src + has_null_block - 1);
  auto tgt
    = (num_nonlocals_src == 0 ? expr(true) : bid.ugt(num_nonlocals_src-1)) &&
      expr(num_nonlocals != num_nonlocals_src);
  return !isLocal() && (generic || tgt);
}

expr Pointer::isWritableGlobal() const {
  auto bid = getShortBid();
  return !isLocal() &&
         bid.uge(has_null_block + num_consts_src) &&
         expr(num_globals_src > 0) &&
         bid.ule(has_null_block + num_globals_src - 1);
}

expr Pointer::getBid() const {
  auto start = bits_for_offset + bits_for_ptrattrs;
  return p.extract(start + bits_for_bid - 1, start);
}

expr Pointer::getShortBid() const {
  auto start = bits_for_offset + bits_for_ptrattrs;
  return p.extract(start + bits_for_bid - 1 - hasLocalBit(), start);
}

expr Pointer::getOffset() const {
  return p.extract(bits_for_offset + bits_for_ptrattrs - 1, bits_for_ptrattrs);
}

expr Pointer::getOffsetSizet() const {
  auto off = getOffset();
  return bits_for_offset >= bits_size_t ? off : off.sextOrTrunc(bits_size_t);
}

expr Pointer::getShortOffset() const {
  return p.extract(bits_for_offset + bits_for_ptrattrs - 1,
                   bits_for_ptrattrs + zeroBitsShortOffset());
}

expr Pointer::getAttrs() const {
  return bits_for_ptrattrs ? p.extract(bits_for_ptrattrs - 1, 0) : expr();
}

expr Pointer::getValue(const char *name, const FunctionExpr &local_fn,
                       const FunctionExpr &nonlocal_fn,
                       const expr &ret_type, bool src_name) const {
  if (!p.isValid())
    return {};

  auto bid = getShortBid();
  expr non_local;

  if (auto val = nonlocal_fn.lookup(bid))
    non_local = *val;
  else {
    string uf = src_name ? local_name(m.state, name) : name;
    non_local = expr::mkUF(uf.c_str(), { bid }, ret_type);
  }

  if (auto local = local_fn(bid))
    return expr::mkIf(isLocal(), *local, non_local);
  return non_local;
}

expr Pointer::getBlockBaseAddress(bool simplify) const {
  assert(Memory::observesAddresses());

  auto bid = getShortBid();
  auto zero = expr::mkUInt(0, bits_ptr_address - hasLocalBit());
  // fast path for null ptrs
  auto non_local
    = simplify && bid.isZero() && has_null_block ?
          zero : expr::mkUF("blk_addr", { bid }, zero);
  // Non-local block area is the lower half
  if (hasLocalBit())
    non_local = expr::mkUInt(0, 1).concat(non_local);

  if (auto local = m.local_blk_addr(bid)) {
    // Local block area is the upper half of the memory
    expr lc = hasLocalBit() ? expr::mkUInt(1, 1).concat(*local) : *local;
    return expr::mkIf(isLocal(), lc, non_local);
  } else
    return non_local;
}

expr Pointer::getLogAddress(bool simplify) const {
  return getBlockBaseAddress(simplify)
           + getOffset().sextOrTrunc(bits_ptr_address);
}

expr Pointer::getAddress(bool simplify) const {
  return mkIf_fold(isLogical(), getPhysicalAddress(), getLogAddress(simplify));
}

expr Pointer::getPhysicalAddress() const {
  return p.extract(bits_for_ptrattrs + bits_ptr_address - 1, bits_for_ptrattrs);
}

expr Pointer::blockSize() const {
  return getValue("blk_size", m.local_blk_size, m.non_local_blk_size,
                  expr::mkUInt(0, bits_size_t));
}

expr Pointer::blockSizeOffsetT() const {
  expr sz = blockSize();
  return bits_for_offset > bits_size_t ? sz.zextOrTrunc(bits_for_offset) : sz;
}

expr Pointer::reprWithoutAttrs() const {
  return p.extract(totalBits() - 1, bits_for_ptrattrs);
}

Pointer Pointer::mkPointerFromNoAttrs(const Memory &m, const expr &e) {
  return { m, e.concat_zeros(bits_for_ptrattrs) };
}

Pointer Pointer::operator+(const expr &bytes) const {
  return mkIf_fold(isLogical(),
    mkPhysical(m, getPhysicalAddress() + bytes.zextOrTrunc(bits_ptr_address),
               getAttrs()),
    (Pointer{m, getBid(),
             getOffset() + bytes.zextOrTrunc(bits_for_offset), getAttrs()}));
}

Pointer Pointer::operator+(unsigned bytes) const {
  return *this + expr::mkUInt(bytes, bits_for_offset);
}

void Pointer::operator+=(const expr &bytes) {
  *this = *this + bytes;
}

Pointer Pointer::maskOffset(const expr &mask) const {
  return mkIf_fold(isLogical(),
    mkPhysical(m, getPhysicalAddress() & mask.zextOrTrunc(bits_ptr_address),
               getAttrs()),
    (Pointer{ m, getBid(),
              ((getLogAddress() & mask.zextOrTrunc(bits_ptr_address))
                 - getBlockBaseAddress()).zextOrTrunc(bits_for_offset),
              getAttrs() }));
}

expr Pointer::addNoUSOverflow(const expr &offset, bool offset_only) const {
  if (offset_only)
    return getOffset().add_no_soverflow(offset);
  return getAddress().add_no_usoverflow(offset.sextOrTrunc(bits_ptr_address));
}

expr Pointer::addNoUOverflow(const expr &offset, bool offset_only) const {
  if (offset_only)
    return getOffset().add_no_uoverflow(offset);
  return getAddress().add_no_uoverflow(offset.sextOrTrunc(bits_ptr_address));
}

expr Pointer::operator==(const Pointer &rhs) const {
  return reprWithoutAttrs() == rhs.reprWithoutAttrs();
}

expr Pointer::operator!=(const Pointer &rhs) const {
  return !operator==(rhs);
}

expr Pointer::isOfBlock(const Pointer &block, const expr &bytes0) const {
  assert(block.getOffset().isZero());
  expr bytes = bytes0.zextOrTrunc(bits_ptr_address);
  expr addr  = getAddress();
  expr block_addr = block.getAddress();
  expr block_size = block.blockSize().zextOrTrunc(bits_ptr_address);

  if (bytes.eq(block_size))
    return addr == block_addr;
  if (bytes.ugt(block_size).isTrue())
    return false;

  return addr.uge(block_addr) && addr.ult(block_addr + block_size);
}

expr Pointer::isInboundsOf(const Pointer &block, const expr &bytes0) const {
  assert(block.getOffset().isZero());
  expr bytes = bytes0.zextOrTrunc(bits_ptr_address);
  expr addr  = getAddress();
  expr block_addr = block.getAddress();
  expr block_size = block.blockSize().zextOrTrunc(bits_ptr_address);

  if (bytes.eq(block_size))
    return addr == block_addr;
  if (bytes.ugt(block_size).isTrue())
    return false;

  return addr.uge(block_addr) &&
         addr.add_no_uoverflow(bytes) &&
         (addr + bytes).ule(block_addr + block_size);
}

expr Pointer::isInboundsOf(const Pointer &block, unsigned bytes) const {
  return isInboundsOf(block, expr::mkUInt(bytes, bits_ptr_address));
}

expr Pointer::isInbounds(bool strict) const {
  auto offset = getOffsetSizet();
  auto size   = blockSizeOffsetT();
  return (strict ? offset.ult(size) : offset.ule(size)) && !offset.isNegative();
}

expr Pointer::inbounds() {
  DisjointExpr<expr> ret(expr(false)), all_ptrs;
  for (auto &[ptr_expr, domain] : DisjointExpr<expr>(p, 3)) {
    expr inb = Pointer(m, ptr_expr).isInbounds(false);
    if (!inb.isFalse())
      all_ptrs.add(ptr_expr, domain);
    ret.add(std::move(inb), domain);
  }

  // trim set of valid ptrs
  auto ptrs = std::move(all_ptrs)();
  p = ptrs ? *std::move(ptrs) : expr::mkUInt(0, totalBits());

  return *std::move(ret)();
}

expr Pointer::blockAlignment() const {
  return getValue("blk_align", m.local_blk_align, m.non_local_blk_align,
                   expr::mkUInt(0, 6), true);
}

expr Pointer::isBlockAligned(uint64_t align, bool exact) const {
  if (!exact && align == 1)
    return true;

  auto bits = ilog2(align);
  expr blk_align = blockAlignment();
  return exact ? blk_align == bits : blk_align.uge(bits);
}

expr Pointer::isAligned(uint64_t align) {
  if (align == 1)
    return true;
  if (!is_power2(align))
    return isNull();

  auto offset = getOffset();
  if (isUndef(offset))
    return false;

  auto bits = min(ilog2(align), bits_for_offset);

  expr blk_align = isBlockAligned(align);

  // FIXME: this is not sound; the function may be called only when a global
  // has a better alignment at run time
  if (blk_align.isConst() && offset.isConst() && isLogical().isTrue()) {
    // This is stricter than checking getAddress(), but as addresses are not
    // observed, program shouldn't be able to distinguish this from checking
    // getAddress()
    auto zero = expr::mkUInt(0, bits);
    auto newp = p.extract(totalBits() - 1, bits_for_ptrattrs + bits)
                 .concat(zero);
    if (bits_for_ptrattrs)
      newp = newp.concat(getAttrs());
    p = std::move(newp);
    assert(!p.isValid() || p.bits() == totalBits());
    return { blk_align && offset.trunc(bits) == zero };
  }

  return getAddress().trunc(bits) == 0;
}

expr Pointer::isAligned(const expr &align) {
  uint64_t n;
  if (align.isUInt(n))
    return isAligned(n);

  return
    expr::mkIf(align.isPowerOf2(),
               getAddress().urem(align.zextOrTrunc(bits_ptr_address)) == 0,
               isNull());
}

// When bytes is 0, pointer is always dereferenceable
pair<AndExpr, expr>
Pointer::isDereferenceable(const expr &bytes0, uint64_t align,
                           bool iswrite, bool ignore_accessability,
                           bool round_size_to_align) {
  expr bytes = bytes0.zextOrTrunc(bits_size_t);
  if (round_size_to_align)
    bytes = bytes.round_up(expr::mkUInt(align, bytes));
  expr bytes_off = bytes.zextOrTrunc(bits_for_offset);

  auto block_constraints = [=](const Pointer &p) {
    expr ret = p.isBlockAlive();
    if (iswrite)
      ret &= p.isWritable() && !p.isNoWrite();
    else if (!ignore_accessability)
      ret &= !p.isNoRead();
    return ret;
  };

  auto is_dereferenceable = [&](Pointer &p) {
    expr block_sz = p.blockSizeOffsetT();
    expr offset   = p.getOffset();
    expr cond;

    if (isUndef(offset) || isUndef(p.getBid())) {
      cond = false;
    } else if (m.state->isAsmMode()) {
      // Pointers have full provenance in ASM mode
      auto check = [&](unsigned limit, bool local) {
        for (unsigned i = 0; i != limit; ++i) {
          Pointer this_ptr(m, i, local);
          cond |= p.isInboundsOf(this_ptr, bytes) &&
                  block_constraints(this_ptr);
        }
      };
      cond = false;
      check(m.numLocals(), true);
      check(m.numNonlocals(), false);

    // try some constant folding
    } else if (bytes.ugt(p.blockSize()).isTrue() ||
               p.getOffsetSizet().uge(block_sz).isTrue()) {
      cond = false;
    } else {
      // check that the offset is within bounds and that arith doesn't overflow
      cond  = (offset + bytes_off).sextOrTrunc(block_sz.bits()).ule(block_sz);
      cond &= !offset.isNegative();
      if (!block_sz.isNegative().isFalse()) // implied if block_sz >= 0
        cond &= offset.add_no_soverflow(bytes_off);

      cond &= block_constraints(p);
    }
    return make_pair(std::move(cond), p.isAligned(align));
  };

  DisjointExpr<expr> UB(expr(false)), is_aligned(expr(false)), all_ptrs;

  for (auto &[ptr_expr, domain] : DisjointExpr<expr>(p, 3)) {
    auto [ptr, inboundsd] = Pointer(m, ptr_expr).toLogical(bytes);
    auto [ub, aligned] = is_dereferenceable(ptr);

    // record pointer if not definitely unfeasible
    if (!ub.isFalse() && !aligned.isFalse() && !ptr.blockSize().isZero())
      all_ptrs.add(std::move(ptr).release(), domain);

    UB.add(ub && inboundsd, domain);
    is_aligned.add(std::move(aligned), std::move(domain));
  }

  AndExpr exprs;
  exprs.add(bytes == 0 || *std::move(UB)());

  // cannot store more bytes than address space
  if (bytes0.bits() > bits_size_t)
    exprs.add(bytes0.extract(bytes0.bits() - 1, bits_size_t) == 0);

  // address must be always aligned regardless of access size
  // exprs.add(*std::move(is_aligned)());

  // trim set of valid ptrs
  auto ptrs = std::move(all_ptrs)();
  p = ptrs ? *std::move(ptrs) : expr::mkUInt(0, totalBits());

  return { std::move(exprs), *std::move(is_aligned)() };
}

pair<AndExpr, expr>
Pointer::isDereferenceable(uint64_t bytes, uint64_t align,
                           bool iswrite, bool ignore_accessability,
                           bool round_size_to_align) {
  return isDereferenceable(expr::mkUInt(bytes, bits_size_t), align, iswrite,
                           ignore_accessability, round_size_to_align);
}

// This function assumes that both begin + len don't overflow
void Pointer::isDisjointOrEqual(const expr &len1, const Pointer &ptr2,
                                const expr &len2) const {
  auto off = getOffsetSizet();
  auto off2 = ptr2.getOffsetSizet();
  unsigned bits = off.bits();
  m.state->addUB(getBid() != ptr2.getBid() ||
                 off == off2 ||
                 disjoint(off, len1.zextOrTrunc(bits), off2,
                          len2.zextOrTrunc(bits)));
}

expr Pointer::isBlockAlive() const {
  // NULL block is dead
  if (has_null_block && !null_is_dereferenceable && getBid().isZero())
    return false;

  auto bid = getShortBid();
  return mkIf_fold(isLocal(), m.isBlockAlive(bid, true),
                   m.isBlockAlive(bid, false));
}

expr Pointer::getAllocType() const {
  return getValue("blk_kind", m.local_blk_kind, m.non_local_blk_kind,
                   expr::mkUInt(0, 2));
}

expr Pointer::isStackAllocated(bool simplify) const {
  // 1) if a stack object is returned by a callee it's UB
  // 2) if a stack object is given as argument by the caller, we can upgrade it
  //    to a global object, so we can do POR here.
  if (simplify && (!has_alloca || isLocal().isFalse()))
    return false;
  return getAllocType() == STACK;
}

expr Pointer::isHeapAllocated() const {
  assert(MALLOC == 2 && CXX_NEW == 3);
  return getAllocType().extract(1, 1) == 1;
}

expr Pointer::refined(const Pointer &other) const {
  bool is_asm = other.m.isAsmMode();
  auto [p1l, d1] = toLogical(expr::mkUInt(1, bits_size_t));
  auto [p2l, d2] = other.toLogical(expr::mkUInt(1, bits_size_t));

  // This refers to a block that was malloc'ed within the function
  expr local = other.isLocal();
  local &= getAllocType() == other.getAllocType();
  local &= blockSize() == other.blockSize();
  local &= getOffset() == other.getOffset();
  // Attributes are ignored at refinement.

  // TODO: this induces an infinite loop
  //local &= block_refined(other);

  auto l1 = isLogical();
  auto l2 = other.isLogical();

  expr nonlocal = *this == other;

  return expr::mkIf(isNull(), other.isNull(),
                    (is_asm ? expr(true) : l1 == l2) &&
                      expr::mkIf(l1,
                                 expr::mkIf(isLocal(), local, nonlocal),
                                 getAddress() == other.getAddress())) &&
                      d1.implies(d2 &&
                                 isBlockAlive().implies(other.isBlockAlive()));
}

expr Pointer::fninputRefined(const Pointer &other, set<expr> &undef,
                             const expr &byval_bytes) const {
  bool is_asm = other.m.isAsmMode();
  auto [p1l, d1] = toLogical(expr::mkUInt(1, bits_size_t));
  auto [p2l, d2] = other.toLogical(expr::mkUInt(1, bits_size_t));
  expr size = blockSizeOffsetT();
  expr off = getOffsetSizet();
  expr size2 = other.blockSizeOffsetT();
  expr off2 = other.getOffsetSizet();

  // TODO: check block value for byval_bytes
  if (!byval_bytes.isZero())
    return true;

  expr local
    = expr::mkIf(isHeapAllocated(),
                 getAllocType() == other.getAllocType() &&
                   off == off2 && size == size2,

                 expr::mkIf(off.sge(0),
                            off2.sge(0) &&
                              expr::mkIf(off.ule(size),
                                         off2.ule(size2) && off2.uge(off) &&
                                           (size2 - off2).uge(size - off),
                                         off2.ugt(size2) && off == off2 &&
                                           size2.uge(size)),
                            // maintains same dereferenceability before/after
                            off == off2 && size2.uge(size)));
  local = (other.isLocal() || other.isByval()) && local;

  // TODO: this induces an infinite loop
  // block_refined(other);

  auto l1 = isLogical();
  auto l2 = other.isLogical();

  expr nonlocal = *this == other;

  return expr::mkIf(isNull(), other.isNull(),
                    (is_asm ? expr(true) : l1 == l2) &&
                      expr::mkIf(l1,
                                 expr::mkIf(isLocal(), local, nonlocal),
                                 getAddress() == other.getAddress())) &&
                      d1.implies(d2 &&
                                 isBlockAlive().implies(other.isBlockAlive()));
}

expr Pointer::isWritable() const {
  auto bid = getShortBid();
  expr non_local
    = num_consts_src == 1 ? bid != has_null_block :
        (num_consts_src ? bid.ugt(has_null_block + num_consts_src - 1) : true);
  if (m.numNonlocals() > num_nonlocals_src)
    non_local &= bid.ult(num_nonlocals_src);
  if (has_null_block && null_is_dereferenceable)
    non_local |= bid == 0;

  non_local &= expr::mkUF("#is_writable", {bid}, expr(false));

  // check for non-writable byval blocks (which are non-local)
  for (auto [byval, is_const] : m.byval_blks) {
    if (is_const)
      non_local &= bid != byval;
  }

  return isLocal() || non_local;
}

expr Pointer::isByval() const {
  auto this_bid = getShortBid();
  expr non_local(false);
  for (auto [bid, is_const] : m.byval_blks) {
    non_local |= this_bid == bid;
  }
  return !isLocal() && non_local;
}

expr Pointer::isNocapture(bool simplify) const {
  if (!has_nocapture)
    return false;

  // local pointers can't be no-capture
  if (isLocal(simplify).isTrue())
    return false;

  return p.extract(0, 0) == 1;
}

#define GET_ATTR(attr, idx)          \
  if (!attr)                         \
    return false;                    \
  unsigned idx_ = idx;               \
  return p.extract(idx_, idx_) == 1;

expr Pointer::isNoRead() const {
  GET_ATTR(has_noread, (unsigned)has_nocapture);
}

expr Pointer::isNoWrite() const {
  GET_ATTR(has_nowrite, (unsigned)has_nocapture + (unsigned)has_noread);
}

expr Pointer::isBasedOnArg() const {
  GET_ATTR(has_ptr_arg, (unsigned)has_nocapture + (unsigned)has_noread +
                        (unsigned)has_nowrite);
}

Pointer Pointer::setAttrs(const ParamAttrs &attr) const {
  return { m, getBid(), getOffset(), getAttrs() | attr_to_bitvec(attr) };
}

Pointer Pointer::setIsBasedOnArg() const {
  unsigned idx = (unsigned)has_nocapture + (unsigned)has_noread +
                 (unsigned)has_nowrite;
  auto attrs = getAttrs();
  return { m, getBid(), getOffset(), attrs | expr::mkUInt(1 << idx, attrs) };
}

Pointer Pointer::mkNullPointer(const Memory &m) {
  assert(has_null_block);
  // A null pointer points to block 0 without any attribute.
  return { m, 0, false };
}

expr Pointer::isNull() const {
  if (Memory::observesAddresses())
    return getAddress() == 0;
  if (!has_null_block)
    return false;
  return *this == mkNullPointer(m);
}

bool Pointer::isBlkSingleByte() const {
  uint64_t blk_size;
  return blockSize().isUInt(blk_size) && blk_size == bits_byte/8;
}

pair<Pointer, expr> Pointer::findLogicalPointer(const expr &addr,
                                                const expr &deref_bytes) const {
  DisjointExpr<Pointer> ret;
  expr val = addr.zextOrTrunc(bits_ptr_address);

  auto add = [&](unsigned limit, bool local) {
    for (unsigned i = 0; i != limit; ++i) {
      // address not observed; can't alias with that
      if (local && !m.observed_addrs.mayAlias(true, i))
        continue;

      Pointer p(m, i, local);
      expr size = p.blockSize();

      if (deref_bytes.eq(size)) {
        ret.add(std::move(p), p.getAddress() == val);
        continue;
      }
      if (deref_bytes.ugt(size).isTrue())
        continue;

      ret.add(p + (val - p.getAddress()),
              !local && i == 0 && has_null_block
                ? val == 0
                : val.uge(p.getAddress()) && val.ult((p + size).getAddress()));
    }
  };
  add(m.numLocals(), true);
  add(m.numNonlocals(), false);
  return { *std::move(ret)(), ret.domain() };
}

pair<Pointer, expr> Pointer::toLogical(const expr &deref_bytes) const {
  if (isLogical().isTrue())
    return { *this, true };

  DisjointExpr<Pointer> ret;
  DisjointExpr<expr> leftover;

  // Try to optimize the conversion
  for (auto [e, cond] : DisjointExpr<expr>(p, 5)) {
    Pointer p(m, e);
    if (p.isLogical().isTrue()) {
      ret.add(std::move(p), std::move(cond));
      continue;
    }

    leftover.add(std::move(e), std::move(cond));
  }

  if (!leftover.empty()) {
    auto [ptr, domain]
      = findLogicalPointer(*std::move(leftover)(), deref_bytes);
    ret.add(std::move(ptr), leftover.domain() && domain);
  }

  return { *std::move(ret)(), ret.domain() };
}

Pointer
Pointer::mkIf(const expr &cond, const Pointer &then, const Pointer &els) {
  assert(&then.m == &els.m);
  return Pointer(then.m, expr::mkIf(cond, then.p, els.p));
}

ostream& operator<<(ostream &os, const Pointer &p) {
  auto logical = p.isLogical();

  if (p.isNull().isTrue())
    return os << (logical.isFalse() ? "0x0" : "null");

#define P(field, fn)   \
  if (field.isConst()) \
    field.fn(os);      \
  else                 \
    os << field

  os << (logical.isFalse() ? "phy-ptr(" : "pointer(");

  bool needs_comma = false;
  if (!logical.isFalse()) {
    if (p.isLocal().isConst())
      os << (p.isLocal().isTrue() ? "local" : "non-local");
    else
      os << "local=" << p.isLocal();
    os << ", block_id=";
    P(p.getShortBid(), printUnsigned);

    os << ", offset=";
    P(p.getOffset(), printSigned);
    needs_comma = true;
  }
  if (auto addr = p.getPhysicalAddress();
      addr.isConst() || logical.isFalse()) {
    if (needs_comma)
      os << ", ";
    os << "address=";
    P(addr, printUnsigned);
  }

  if (bits_for_ptrattrs && !p.getAttrs().isZero()) {
    os << ", attrs=";
    P(p.getAttrs(), printUnsigned);
  }
#undef P
  return os << ')';
}

}
