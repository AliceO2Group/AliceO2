// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonDoubleBinary64.h
/// \brief IEEE-754 binary64 in software, for a device that has none

#ifndef GPUCOMMONDOUBLEBINARY64_H
#define GPUCOMMONDOUBLEBINARY64_H

#if !defined(__METAL__) && !defined(GPUCA_B64_HOST_REFERENCE)
#error "the emulated binary64 is for Metal, which has no double of its own; every other backend has a real one. Define GPUCA_B64_HOST_REFERENCE to build it on the host as a test reference."
#endif

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#endif

// IEEE-754 binary64 in software, for a device that has no double at all. Round to
// nearest even only, with subnormals, infinities and NaNs; NaN propagation follows
// the ARM64 order, so an Apple host is a bit-exact reference down to the payload.
// There is no fused multiply-add, no square root and no other rounding mode.
//
// Addition, subtraction, multiplication, division and the conversions to and from
// float and the 32-bit integers are exact, so a Metal build reproduces the CPU
// result bit for bit wherever it uses only those. sin and cos come from fdlibm and
// land within 2 ulp of libm rather than matching it.
//
// It costs of the order of a hundred times plain float on an M-series GPU, which
// the tracking can afford because double is a small fraction of its floating point
// work.

namespace o2::gpu
{

namespace binary64_detail
{

#ifdef __METAL__
typedef ulong u64;
typedef long i64;
typedef uint u32;
#else
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
#endif

#define GPUCA_B64_ALWAYS inline __attribute__((always_inline))
#ifdef __METAL__
// The arithmetic has to stay out of line: inlined into a kernel it drops the
// occupancy (maxTotalThreadsPerThreadgroup 832 -> 384) and runs two to five
// times slower than the call.
#define GPUCA_B64_OP __attribute__((noinline))
#else
#define GPUCA_B64_OP inline
#endif

#ifdef __METAL__
// metal::clz is not constant-evaluable, which would keep the whole class out of
// constant expressions; the builtin folds to the same ctlz at run time.
GPUCA_B64_ALWAYS constexpr int32_t clz64(u64 x) { return x ? __builtin_clzl(x) : 64; }
GPUCA_B64_ALWAYS constexpr int32_t clz32(u32 x) { return x ? __builtin_clz(x) : 32; }
GPUCA_B64_ALWAYS constexpr u32 asu32(float f) { return as_type<u32>(f); }
GPUCA_B64_ALWAYS constexpr float asf32(u32 u) { return as_type<float>(u); }
#else
GPUCA_B64_ALWAYS constexpr int32_t clz64(u64 x) { return x ? __builtin_clzll(x) : 64; }
GPUCA_B64_ALWAYS constexpr int32_t clz32(u32 x) { return x ? __builtin_clz(x) : 32; }

GPUCA_B64_ALWAYS constexpr u32 asu32(float f) { return __builtin_bit_cast(uint32_t, f); }
GPUCA_B64_ALWAYS constexpr float asf32(u32 u) { return __builtin_bit_cast(float, u); }
#endif

// The high half of a 64x64 product, from 32-bit partial products. Neither
// metal::mulhi nor __int128 can appear in a constant expression, and the
// program-scope constants are constexpr.
GPUCA_B64_ALWAYS constexpr u64 mulhiu(u64 a, u64 b)
{
  const u64 al = a & 0xffffffffu, ah = a >> 32, bl = b & 0xffffffffu, bh = b >> 32;
  const u64 ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
  const u64 mid = (ll >> 32) + (lh & 0xffffffffu) + (hl & 0xffffffffu);
  return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}
GPUCA_B64_ALWAYS constexpr i64 mulhis(i64 a, i64 b)
{
  u64 h = mulhiu((u64)a, (u64)b);
  h -= (a < 0) ? (u64)b : 0ULL;
  h -= (b < 0) ? (u64)a : 0ULL;
  return (i64)h;
}

#define GPUCA_B64_SIGN 0x8000000000000000ULL
#define GPUCA_B64_FRAC 0x000fffffffffffffULL
#define GPUCA_B64_IMPL 0x0010000000000000ULL
#define GPUCA_B64_QUIET 0x0008000000000000ULL
#define GPUCA_B64_INF 0x7ff0000000000000ULL
#define GPUCA_B64_DNAN 0x7ff8000000000000ULL // ARM default NaN

// right shift keeping a sticky bit; any count >= 0, no shift count ever reaches 64
GPUCA_B64_ALWAYS constexpr u64 shrJam(u64 m, int32_t s)
{
  const int32_t sc = s > 63 ? 63 : s;
  const u64 r = m >> sc;
  const u64 lost = (m << (63 - sc)) << 1;
  const bool big = s > 63;
  const u64 rr = big ? 0ULL : r;
  const u64 st = big ? m : lost;
  return rr | (st != 0 ? 1ULL : 0ULL);
}

// m: leading bit at 62 for a normal result, 10 guard bits below the 53-bit significand,
// value = m * 2^(e - 1085) with e the biased exponent. Packing (e-1)<<52 + m lets a
// rounding carry ripple into the exponent.
GPUCA_B64_ALWAYS constexpr u64 roundPack(u64 sign, int32_t e, u64 m)
{
  if (e >= 0x7ff) {
    return sign | GPUCA_B64_INF;
  }
  if (e <= 0) {
    m = shrJam(m, 1 - e);
    e = 1;
  }
  const u64 r = m & 0x3ffULL;
  m >>= 10;
  if (r > 0x200ULL || (r == 0x200ULL && (m & 1ULL))) {
    ++m;
  }
  return sign | (((u64)(e - 1) << 52) + m);
}

// an sNaN operand wins over a qNaN one, among equals the first wins, the result is quietened
GPUCA_B64_ALWAYS constexpr u64 propNaN(u64 a, u64 b, bool an, bool bn)
{
  const bool as = an && !(a & GPUCA_B64_QUIET), bs = bn && !(b & GPUCA_B64_QUIET);
  if (as) {
    return a | GPUCA_B64_QUIET;
  }
  if (bs) {
    return b | GPUCA_B64_QUIET;
  }
  return an ? a : b;
}

GPUCA_B64_OP constexpr u64 addsub(u64 a, u64 b0, bool neg)
{
  const u64 b = neg ? (b0 ^ GPUCA_B64_SIGN) : b0;
  const bool sw = (a & ~GPUCA_B64_SIGN) < (b & ~GPUCA_B64_SIGN);
  const u64 x = sw ? b : a, y = sw ? a : b; // |x| >= |y|
  int32_t ex = (int32_t)((x >> 52) & 0x7ff), ey = (int32_t)((y >> 52) & 0x7ff);
  u64 mx = x & GPUCA_B64_FRAC, my = y & GPUCA_B64_FRAC;
  if (ex == 0x7ff) { // y can only be inf/NaN if x is too
    const bool an = ((a >> 52) & 0x7ff) == 0x7ff && (a & GPUCA_B64_FRAC) != 0, bn = ((b0 >> 52) & 0x7ff) == 0x7ff && (b0 & GPUCA_B64_FRAC) != 0;
    if (an || bn) {
      return propNaN(a, b0, an, bn);
    }
    if (ey == 0x7ff) {
      return ((x ^ y) & GPUCA_B64_SIGN) ? GPUCA_B64_DNAN : x;
    }
    return x;
  }
  const u64 sx = x & GPUCA_B64_SIGN;
  const bool sub = ((x ^ y) >> 63) != 0;
  mx = (ex ? (mx | GPUCA_B64_IMPL) : mx) << 10;
  ex = ex ? ex : 1;
  my = (ey ? (my | GPUCA_B64_IMPL) : my) << 10;
  ey = ey ? ey : 1;
  my = shrJam(my, ex - ey);
  const u64 s = sub ? mx - my : mx + my;
  const bool carry = (s >> 63) != 0;
  const int32_t lz = clz64(s) - 1; // -1 on carry, 63 on zero
  const u64 sN = carry ? ((s >> 1) | (s & 1ULL)) : (s << (lz & 63));
  const int32_t eN = carry ? ex + 1 : ex - lz;
  const bool zero = s == 0;
  return roundPack((zero && sub) ? 0ULL : sx, zero ? -1 : eN, sN);
}

GPUCA_B64_OP constexpr u64 mul(u64 a, u64 b)
{
  const u64 sign = (a ^ b) & GPUCA_B64_SIGN;
  int32_t ea = (int32_t)((a >> 52) & 0x7ff), eb = (int32_t)((b >> 52) & 0x7ff);
  u64 ma = a & GPUCA_B64_FRAC, mb = b & GPUCA_B64_FRAC;
  if (ea == 0x7ff || eb == 0x7ff) {
    const bool an = ea == 0x7ff && ma != 0, bn = eb == 0x7ff && mb != 0;
    if (an || bn) {
      return propNaN(a, b, an, bn);
    }
    if ((ea == 0 && ma == 0) || (eb == 0 && mb == 0)) {
      return GPUCA_B64_DNAN; // inf * 0
    }
    return sign | GPUCA_B64_INF;
  }
  if (ea == 0) {
    if (ma == 0) {
      return sign;
    }
    const int32_t lz = clz64(ma) - 11;
    ma <<= lz;
    ea = 1 - lz;
  } else {
    ma |= GPUCA_B64_IMPL;
  }
  if (eb == 0) {
    if (mb == 0) {
      return sign;
    }
    const int32_t lz = clz64(mb) - 11;
    mb <<= lz;
    eb = 1 - lz;
  } else {
    mb |= GPUCA_B64_IMPL;
  }
  const u64 lo = ma * mb, hi = mulhiu(ma, mb); // 106-bit product in [2^104, 2^106)
  const bool top = (hi & (1ULL << 41)) != 0;
  const u64 m1 = (hi << 21) | (lo >> 43), m0 = (hi << 22) | (lo >> 42);
  const u64 st = top ? (lo & ((1ULL << 43) - 1)) : (lo & ((1ULL << 42) - 1));
  const u64 m = (top ? m1 : m0) | (st != 0 ? 1ULL : 0ULL);
  return roundPack(sign, ea + eb - 1023 + (top ? 1 : 0), m);
}

GPUCA_B64_OP constexpr u64 div(u64 a, u64 b)
{
  const u64 sign = (a ^ b) & GPUCA_B64_SIGN;
  int32_t ea = (int32_t)((a >> 52) & 0x7ff), eb = (int32_t)((b >> 52) & 0x7ff);
  u64 ma = a & GPUCA_B64_FRAC, mb = b & GPUCA_B64_FRAC;
  if (ea == 0x7ff || eb == 0x7ff) {
    const bool an = ea == 0x7ff && ma != 0, bn = eb == 0x7ff && mb != 0;
    if (an || bn) {
      return propNaN(a, b, an, bn);
    }
    if (ea == 0x7ff && eb == 0x7ff) {
      return GPUCA_B64_DNAN; // inf / inf
    }
    return ea == 0x7ff ? (sign | GPUCA_B64_INF) : sign; // inf / x, x / inf
  }
  if (eb == 0 && mb == 0) {
    return (ea == 0 && ma == 0) ? GPUCA_B64_DNAN : (sign | GPUCA_B64_INF); // x / 0
  }
  if (ea == 0) {
    if (ma == 0) {
      return sign;
    }
    const int32_t lz = clz64(ma) - 11;
    ma <<= lz;
    ea = 1 - lz;
  } else {
    ma |= GPUCA_B64_IMPL;
  }
  if (eb == 0) {
    const int32_t lz = clz64(mb) - 11;
    mb <<= lz;
    eb = 1 - lz;
  } else {
    mb |= GPUCA_B64_IMPL;
  }
  const bool lt = ma < mb;
  const int32_t e = ea - eb + 1023 - (lt ? 1 : 0);
  const u64 A2 = lt ? (ma << 1) : ma; // A2 / mb in [1, 2)
  // reciprocal R ~ 2^114 / mb in (2^61, 2^62], seeded from a float division on the top 24 bits
  const u32 rb = asu32(1.0f / (float)(u32)(mb >> 29));
  u64 R = (u64)((rb & 0x7fffffu) | 0x800000u) << ((int32_t)((rb >> 23) & 0xff) - 127 + 62);
  const u64 Bn = mb << 11;
  for (int32_t it = 0; it < 2; ++it) {
    const i64 E = (i64)(1ULL << 61) - (i64)mulhiu(Bn, R);
    R = (u64)((i64)R + mulhis((i64)R, E * 8)); // not E << 3: shifting a negative value is not a constant expression
  }
  R = R > (1ULL << 62) ? (1ULL << 62) : R;
  // Q ~ A2 * 2^62 / mb, then the exact remainder, which fits in a signed 64-bit
  // word because Q is within a few units
  u64 Q = mulhiu(A2 << 10, (R << 2) - 1);
  i64 rem = (i64)((A2 << 62) - Q * mb);
  const i64 adj = mulhis(rem, (i64)R) >> 50;
  Q = (u64)((i64)Q + adj);
  rem -= adj * (i64)mb;
  // after adj the remainder is within one divisor of [0, mb): one predicated step each way
  const bool ng = rem < 0;
  Q = ng ? Q - 1 : Q;
  rem = ng ? rem + (i64)mb : rem;
  const bool bg = rem >= (i64)mb;
  Q = bg ? Q + 1 : Q;
  rem = bg ? rem - (i64)mb : rem;
  return roundPack(sign, e, Q | (rem != 0 ? 1ULL : 0ULL));
}

// an int32 or a uint32 always fits the 53-bit significand, so these are exact
GPUCA_B64_ALWAYS constexpr u64 fromU32(u32 x)
{
  if (x == 0) {
    return 0ULL;
  }
  const int32_t lz = clz32(x);
  return ((u64)(31 - lz + 1023) << 52) | (((u64)x << (21 + lz)) & GPUCA_B64_FRAC);
}

GPUCA_B64_ALWAYS constexpr u64 fromI32(int32_t x)
{
  return fromU32(x < 0 ? (u32)(-(i64)x) : (u32)x) | (x < 0 ? GPUCA_B64_SIGN : 0ULL);
}

GPUCA_B64_ALWAYS constexpr u64 fromFloat(float f)
{
  const u32 u = asu32(f);
  const u64 sign = (u64)(u & 0x80000000u) << 32;
  int32_t e = (int32_t)((u >> 23) & 0xff);
  u32 m = u & 0x7fffffu;
  if (e == 0xff) {
    return sign | GPUCA_B64_INF | ((u64)m << 29) | (m ? GPUCA_B64_QUIET : 0ULL);
  }
  if (e == 0) {
    if (m == 0) {
      return sign;
    }
    const int32_t lz = clz32(m) - 8;
    m <<= lz;
    e = 1 - lz;
  }
  return sign | ((u64)(e - 127 + 1023) << 52) | ((u64)(m & 0x7fffffu) << 29);
}

// binary64 -> binary32, round to nearest even, subnormals, inf, NaN (payload kept, quietened)
GPUCA_B64_OP constexpr float toFloat(u64 d)
{
  const u32 sign = (u32)(d >> 32) & 0x80000000u;
  const int32_t be = (int32_t)((d >> 52) & 0x7ff);
  const u32 man = (u32)((d & GPUCA_B64_FRAC) >> 29);
  const u32 drop = (u32)d & 0x1fffffffu;
  u32 bits = 0;
  if (be == 0x7ff) {
    bits = sign | 0x7f800000u | man | ((d & GPUCA_B64_FRAC) ? 0x400000u : 0u);
  } else if (be == 0) {
    bits = sign;
  } else {
    const int32_t e = be - 1023 + 127;
    if (e >= 0xff) {
      bits = sign | 0x7f800000u;
    } else if (e > 0) {
      bits = sign | ((u32)e << 23) | man;
      if ((drop & 0x10000000u) && ((drop & 0x0fffffffu) || (man & 1u))) {
        bits += 1u;
      }
    } else if (e > -24) {
      const u32 full = man | 0x800000u;
      const u32 sh = (u32)(1 - e);
      const u32 lost = full & ((1u << sh) - 1u);
      const u32 halfb = 1u << (sh - 1);
      u32 sub = full >> sh;
      if (lost > halfb || (lost == halfb && ((sub & 1u) || drop))) {
        sub += 1u;
      }
      bits = sign | sub;
    } else {
      bits = sign;
    }
  }
  return asf32(bits);
}

} // namespace binary64_detail

class GPUdoubleBinary64
{
 public:
  GPUdDefault() GPUdoubleBinary64() = default;
  GPUdi() constexpr GPUdoubleBinary64(float v) : mBits(binary64_detail::fromFloat(v)) {}
  GPUdi() constexpr operator float() const { return binary64_detail::toFloat(mBits); }

  GPUdi() static constexpr GPUdoubleBinary64 fromBits(binary64_detail::u64 b) { return GPUdoubleBinary64(b, FromBits{}); }
  GPUdi() constexpr binary64_detail::u64 bits() const { return mBits; }

  GPUdi() constexpr GPUdoubleBinary64 operator-() const { return fromBits(mBits ^ GPUCA_B64_SIGN); }
  GPUdi() constexpr GPUdoubleBinary64 operator+(GPUdoubleBinary64 b) const { return fromBits(binary64_detail::addsub(mBits, b.mBits, false)); }
  GPUdi() constexpr GPUdoubleBinary64 operator-(GPUdoubleBinary64 b) const { return fromBits(binary64_detail::addsub(mBits, b.mBits, true)); }
  GPUdi() constexpr GPUdoubleBinary64 operator*(GPUdoubleBinary64 b) const { return fromBits(binary64_detail::mul(mBits, b.mBits)); }
  GPUdi() constexpr GPUdoubleBinary64 operator/(GPUdoubleBinary64 b) const { return fromBits(binary64_detail::div(mBits, b.mBits)); }

  GPUdi() constexpr GPUdoubleBinary64 operator+(float b) const { return *this + GPUdoubleBinary64(b); }
  GPUdi() constexpr GPUdoubleBinary64 operator-(float b) const { return *this - GPUdoubleBinary64(b); }
  GPUdi() constexpr GPUdoubleBinary64 operator*(float b) const { return *this * GPUdoubleBinary64(b); }
  GPUdi() constexpr GPUdoubleBinary64 operator/(float b) const { return *this / GPUdoubleBinary64(b); }
#ifndef __METAL__
  GPUdi() constexpr GPUdoubleBinary64 operator+(double b) const { return *this + (float)b; }
  GPUdi() constexpr GPUdoubleBinary64 operator-(double b) const { return *this - (float)b; }
  GPUdi() constexpr GPUdoubleBinary64 operator*(double b) const { return *this * (float)b; }
  GPUdi() constexpr GPUdoubleBinary64 operator/(double b) const { return *this / (float)b; }
#endif

  // integral operands: an exact match, so `2 * x` does not sit ambiguously between
  // converting the int up and converting *this down
  GPUdi() constexpr GPUdoubleBinary64 operator+(int32_t b) const { return *this + fromBits(binary64_detail::fromI32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator-(int32_t b) const { return *this - fromBits(binary64_detail::fromI32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator*(int32_t b) const { return *this * fromBits(binary64_detail::fromI32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator/(int32_t b) const { return *this / fromBits(binary64_detail::fromI32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator+(uint32_t b) const { return *this + fromBits(binary64_detail::fromU32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator-(uint32_t b) const { return *this - fromBits(binary64_detail::fromU32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator*(uint32_t b) const { return *this * fromBits(binary64_detail::fromU32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator/(uint32_t b) const { return *this / fromBits(binary64_detail::fromU32(b)); }
#ifdef __METAL__
  // the same surface for an object that lives in the constant address space, which
  // a generic `this` does not reach
  GPUdi() constexpr GPUdoubleBinary64(float v) constant : mBits(binary64_detail::fromFloat(v)) {}
  GPUdi() constexpr operator float() constant { return binary64_detail::toFloat(mBits); }
  GPUdi() constexpr GPUdoubleBinary64 operator+(GPUdoubleBinary64 b) constant { return fromBits(binary64_detail::addsub(mBits, b.mBits, false)); }
  GPUdi() constexpr GPUdoubleBinary64 operator-(GPUdoubleBinary64 b) constant { return fromBits(binary64_detail::addsub(mBits, b.mBits, true)); }
  GPUdi() constexpr GPUdoubleBinary64 operator*(GPUdoubleBinary64 b) constant { return fromBits(binary64_detail::mul(mBits, b.mBits)); }
  GPUdi() constexpr GPUdoubleBinary64 operator/(GPUdoubleBinary64 b) constant { return fromBits(binary64_detail::div(mBits, b.mBits)); }
  GPUdi() constexpr GPUdoubleBinary64 operator+(float b) constant { return *this + GPUdoubleBinary64(b); }
  GPUdi() constexpr GPUdoubleBinary64 operator-(float b) constant { return *this - GPUdoubleBinary64(b); }
  GPUdi() constexpr GPUdoubleBinary64 operator*(float b) constant { return *this * GPUdoubleBinary64(b); }
  GPUdi() constexpr GPUdoubleBinary64 operator/(float b) constant { return *this / GPUdoubleBinary64(b); }
  GPUdi() constexpr GPUdoubleBinary64 operator*(int32_t b) constant { return *this * fromBits(binary64_detail::fromI32(b)); }
  GPUdi() constexpr GPUdoubleBinary64 operator/(int32_t b) constant { return *this / fromBits(binary64_detail::fromI32(b)); }
#endif

  GPUdi() GPUdoubleBinary64& operator+=(GPUdoubleBinary64 b) { return *this = *this + b; }
  GPUdi() GPUdoubleBinary64& operator-=(GPUdoubleBinary64 b) { return *this = *this - b; }
  GPUdi() GPUdoubleBinary64& operator*=(GPUdoubleBinary64 b) { return *this = *this * b; }
  GPUdi() GPUdoubleBinary64& operator/=(GPUdoubleBinary64 b) { return *this = *this / b; }

 private:
  struct FromBits {
  };
  GPUdi() constexpr GPUdoubleBinary64(binary64_detail::u64 b, FromBits) : mBits(b) {}

  binary64_detail::u64 mBits;
};

GPUdi() constexpr GPUdoubleBinary64 operator+(int32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromI32(a)) + b; }
GPUdi() constexpr GPUdoubleBinary64 operator-(int32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromI32(a)) - b; }
GPUdi() constexpr GPUdoubleBinary64 operator*(int32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromI32(a)) * b; }
GPUdi() constexpr GPUdoubleBinary64 operator/(int32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromI32(a)) / b; }
GPUdi() constexpr GPUdoubleBinary64 operator+(uint32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromU32(a)) + b; }
GPUdi() constexpr GPUdoubleBinary64 operator-(uint32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromU32(a)) - b; }
GPUdi() constexpr GPUdoubleBinary64 operator*(uint32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromU32(a)) * b; }
GPUdi() constexpr GPUdoubleBinary64 operator/(uint32_t a, GPUdoubleBinary64 b) { return GPUdoubleBinary64::fromBits(binary64_detail::fromU32(a)) / b; }
GPUdi() constexpr GPUdoubleBinary64 operator+(float a, GPUdoubleBinary64 b) { return GPUdoubleBinary64(a) + b; }
GPUdi() constexpr GPUdoubleBinary64 operator-(float a, GPUdoubleBinary64 b) { return GPUdoubleBinary64(a) - b; }
GPUdi() constexpr GPUdoubleBinary64 operator*(float a, GPUdoubleBinary64 b) { return GPUdoubleBinary64(a) * b; }
GPUdi() constexpr GPUdoubleBinary64 operator/(float a, GPUdoubleBinary64 b) { return GPUdoubleBinary64(a) / b; }
#ifndef __METAL__
GPUdi() constexpr GPUdoubleBinary64 operator+(double a, GPUdoubleBinary64 b) { return GPUdoubleBinary64((float)a) + b; }
GPUdi() constexpr GPUdoubleBinary64 operator-(double a, GPUdoubleBinary64 b) { return GPUdoubleBinary64((float)a) - b; }
GPUdi() constexpr GPUdoubleBinary64 operator*(double a, GPUdoubleBinary64 b) { return GPUdoubleBinary64((float)a) * b; }
GPUdi() constexpr GPUdoubleBinary64 operator/(double a, GPUdoubleBinary64 b) { return GPUdoubleBinary64((float)a) / b; }
#endif

// rounds once, as `someFloat += someDouble` does on the host
#ifdef __METAL__
GPUdi() thread float& operator+=(thread float& a, GPUdoubleBinary64 b) { return a = (float)(GPUdoubleBinary64(a) + b); }
GPUdi() device float& operator+=(device float& a, GPUdoubleBinary64 b) { return a = (float)(GPUdoubleBinary64(a) + b); }
GPUdi() threadgroup float& operator+=(threadgroup float& a, GPUdoubleBinary64 b) { return a = (float)(GPUdoubleBinary64(a) + b); }
#else
GPUdi() float& operator+=(float& a, GPUdoubleBinary64 b) { return a = (float)(GPUdoubleBinary64(a) + b); }
#endif

namespace binary64_detail
{
// sin and cos are fdlibm's __kernel_sin / __kernel_cos and the medium-range
// branch of __ieee754_rem_pio2. The coefficients are spelled as bit patterns
// because MSL has no double literals. The argument reduction is exact for
// |x| <= 2^19 * pi/2; beyond that the accuracy degrades gracefully.
GPUCA_B64_OP GPUdoubleBinary64 kernelSin(GPUdoubleBinary64 x, GPUdoubleBinary64 y, bool iy)
{
  typedef GPUdoubleBinary64 b64;
  const b64 S1 = b64::fromBits(0xBFC5555555555549ULL);
  const b64 S2 = b64::fromBits(0x3F8111111110F8A6ULL);
  const b64 S3 = b64::fromBits(0xBF2A01A019C161D5ULL);
  const b64 S4 = b64::fromBits(0x3EC71DE357B1FE7DULL);
  const b64 S5 = b64::fromBits(0xBE5AE5E68A2B9CEBULL);
  const b64 S6 = b64::fromBits(0x3DE5D93A5ACFD57CULL);
  const b64 z = x * x;
  const b64 v = z * x;
  const b64 r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
  if (!iy) {
    return x + v * (S1 + z * r);
  }
  return x - ((z * (b64::fromBits(0x3FE0000000000000ULL) * y - v * r) - y) - v * S1);
}

GPUCA_B64_OP GPUdoubleBinary64 kernelCos(GPUdoubleBinary64 x, GPUdoubleBinary64 y)
{
  typedef GPUdoubleBinary64 b64;
  const b64 C1 = b64::fromBits(0x3FA555555555554CULL);
  const b64 C2 = b64::fromBits(0xBF56C16C16C15177ULL);
  const b64 C3 = b64::fromBits(0x3EFA01A019CB1590ULL);
  const b64 C4 = b64::fromBits(0xBE927E4F809C52ADULL);
  const b64 C5 = b64::fromBits(0x3E21EE9EBDB4B1C4ULL);
  const b64 C6 = b64::fromBits(0xBDA8FAE9BE8838D4ULL);
  const b64 one = b64::fromBits(0x3FF0000000000000ULL);
  const b64 oneHalf = b64::fromBits(0x3FE0000000000000ULL);
  const u32 ix = (u32)(x.bits() >> 32) & 0x7fffffffu;
  const b64 z = x * x;
  const b64 r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
  if (ix < 0x3FD33333u) { // |x| < 0.3
    return one - (oneHalf * z - (z * r - x * y));
  }
  const b64 qx = (ix > 0x3FE90000u) ? b64::fromBits(0x3FD2000000000000ULL) : b64::fromBits((u64)(ix - 0x00200000u) << 32); // 0.28125, else |x| / 4
  return (one - qx) - ((oneHalf * z - qx) - (z * r - x * y));
}

GPUCA_B64_ALWAYS int32_t truncToInt32(GPUdoubleBinary64 x)
{
  const u64 b = x.bits();
  const int32_t e = (int32_t)((b >> 52) & 0x7ff) - 1023;
  if (e < 0) {
    return 0;
  }
  const u64 m = (b & GPUCA_B64_FRAC) | GPUCA_B64_IMPL;
  const int32_t v = (int32_t)(e >= 52 ? (m << (e - 52)) : (m >> (52 - e)));
  return (b & GPUCA_B64_SIGN) ? -v : v;
}

struct SinCosPair {
  GPUdoubleBinary64 s, c;
};

GPUCA_B64_OP SinCosPair sincos(GPUdoubleBinary64 x)
{
  typedef GPUdoubleBinary64 b64;
  const u32 ix = (u32)(x.bits() >> 32) & 0x7fffffffu;
  SinCosPair out;
  if (ix >= 0x7FF00000u) { // inf or NaN
    out.s = out.c = b64::fromBits(GPUCA_B64_DNAN);
    return out;
  }

  if (ix < 0x3E400000u) { // |x| < 2^-27, where the sign of a zero x has to survive
    out.s = x;
    out.c = b64::fromBits(0x3FF0000000000000ULL);
    return out;
  }

  b64 y0 = x, y1 = b64::fromBits(0ULL);
  int32_t n = 0;
  const bool reduced = ix > 0x3FE921FBu; // |x| > pi/4
  if (reduced) {
    const b64 t = b64::fromBits(x.bits() & ~GPUCA_B64_SIGN);
    n = truncToInt32(t * b64::fromBits(0x3FE45F306DC9C883ULL) + b64::fromBits(0x3FE0000000000000ULL));
    const b64 fn = b64((float)n);
    // Cody-Waite with pi/2 split over three terms, good to 151 bits
    b64 r = t - fn * b64::fromBits(0x3FF921FB54400000ULL);
    b64 s = r;
    b64 w = fn * b64::fromBits(0x3DD0B4611A600000ULL);
    r = s - w;
    w = fn * b64::fromBits(0x3BA3198A2E037073ULL) - ((s - r) - w);
    s = r;
    w = fn * b64::fromBits(0x3BA3198A2E000000ULL);
    r = s - w;
    w = fn * b64::fromBits(0x397B839A252049C1ULL) - ((s - r) - w);
    y0 = r - w;
    y1 = (r - y0) - w;
    if (x.bits() & GPUCA_B64_SIGN) {
      y0 = -y0;
      y1 = -y1;
      n = -n;
    }
  }

  switch (n & 3) {
    case 0:
      out.s = kernelSin(y0, y1, reduced);
      out.c = kernelCos(y0, y1);
      break;
    case 1:
      out.s = kernelCos(y0, y1);
      out.c = -kernelSin(y0, y1, reduced);
      break;
    case 2:
      out.s = -kernelSin(y0, y1, reduced);
      out.c = -kernelCos(y0, y1);
      break;
    default:
      out.s = -kernelCos(y0, y1);
      out.c = kernelSin(y0, y1, reduced);
      break;
  }
  return out;
}
} // namespace binary64_detail

} // namespace o2::gpu

#endif // GPUCOMMONDOUBLEBINARY64_H
