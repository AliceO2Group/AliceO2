// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonDouble.h
/// \brief Storage for double precision members shared with a device that has none

#ifndef GPUCOMMONDOUBLE_H
#define GPUCOMMONDOUBLE_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonDoubleBinary64.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#endif

namespace o2::gpu
{

#ifdef __METAL__

// MSL has no double, and rejects it at the declaration, so a struct holding one
// cannot even be declared. GPUdoubleStore occupies the same eight bytes with the
// same alignment, which keeps the object layout identical to the host's, and the
// stored bits are decoded on read. Storage is therefore untouched: the host still
// writes a double and the same bytes are uploaded.
struct alignas(8) GPUdoubleStore {
  uint32_t mLo;
  uint32_t mHi;
};
typedef float GPUdoubleValue;

// IEEE-754 binary64 -> binary32, round to nearest even, including the subnormal,
// infinity and NaN cases.
GPUdi() GPUdoubleValue GPUdoubleGet(GPUdoubleStore d)
{
  const uint32_t sign = d.mHi & 0x80000000u;
  const int32_t be = int32_t((d.mHi >> 20) & 0x7ffu);
  const uint32_t man = ((d.mHi & 0x000fffffu) << 3) | (d.mLo >> 29);
  const uint32_t drop = d.mLo & 0x1fffffffu;
  uint32_t bits;
  if (be == 0x7ff) {
    bits = sign | 0x7f800000u | (((d.mHi & 0x000fffffu) | d.mLo) ? 0x400000u : 0u);
  } else if (be == 0) {
    bits = sign;
  } else {
    const int32_t e = be - 1023 + 127;
    if (e >= 0xff) {
      bits = sign | 0x7f800000u;
    } else if (e > 0) {
      bits = sign | (uint32_t(e) << 23) | man;
      if ((drop & 0x10000000u) && ((drop & 0x0fffffffu) || (man & 1u))) {
        bits += 1u;
      }
    } else if (e > -24) {
      const uint32_t full = man | 0x800000u;
      const uint32_t sh = uint32_t(1 - e);
      const uint32_t lost = full & ((1u << sh) - 1u);
      const uint32_t halfb = 1u << (sh - 1);
      uint32_t sub = full >> sh;
      if (lost > halfb || (lost == halfb && ((sub & 1u) || drop))) {
        sub += 1u;
      }
      bits = sign | sub;
    } else {
      bits = sign;
    }
  }
  return __builtin_bit_cast(float, bits);
}

#else

typedef double GPUdoubleStore;
typedef double GPUdoubleValue;
GPUhdi() GPUdoubleValue GPUdoubleGet(GPUdoubleStore d) { return d; }

#endif

static_assert(sizeof(GPUdoubleStore) == 8, "GPUdoubleStore must match the size of a double");
static_assert(alignof(GPUdoubleStore) == 8, "GPUdoubleStore must match the alignment of a double");


// Compensated two-float arithmetic, value = mHi + mLo, for the intermediates that
// are deliberately computed in double even when the track itself is float -- the
// Jacobian and covariance terms in TrackParametrizationWithError::propagateTo and
// friends, where differences of nearly equal quantities cancel.
//
// Every operation here depends on the compiler not reassociating the compensation
// terms away, which is why the Metal device code is built without fast math; when
// it is built with fast math, GPUdoubleCalc below is a plain float instead.
//
// mLo is not renormalised after each operation: the value is still mHi + mLo and
// nothing downstream requires |mLo| <= ulp(mHi)/2.
//
// Defined for every backend so it can be tested on the host, but only Metal uses
// it: everywhere else GPUdoubleCalc is a plain double.
class GPUdoubleCalcImpl
{
 public:
  GPUdDefault() GPUdoubleCalcImpl() = default;
  GPUdi() GPUdoubleCalcImpl(float v) : mHi(v), mLo(0.f) {}
  GPUdi() GPUdoubleCalcImpl(float hi, float lo) : mHi(hi), mLo(lo) {}
  // implicit, exactly as double narrows to float: the call sites assign these
  // intermediates straight back into value_t covariance elements
  GPUdi() operator float() const { return mHi + mLo; }

  GPUdi() GPUdoubleCalcImpl operator-() const { return GPUdoubleCalcImpl(-mHi, -mLo); }
  GPUdi() GPUdoubleCalcImpl operator+(GPUdoubleCalcImpl b) const
  {
    const float s = mHi + b.mHi, bb = s - mHi;
    return GPUdoubleCalcImpl(s, ((mHi - (s - bb)) + (b.mHi - bb)) + (mLo + b.mLo));
  }
  GPUdi() GPUdoubleCalcImpl operator-(GPUdoubleCalcImpl b) const { return *this + (-b); }
  GPUdi() GPUdoubleCalcImpl operator*(GPUdoubleCalcImpl b) const
  {
    const float p = mHi * b.mHi;
    return GPUdoubleCalcImpl(p, __builtin_fmaf(mHi, b.mHi, -p) + (mHi * b.mLo + mLo * b.mHi));
  }
  GPUdi() GPUdoubleCalcImpl operator/(GPUdoubleCalcImpl b) const
  {
    const float q = mHi / b.mHi;
    const float r = (__builtin_fmaf(-q, b.mHi, mHi) + mLo) - q * b.mLo;
    return GPUdoubleCalcImpl(q, r / b.mHi);
  }

  // exact matches for the mixed forms, so `a * someFloat` does not sit ambiguously
  // between converting the float up and converting *this down. They also skip the
  // mLo terms that are zero for a float operand, which no reassociation is allowed
  // to fold away here.
  GPUdi() GPUdoubleCalcImpl operator+(float b) const
  {
    const float s = mHi + b, bb = s - mHi;
    return GPUdoubleCalcImpl(s, ((mHi - (s - bb)) + (b - bb)) + mLo);
  }
  GPUdi() GPUdoubleCalcImpl operator-(float b) const { return *this + (-b); }
  GPUdi() GPUdoubleCalcImpl operator*(float b) const
  {
    const float p = mHi * b;
    return GPUdoubleCalcImpl(p, __builtin_fmaf(mHi, b, -p) + mLo * b);
  }
  GPUdi() GPUdoubleCalcImpl operator/(float b) const
  {
    const float q = mHi / b;
    const float r = __builtin_fmaf(-q, b, mHi) + mLo;
    return GPUdoubleCalcImpl(q, r / b);
  }
  // MSL has no double, so on Metal a literal like `1.` is already float and only
  // the float forms are ever selected. The double forms exist so the type can be
  // compiled and tested on the host, where such literals really are double.
#ifndef __METAL__
  GPUdi() GPUdoubleCalcImpl operator+(double b) const { return *this + (float)b; }
  GPUdi() GPUdoubleCalcImpl operator-(double b) const { return *this - (float)b; }
  GPUdi() GPUdoubleCalcImpl operator*(double b) const { return *this * (float)b; }
  GPUdi() GPUdoubleCalcImpl operator/(double b) const { return *this / (float)b; }
#endif

  GPUdi() GPUdoubleCalcImpl& operator+=(GPUdoubleCalcImpl b) { return *this = *this + b; }
  GPUdi() GPUdoubleCalcImpl& operator-=(GPUdoubleCalcImpl b) { return *this = *this - b; }
  GPUdi() GPUdoubleCalcImpl& operator*=(GPUdoubleCalcImpl b) { return *this = *this * b; }
  GPUdi() GPUdoubleCalcImpl& operator/=(GPUdoubleCalcImpl b) { return *this = *this / b; }

 private:
  float mHi, mLo;
};

#ifndef __METAL__
GPUdi() GPUdoubleCalcImpl operator+(double a, GPUdoubleCalcImpl b) { return b + (float)a; }
GPUdi() GPUdoubleCalcImpl operator-(double a, GPUdoubleCalcImpl b) { return (-b) + (float)a; }
GPUdi() GPUdoubleCalcImpl operator*(double a, GPUdoubleCalcImpl b) { return b * (float)a; }
GPUdi() GPUdoubleCalcImpl operator/(double a, GPUdoubleCalcImpl b) { return GPUdoubleCalcImpl((float)a) / b; }
#endif
GPUdi() GPUdoubleCalcImpl operator+(float a, GPUdoubleCalcImpl b) { return b + a; }
GPUdi() GPUdoubleCalcImpl operator-(float a, GPUdoubleCalcImpl b) { return (-b) + a; }
GPUdi() GPUdoubleCalcImpl operator*(float a, GPUdoubleCalcImpl b) { return b * a; }
GPUdi() GPUdoubleCalcImpl operator/(float a, GPUdoubleCalcImpl b) { return GPUdoubleCalcImpl(a) / b; }

// rounds once, as `someFloat += someDouble` does on the host. MSL resolves the
// address spaces separately, and a generic reference would tie with the built-in
// float += float rather than beat it.
#ifdef __METAL__
GPUdi() thread float& operator+=(thread float& a, GPUdoubleCalcImpl b) { return a = (float)(GPUdoubleCalcImpl(a) + b); }
GPUdi() device float& operator+=(device float& a, GPUdoubleCalcImpl b) { return a = (float)(GPUdoubleCalcImpl(a) + b); }
GPUdi() threadgroup float& operator+=(threadgroup float& a, GPUdoubleCalcImpl b) { return a = (float)(GPUdoubleCalcImpl(a) + b); }
#else
GPUdi() float& operator+=(float& a, GPUdoubleCalcImpl b) { return a = (float)(GPUdoubleCalcImpl(a) + b); }
#endif

// CAMath::Abs is a template that deduces its parameter, so a call on one of these
// types picks the primary template, which has no definition. The rest of CAMath
// takes float and is reached through the implicit conversion.
template <>
GPUhdi() GPUdoubleCalcImpl GPUCommonMath::Abs<GPUdoubleCalcImpl>(GPUdoubleCalcImpl x)
{
  return (float)x < 0.f ? -x : x;
}

template <>
GPUhdi() GPUdoubleBinary64 GPUCommonMath::Abs<GPUdoubleBinary64>(GPUdoubleBinary64 x)
{
  return GPUdoubleBinary64::fromBits(x.bits() & ~GPUCA_B64_SIGN);
}

// What GPUdoubleCalc is. Define GPUCA_DOUBLECALC to one of these to override the
// default, which is hardware double everywhere except Metal, and the compensated
// two-float type there. A host build can be set to any of them to compare the
// representations against each other; a Metal build can be set to BINARY64 to
// reproduce the CPU result bit for bit, at two orders of magnitude the cost.
#define GPUCA_DOUBLECALC_DOUBLE 1
#define GPUCA_DOUBLECALC_FLOAT 2
#define GPUCA_DOUBLECALC_TWOFLOAT 3
#define GPUCA_DOUBLECALC_BINARY64 4

#ifndef GPUCA_DOUBLECALC
  #if defined(__METAL__) && defined(__FAST_MATH__)
    // Fast math reassociates the compensation terms away, so the two-float type
    // would cost 1.5x for the accuracy of a plain float.
    #define GPUCA_DOUBLECALC GPUCA_DOUBLECALC_FLOAT
  #elif defined(__METAL__)
    #define GPUCA_DOUBLECALC GPUCA_DOUBLECALC_TWOFLOAT
  #else
    #define GPUCA_DOUBLECALC GPUCA_DOUBLECALC_DOUBLE
  #endif
#endif

#if GPUCA_DOUBLECALC == GPUCA_DOUBLECALC_DOUBLE
  #ifdef __METAL__
    #error "MSL has no double; GPUCA_DOUBLECALC_DOUBLE cannot be selected for Metal"
  #endif
typedef double GPUdoubleCalc;
#elif GPUCA_DOUBLECALC == GPUCA_DOUBLECALC_FLOAT
typedef float GPUdoubleCalc;
#elif GPUCA_DOUBLECALC == GPUCA_DOUBLECALC_TWOFLOAT
typedef GPUdoubleCalcImpl GPUdoubleCalc;
#elif GPUCA_DOUBLECALC == GPUCA_DOUBLECALC_BINARY64
typedef GPUdoubleBinary64 GPUdoubleCalc;
#else
  #error "Invalid setting for GPUCA_DOUBLECALC"
#endif

} // namespace o2::gpu

#endif // GPUCOMMONDOUBLE_H
