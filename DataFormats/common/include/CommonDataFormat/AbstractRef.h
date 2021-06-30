// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  AbstractRef.h
/// \brief Class to refer to object indicating its Indec, Source and status flags
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_ABSTRACT_REF_H
#define ALICEO2_ABSTRACT_REF_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonTypeTraits.h"

namespace o2
{
namespace dataformats
{

template <int NBIdx, int NBSrc, int NBFlg>
class AbstractRef
{
  template <int NBIT>
  static constexpr auto MVAR()
  {
    static_assert(NBIT <= 64, "> 64 bits not supported");
    typename std::conditional<(NBIT > 32), uint64_t, typename std::conditional<(NBIT > 16), uint32_t, typename std::conditional<(NBIT > 8), uint16_t, uint8_t>::type>::type>::type tp = 0;
    return tp;
  }

 public:
  using Base_t = decltype(AbstractRef::MVAR<NBIdx + NBSrc + NBFlg>());
  using Idx_t = decltype(AbstractRef::MVAR<NBIdx>());
  using Src_t = decltype(AbstractRef::MVAR<NBSrc>());
  using Flg_t = decltype(AbstractRef::MVAR<NBFlg>());

  static constexpr Base_t BaseMask = Base_t((((0x1U << (NBIdx + NBSrc + NBFlg - 1)) - 1) << 1) + 1);
  static constexpr Idx_t IdxMask = Idx_t((((0x1U << (NBIdx - 1)) - 1) << 1) + 1);
  static constexpr Src_t SrcMask = Src_t((((0x1U << (NBSrc - 1)) - 1) << 1) + 1);
  static constexpr Flg_t FlgMask = Flg_t((((0x1U << (NBFlg - 1)) - 1) << 1) + 1);
  static constexpr int NBitsIndex() { return NBIdx; }
  static constexpr int NBitsSource() { return NBSrc; }
  static constexpr int NBitsFlags() { return NBFlg; }

  GPUdDefault() AbstractRef() = default;

  GPUdi() AbstractRef(Idx_t idx, Src_t src) { set(idx, src); }
  GPUdi() AbstractRef(Base_t raw) : mRef(raw) {}

  GPUdDefault() AbstractRef(const AbstractRef& src) = default;
  //
  GPUdi() Idx_t getIndex() const { return static_cast<Idx_t>(mRef & IdxMask); }
  GPUdi() void setIndex(Idx_t idx) { mRef = (mRef & (BaseMask & ~IdxMask)) | (IdxMask & idx); }

  //
  GPUdi() Src_t getSource() const { return static_cast<Idx_t>((mRef >> NBIdx) & SrcMask); }
  GPUdi() void setSource(Src_t src) { mRef = (mRef & (BaseMask & ~(SrcMask << NBIdx))) | ((SrcMask & src) << NBIdx); }

  //
  GPUdi() Flg_t getFlags() const { return static_cast<Flg_t>((mRef >> (NBIdx + NBSrc)) & FlgMask); }
  GPUdi() void setFlags(Flg_t f) { mRef = (mRef & (BaseMask & ~(FlgMask << (NBIdx + NBSrc)))) | ((FlgMask & f) << NBIdx); }
  GPUdi() bool testBit(int i) const { return (mRef >> (NBIdx + NBSrc)) & ((0x1U << i) & FlgMask); }
  GPUdi() void setBit(int i) { mRef = (mRef & (BaseMask & ~(0x1U << (i + NBIdx + NBSrc)))) | (((0x1U << i) & FlgMask) << (NBIdx + NBSrc)); }
  GPUdi() void resetBit(int i) { mRef = (mRef & (BaseMask & ~(0x1U << (i + NBIdx + NBSrc)))); }
  GPUdi() void set(Idx_t idx, Src_t src) { mRef = (mRef & ((Base_t)FlgMask << (NBIdx + NBSrc))) | ((SrcMask & Src_t(src)) << NBIdx) | (IdxMask & Idx_t(idx)); }

  GPUdi() Base_t getRaw() const { return mRef; }
  GPUdi() void setRaw(Base_t v) { mRef = v; }
  GPUdi() Base_t getRawWOFlags() const { return mRef & (IdxMask | (SrcMask << NBIdx)); }
  GPUdi() bool isIndexSet() const { return getIndex() != IdxMask; }
  GPUdi() bool isSourceSet() const { return getSource() != SrcMask; }

  GPUdi() bool operator==(const AbstractRef& o) const { return getRawWOFlags() == o.getRawWOFlags(); }
  GPUdi() bool operator!=(const AbstractRef& o) const { return !operator==(o); }

 protected:
  Base_t mRef = IdxMask | (SrcMask << NBIdx); // packed reference, dummy by default

  ClassDefNV(AbstractRef, 1);
};

} // namespace dataformats
} // namespace o2

#endif
