// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  AbstractRef.h
/// \brief Class to refer to object indicating its Indec, Source and status flags
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_ABSTRACT_REF_H
#define ALICEO2_ABSTRACT_REF_H

#include <Rtypes.h>

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

  AbstractRef() = default;

  AbstractRef(Idx_t idx) { setIndex(idx); }

  AbstractRef(Idx_t idx, Src_t src) { set(idx, src); }

  //
  Idx_t getIndex() const { return static_cast<Idx_t>(mRef & IdxMask); }
  void setIndex(Idx_t idx) { mRef = (mRef & (BaseMask & ~IdxMask)) | (IdxMask & idx); }

  //
  Src_t getSource() const { return static_cast<Idx_t>((mRef >> NBIdx) & SrcMask); }
  void setSource(Src_t src) { mRef = (mRef & (BaseMask & ~(SrcMask << NBIdx))) | ((SrcMask & src) << NBIdx); }

  //
  Flg_t getFlags() const { return static_cast<Flg_t>((mRef >> (NBIdx + NBSrc)) & FlgMask); }
  void setFlags(Flg_t f) { mRef = (mRef & (BaseMask & ~(FlgMask << (NBIdx + NBSrc)))) | ((FlgMask & f) << NBIdx); }
  bool testBit(int i) const { return (mRef >> (NBIdx + NBSrc)) & ((0x1U << i) & FlgMask); }
  void setBit(int i) { mRef = (mRef & (BaseMask & ~(0x1U << (i + NBIdx + NBSrc)))) | (((0x1U << i) & FlgMask) << (NBIdx + NBSrc)); }
  void resetBit(int i) { mRef = (mRef & (BaseMask & ~(0x1U << (i + NBIdx + NBSrc)))); }
  void set(Idx_t idx, Src_t src) { mRef = (mRef & (BaseMask & ~((SrcMask << NBIdx) | (BaseMask & ~IdxMask)))) | ((SrcMask & Src_t(src)) << NBIdx) | (IdxMask & Idx_t(idx)); }

  Base_t getRaw() const { return mRef; }

 protected:
  Base_t mRef = 0; // packed reference

  ClassDefNV(AbstractRef, 1);
};

} // namespace dataformats
} // namespace o2

#endif
