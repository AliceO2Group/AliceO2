// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RangeReference.h
/// \brief Class to refered to the 1st entry and N elements of some group in the continuous container
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_RANGEREFERENCE_H
#define ALICEO2_RANGEREFERENCE_H

#include "GPUCommonRtypes.h"

namespace o2
{
namespace dataformats
{
// Composed range reference

template <typename FirstEntry = int, typename NElem = int>
class RangeReference
{
 public:
  RangeReference(FirstEntry ent, NElem n) { set(ent, n); }
  RangeReference(const RangeReference<FirstEntry, NElem>& src) = default;
  RangeReference() = default;
  ~RangeReference() = default;
  void set(FirstEntry ent, NElem n)
  {
    mFirstEntry = ent;
    mEntries = n;
  }
  FirstEntry getFirstEntry() const { return mFirstEntry; }
  NElem getEntries() const { return mEntries; }
  void setFirstEntry(FirstEntry ent) { mFirstEntry = ent; }
  void setEntries(NElem n) { mEntries = n; }
  void changeEntriesBy(NElem inc) { mEntries += inc; }
  bool operator==(const RangeReference& other) const
  {
    return mFirstEntry == other.mFirstEntry && mEntries == other.mEntries;
  }

 private:
  FirstEntry mFirstEntry; ///< 1st entry of the group
  NElem mEntries = 0;     ///< number of entries

  ClassDefNV(RangeReference, 1);
};

// Compact (32bit long) range reference
template <int NBitsN>
class RangeRefComp
{
  using Base = std::uint32_t;

 private:
  static constexpr int NBitsTotal = sizeof(Base) * 8;
  static constexpr Base MaskN = ((0x1 << NBitsN) - 1);
  static constexpr Base MaskR = (~Base(0)) & (~MaskN);
  Base mData = 0; ///< packed 1st entry reference + N entries
  void sanityCheck()
  {
    static_assert(NBitsN < NBitsTotal, "NBitsN too large");
  }

 public:
  RangeRefComp(int ent, int n) { set(ent, n); }
  RangeRefComp() = default;
  RangeRefComp(const RangeRefComp& src) = default;
  void set(int ent, int n)
  {
    mData = (Base(ent) << NBitsN) + (Base(n) & MaskN);
  }
  static constexpr Base getMaxFirstEntry() { return MaskR >> NBitsN; }
  static constexpr Base getMaxEntries() { return MaskN; }
  int getFirstEntry() const { return mData >> NBitsN; }
  int getEntries() const { return mData & ((0x1 << NBitsN) - 1); }
  void setFirstEntry(int ent) { mData = (Base(ent) << NBitsN) | (mData & MaskN); }
  void setEntries(int n) { mData = (mData & MaskR) | (Base(n) & MaskN); }
  void changeEntriesBy(int inc) { setEntries(getEntries() + inc); }
  bool operator==(const RangeRefComp& other) const
  {
    return mData == other.mData;
  }

  ClassDefNV(RangeRefComp, 1);
};

} // namespace dataformats
} // namespace o2

#endif
