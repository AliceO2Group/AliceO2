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

/// \file   CTFHelper.h
/// \author ruben.shahoyan@cern.ch
/// \brief  Helper for MID CTF creation

#ifndef O2_MID_CTF_HELPER_H
#define O2_MID_CTF_HELPER_H

#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/CTF.h"
#include "CommonDataFormat/AbstractRef.h"
#include <gsl/span>

namespace o2
{
namespace utils
{
class IRFrameSelector;
}
namespace mid
{

class CTFHelper
{

 public:
  using OrderRef = o2::dataformats::AbstractRef<29, 2, 1>; // 29 bits for index in event type span, 2 bits for event type, 1 bit flag
  struct TFData {
    std::vector<OrderRef> colDataRefs{};
    std::vector<OrderRef> rofDataRefs{};
    std::array<gsl::span<const o2::mid::ColumnData>, NEvTypes> colData{};
    std::array<gsl::span<const o2::mid::ROFRecord>, NEvTypes> rofData{};
    void buildReferences(o2::utils::IRFrameSelector& irSelector);
  };

  CTFHelper(const TFData& data) : mTFData(data) {}
  CTFHelper() = delete;

  CTFHeader createHeader()
  {
    CTFHeader h{o2::detectors::DetID::MID, 0, 1, 0, // dummy timestamp, version 1.0
                uint32_t(mTFData.rofDataRefs.size()), uint32_t(mTFData.colDataRefs.size()), 0, 0};
    if (h.nROFs) {
      auto id0 = mTFData.rofDataRefs.front();
      const auto& rof = mTFData.rofData[id0.getSource()][id0.getIndex()];
      h.firstOrbit = rof.interactionRecord.orbit;
      h.firstBC = rof.interactionRecord.bc;
    }
    return h;
  }

  size_t getSize() const { return mTFData.rofDataRefs.size() * sizeof(o2::mid::ROFRecord) + mTFData.colDataRefs.size() * sizeof(o2::mid::ColumnData); }

  //>>> =========================== ITERATORS ========================================

  template <typename I, typename D, typename T, int M = 1>
  class _Iter
  {
   public:
    using difference_type = int64_t;
    using value_type = T;
    using pointer = const T*;
    using reference = const T&;
    using iterator_category = std::random_access_iterator_tag;

    _Iter(const std::vector<OrderRef>& ord, const std::array<gsl::span<const D>, NEvTypes>& data, bool end = false) : mOrder(ord), mData(&data), mIndex(end ? M * ord.size() : 0) {}
    _Iter() = default;

    const I& operator++()
    {
      ++mIndex;
      return (I&)(*this);
    }

    const I& operator--()
    {
      mIndex--;
      return (I&)(*this);
    }
    const I operator++(int)
    {
      auto res = *this;
      ++mIndex;
      return res;
    }

    const I operator--(int)
    {
      auto res = *this;
      --mIndex;
      return res;
    }

    const I& operator+=(difference_type i)
    {
      mIndex += i;
      return (I&)(*this);
    }

    const I operator+=(difference_type i) const
    {
      auto tmp = *const_cast<I*>(this);
      return tmp += i;
    }

    const I& operator-=(difference_type i)
    {
      mIndex -= i;
      return (I&)(*this);
    }

    const I operator-=(difference_type i) const
    {
      auto tmp = *const_cast<I*>(this);
      return tmp -= i;
    }

    difference_type operator-(const I& other) const { return mIndex - other.mIndex; }

    difference_type operator-(size_t idx) const { return mIndex - idx; }

    const I& operator-(size_t idx)
    {
      mIndex -= idx;
      return (I&)(*this);
    }

    bool operator!=(const I& other) const { return mIndex != other.mIndex; }
    bool operator==(const I& other) const { return mIndex == other.mIndex; }
    bool operator>(const I& other) const { return mIndex > other.mIndex; }
    bool operator<(const I& other) const { return mIndex < other.mIndex; }
    bool operator>=(const I& other) const { return mIndex >= other.mIndex; }
    bool operator<=(const I& other) const { return mIndex <= other.mIndex; }

   protected:
    gsl::span<const OrderRef> mOrder{};
    const std::array<gsl::span<const D>, NEvTypes>* mData{};
    size_t mIndex = 0;
  };

  //_______________________________________________
  // BC difference wrt previous if in the same orbit, otherwise the abs.value.
  // For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_bcIncROF : public _Iter<Iter_bcIncROF, ROFRecord, uint16_t>
  {
   public:
    using _Iter<Iter_bcIncROF, ROFRecord, uint16_t>::_Iter;
    value_type operator*() const
    {
      const auto ir = (*mData)[mOrder[mIndex].getSource()][mOrder[mIndex].getIndex()].interactionRecord;
      if (mIndex) {
        const auto irP = (*mData)[mOrder[mIndex - 1].getSource()][mOrder[mIndex - 1].getIndex()].interactionRecord;
        if (ir.orbit == irP.orbit) {
          return ir.bc - irP.bc;
        } else {
          return ir.bc;
        }
      }
      return 0;
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      const auto ir = (*mData)[mOrder[id].getSource()][mOrder[id].getIndex()].interactionRecord;
      if (id) {
        const auto irP = (*mData)[mOrder[id - 1].getSource()][mOrder[id - 1].getIndex()].interactionRecord;
        if (ir.orbit == irP.orbit) {
          return ir.bc - irP.bc;
        } else {
          return ir.bc;
        }
      }
      return 0;
    }
  };

  //_______________________________________________
  // Orbit difference wrt previous. For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_orbitIncROF : public _Iter<Iter_orbitIncROF, ROFRecord, uint32_t>
  {
   public:
    using _Iter<Iter_orbitIncROF, ROFRecord, uint32_t>::_Iter;
    value_type operator*() const
    {
      if (mIndex) {
        const auto ir = (*mData)[mOrder[mIndex].getSource()][mOrder[mIndex].getIndex()].interactionRecord;
        const auto irP = (*mData)[mOrder[mIndex - 1].getSource()][mOrder[mIndex - 1].getIndex()].interactionRecord;
        return ir.orbit - irP.orbit;
      }
      return 0;
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      if (id) {
        const auto ir = (*mData)[mOrder[id].getSource()][mOrder[id].getIndex()].interactionRecord;
        const auto irP = (*mData)[mOrder[id - 1].getSource()][mOrder[id - 1].getIndex()].interactionRecord;
        return ir.orbit - irP.orbit;
      }
      return 0;
    }
  };

  //_______________________________________________
  // Number of entries in the ROF
  class Iter_entriesROF : public _Iter<Iter_entriesROF, ROFRecord, uint16_t>
  {
   public:
    using _Iter<Iter_entriesROF, ROFRecord, uint16_t>::_Iter;
    value_type operator*() const { return (*mData)[mOrder[mIndex].getSource()][mOrder[mIndex].getIndex()].nEntries; }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return (*mData)[mOrder[id].getSource()][mOrder[id].getIndex()].nEntries;
    }
  };

  //_______________________________________________
  // Event type for the ROF
  class Iter_evtypeROF : public _Iter<Iter_evtypeROF, ROFRecord, uint8_t>
  {
   public:
    using _Iter<Iter_evtypeROF, ROFRecord, uint8_t>::_Iter;
    value_type operator*() const { return value_type((*mData)[mOrder[mIndex].getSource()][mOrder[mIndex].getIndex()].eventType); }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return value_type((*mData)[mOrder[id].getSource()][mOrder[id].getIndex()].eventType);
    }
  };

  //_______________________________________________
  class Iter_pattern : public _Iter<Iter_pattern, ColumnData, uint16_t, 5>
  {
   public:
    using _Iter<Iter_pattern, ColumnData, uint16_t, 5>::_Iter;
    value_type operator*() const
    {
      auto idx = mOrder[mIndex / 5];
      return (*mData)[idx.getSource()][idx.getIndex()].patterns[mIndex % 5];
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      auto idx = mOrder[id / 5];
      return (*mData)[idx.getSource()][idx.getIndex()].patterns[id % 5];
    }
  };

  //_______________________________________________
  class Iter_deId : public _Iter<Iter_deId, ColumnData, uint8_t>
  {
   public:
    using _Iter<Iter_deId, ColumnData, uint8_t>::_Iter;
    value_type operator*() const
    {
      auto idx = mOrder[mIndex];
      return (*mData)[idx.getSource()][idx.getIndex()].deId;
    }
    value_type operator[](difference_type i) const
    {
      auto idx = mOrder[mIndex + i];
      return (*mData)[idx.getSource()][idx.getIndex()].deId;
    }
  };

  //_______________________________________________
  class Iter_colId : public _Iter<Iter_colId, ColumnData, uint8_t>
  {
   public:
    using _Iter<Iter_colId, ColumnData, uint8_t>::_Iter;
    value_type operator*() const
    {
      auto idx = mOrder[mIndex];
      return (*mData)[idx.getSource()][idx.getIndex()].columnId;
    }
    value_type operator[](difference_type i) const
    {
      auto idx = mOrder[mIndex + i];
      return (*mData)[idx.getSource()][idx.getIndex()].columnId;
    }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncROF begin_bcIncROF() const { return Iter_bcIncROF(mTFData.rofDataRefs, mTFData.rofData, false); }
  Iter_bcIncROF end_bcIncROF() const { return Iter_bcIncROF(mTFData.rofDataRefs, mTFData.rofData, true); }

  Iter_orbitIncROF begin_orbitIncROF() const { return Iter_orbitIncROF(mTFData.rofDataRefs, mTFData.rofData, false); }
  Iter_orbitIncROF end_orbitIncROF() const { return Iter_orbitIncROF(mTFData.rofDataRefs, mTFData.rofData, true); }

  Iter_entriesROF begin_entriesROF() const { return Iter_entriesROF(mTFData.rofDataRefs, mTFData.rofData, false); }
  Iter_entriesROF end_entriesROF() const { return Iter_entriesROF(mTFData.rofDataRefs, mTFData.rofData, true); }

  Iter_evtypeROF begin_evtypeROF() const { return Iter_evtypeROF(mTFData.rofDataRefs, mTFData.rofData, false); }
  Iter_evtypeROF end_evtypeROF() const { return Iter_evtypeROF(mTFData.rofDataRefs, mTFData.rofData, true); }

  Iter_pattern begin_pattern() const { return Iter_pattern(mTFData.colDataRefs, mTFData.colData, false); }
  Iter_pattern end_pattern() const { return Iter_pattern(mTFData.colDataRefs, mTFData.colData, true); }

  Iter_deId begin_deId() const { return Iter_deId(mTFData.colDataRefs, mTFData.colData, false); }
  Iter_deId end_deId() const { return Iter_deId(mTFData.colDataRefs, mTFData.colData, true); }

  Iter_colId begin_colId() const { return Iter_colId(mTFData.colDataRefs, mTFData.colData, false); }
  Iter_colId end_colId() const { return Iter_colId(mTFData.colDataRefs, mTFData.colData, true); }

 private:
  const TFData& mTFData;
};

} // namespace mid
} // namespace o2

#endif
