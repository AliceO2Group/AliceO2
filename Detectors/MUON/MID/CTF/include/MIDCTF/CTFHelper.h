// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <gsl/span>

namespace o2
{
namespace mid
{

class CTFHelper
{

 public:
  CTFHelper(const gsl::span<const o2::mid::ROFRecord>& rofData, const gsl::span<const o2::mid::ColumnData>& colData)
    : mROFData(rofData), mColData(colData) {}

  CTFHeader createHeader()
  {
    CTFHeader h{uint32_t(mROFData.size()), uint32_t(mColData.size()), 0, 0};
    if (mROFData.size()) {
      h.firstOrbit = mROFData[0].interactionRecord.orbit;
      h.firstBC = mROFData[0].interactionRecord.bc;
    }
    return h;
  }

  size_t getSize() const { return mROFData.size() * sizeof(o2::mid::ROFRecord) + mColData.size() * sizeof(o2::mid::ColumnData); }

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

    _Iter(const gsl::span<const D>& data, bool end = false) : mData(data), mIndex(end ? M * data.size() : 0){};
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

   protected:
    gsl::span<const D> mData{};
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
      if (mIndex) {
        if (mData[mIndex].interactionRecord.orbit == mData[mIndex - 1].interactionRecord.orbit) {
          return mData[mIndex].interactionRecord.bc - mData[mIndex - 1].interactionRecord.bc;
        } else {
          return mData[mIndex].interactionRecord.bc;
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
    value_type operator*() const { return mIndex ? mData[mIndex].interactionRecord.orbit - mData[mIndex - 1].interactionRecord.orbit : 0; }
  };

  //_______________________________________________
  // Number of entries in the ROF
  class Iter_entriesROF : public _Iter<Iter_entriesROF, ROFRecord, uint16_t>
  {
   public:
    using _Iter<Iter_entriesROF, ROFRecord, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].nEntries; }
  };

  //_______________________________________________
  // Event type for the ROF
  class Iter_evtypeROF : public _Iter<Iter_evtypeROF, ROFRecord, uint8_t>
  {
   public:
    using _Iter<Iter_evtypeROF, ROFRecord, uint8_t>::_Iter;
    value_type operator*() const { return value_type(mData[mIndex].eventType); }
  };

  //_______________________________________________
  class Iter_pattern : public _Iter<Iter_pattern, ColumnData, uint16_t, 5>
  {
   public:
    using _Iter<Iter_pattern, ColumnData, uint16_t, 5>::_Iter;
    value_type operator*() const { return mData[mIndex / 5].patterns[mIndex % 5]; }
  };

  //_______________________________________________
  class Iter_deId : public _Iter<Iter_deId, ColumnData, uint8_t>
  {
   public:
    using _Iter<Iter_deId, ColumnData, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].deId; }
  };

  //_______________________________________________
  class Iter_colId : public _Iter<Iter_colId, ColumnData, uint8_t>
  {
   public:
    using _Iter<Iter_colId, ColumnData, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].columnId; }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncROF begin_bcIncROF() const { return Iter_bcIncROF(mROFData, false); }
  Iter_bcIncROF end_bcIncROF() const { return Iter_bcIncROF(mROFData, true); }

  Iter_orbitIncROF begin_orbitIncROF() const { return Iter_orbitIncROF(mROFData, false); }
  Iter_orbitIncROF end_orbitIncROF() const { return Iter_orbitIncROF(mROFData, true); }

  Iter_entriesROF begin_entriesROF() const { return Iter_entriesROF(mROFData, false); }
  Iter_entriesROF end_entriesROF() const { return Iter_entriesROF(mROFData, true); }

  Iter_evtypeROF begin_evtypeROF() const { return Iter_evtypeROF(mROFData, false); }
  Iter_evtypeROF end_evtypeROF() const { return Iter_evtypeROF(mROFData, true); }

  Iter_pattern begin_pattern() const { return Iter_pattern(mColData, false); }
  Iter_pattern end_pattern() const { return Iter_pattern(mColData, true); }

  Iter_deId begin_deId() const { return Iter_deId(mColData, false); }
  Iter_deId end_deId() const { return Iter_deId(mColData, true); }

  Iter_colId begin_colId() const { return Iter_colId(mColData, false); }
  Iter_colId end_colId() const { return Iter_colId(mColData, true); }

 private:
  const gsl::span<const o2::mid::ROFRecord> mROFData;
  const gsl::span<const o2::mid::ColumnData> mColData;
};

} // namespace mid
} // namespace o2

#endif
