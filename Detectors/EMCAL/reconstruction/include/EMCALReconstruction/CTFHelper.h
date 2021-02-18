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
/// \brief  Helper for EMCAL CTF creation

#ifndef O2_EMCAL_CTF_HELPER_H
#define O2_EMCAL_CTF_HELPER_H

#include "DataFormatsEMCAL/CTF.h"
#include <gsl/span>

namespace o2
{
namespace emcal
{

class CTFHelper
{

 public:
  CTFHelper(const gsl::span<const TriggerRecord>& trgData, const gsl::span<const Cell>& cellData)
    : mTrigData(trgData), mCellData(cellData) {}

  CTFHeader createHeader()
  {
    CTFHeader h{uint32_t(mTrigData.size()), uint32_t(mCellData.size()), 0, 0};
    if (mTrigData.size()) {
      h.firstOrbit = mTrigData[0].getBCData().orbit;
      h.firstBC = mTrigData[0].getBCData().bc;
    }
    return h;
  }

  size_t getSize() const { return mTrigData.size() * sizeof(TriggerRecord) + mCellData.size() * sizeof(Cell); }

  //>>> =========================== ITERATORS ========================================

  template <typename I, typename D, typename T>
  class _Iter
  {
   public:
    using difference_type = int64_t;
    using value_type = T;
    using pointer = const T*;
    using reference = const T&;
    using iterator_category = std::random_access_iterator_tag;

    _Iter(const gsl::span<const D>& data, bool end = false) : mData(data), mIndex(end ? data.size() : 0){};
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
  class Iter_bcIncTrig : public _Iter<Iter_bcIncTrig, TriggerRecord, uint16_t>
  {
   public:
    using _Iter<Iter_bcIncTrig, TriggerRecord, uint16_t>::_Iter;
    value_type operator*() const
    {
      if (mIndex) {
        if (mData[mIndex].getBCData().orbit == mData[mIndex - 1].getBCData().orbit) {
          return mData[mIndex].getBCData().bc - mData[mIndex - 1].getBCData().bc;
        } else {
          return mData[mIndex].getBCData().bc;
        }
      }
      return 0;
    }
  };

  //_______________________________________________
  // Orbit difference wrt previous. For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_orbitIncTrig : public _Iter<Iter_orbitIncTrig, TriggerRecord, uint32_t>
  {
   public:
    using _Iter<Iter_orbitIncTrig, TriggerRecord, uint32_t>::_Iter;
    value_type operator*() const { return mIndex ? mData[mIndex].getBCData().orbit - mData[mIndex - 1].getBCData().orbit : 0; }
  };

  //_______________________________________________
  // Number of cells for trigger
  class Iter_entriesTrig : public _Iter<Iter_entriesTrig, TriggerRecord, uint16_t>
  {
   public:
    using _Iter<Iter_entriesTrig, TriggerRecord, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getNumberOfObjects(); }
  };

  //_______________________________________________
  class Iter_towerID : public _Iter<Iter_towerID, Cell, uint16_t>
  {
   public:
    using _Iter<Iter_towerID, Cell, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPackedTowerID(); }
  };

  //_______________________________________________
  class Iter_time : public _Iter<Iter_time, Cell, uint16_t>
  {
   public:
    using _Iter<Iter_time, Cell, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPackedTime(); }
  };

  //_______________________________________________
  class Iter_energy : public _Iter<Iter_energy, Cell, uint16_t>
  {
   public:
    using _Iter<Iter_energy, Cell, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPackedEnergy(); }
  };

  //_______________________________________________
  class Iter_status : public _Iter<Iter_status, Cell, uint8_t>
  {
   public:
    using _Iter<Iter_status, Cell, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPackedCellStatus(); }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncTrig begin_bcIncTrig() const { return Iter_bcIncTrig(mTrigData, false); }
  Iter_bcIncTrig end_bcIncTrig() const { return Iter_bcIncTrig(mTrigData, true); }

  Iter_orbitIncTrig begin_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigData, false); }
  Iter_orbitIncTrig end_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigData, true); }

  Iter_entriesTrig begin_entriesTrig() const { return Iter_entriesTrig(mTrigData, false); }
  Iter_entriesTrig end_entriesTrig() const { return Iter_entriesTrig(mTrigData, true); }

  Iter_towerID begin_towerID() const { return Iter_towerID(mCellData, false); }
  Iter_towerID end_towerID() const { return Iter_towerID(mCellData, true); }

  Iter_time begin_time() const { return Iter_time(mCellData, false); }
  Iter_time end_time() const { return Iter_time(mCellData, true); }

  Iter_energy begin_energy() const { return Iter_energy(mCellData, false); }
  Iter_energy end_energy() const { return Iter_energy(mCellData, true); }

  Iter_status begin_status() const { return Iter_status(mCellData, false); }
  Iter_status end_status() const { return Iter_status(mCellData, true); }

 private:
  const gsl::span<const o2::emcal::TriggerRecord> mTrigData;
  const gsl::span<const o2::emcal::Cell> mCellData;
};

} // namespace emcal
} // namespace o2

#endif
