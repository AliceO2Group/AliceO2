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
/// \brief  Helper for HMPID CTF creation

#ifndef O2_HMPID_CTF_HELPER_H
#define O2_HMPID_CTF_HELPER_H

#include "DataFormatsHMP/CTF.h"
#include "DataFormatsHMP/Trigger.h"
#include "DataFormatsHMP/Digit.h"
#include <gsl/span>

namespace o2
{
namespace hmpid
{

class CTFHelper
{

 public:
  CTFHelper(const gsl::span<const Trigger>& trgRec,
            const gsl::span<const Digit>& digData)
    : mTrigRec(trgRec), mDigData(digData), mDigStart(digData.size())
  {
    // flag start of new trigger for digits
    for (const auto& trg : mTrigRec) {
      if (trg.getNumberOfObjects()) {
        mDigStart[trg.getFirstEntry()] = true;
      }
    }
  }

  CTFHeader createHeader()
  {
    CTFHeader h{o2::detectors::DetID::HMP, 0, 1, 0, // dummy timestamp, version 1.0
                uint32_t(mTrigRec.size()), uint32_t(mDigData.size()), 0, 0};
    if (mTrigRec.size()) {
      h.firstOrbit = mTrigRec[0].getOrbit();
      h.firstBC = mTrigRec[0].getBc();
    }
    return h;
  }

  size_t getSize() const { return mTrigRec.size() * sizeof(Trigger) + mDigData.size() * sizeof(Digit); }

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

    const I operator++(int)
    {
      auto res = *this;
      ++mIndex;
      return res;
    }

    const I& operator--()
    {
      mIndex--;
      return (I&)(*this);
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
    gsl::span<const D> mData{};
    size_t mIndex = 0;
  };

  //_______________________________________________
  // BC difference wrt previous if in the same orbit, otherwise the abs.value.
  // For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_bcIncTrig : public _Iter<Iter_bcIncTrig, Trigger, uint16_t>
  {
   public:
    using _Iter<Iter_bcIncTrig, Trigger, uint16_t>::_Iter;
    value_type operator*() const
    {
      if (mIndex) {
        if (mData[mIndex].getOrbit() == mData[mIndex - 1].getOrbit()) {
          return mData[mIndex].getBc() - mData[mIndex - 1].getBc();
        } else {
          return mData[mIndex].getBc();
        }
      }
      return 0;
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      if (id) {
        if (mData[id].getOrbit() == mData[id - 1].getOrbit()) {
          return mData[id].getBc() - mData[id - 1].getBc();
        } else {
          return mData[id].getBc();
        }
      }
      return 0;
    }
  };

  //_______________________________________________
  // Orbit difference wrt previous. For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_orbitIncTrig : public _Iter<Iter_orbitIncTrig, Trigger, uint32_t>
  {
   public:
    using _Iter<Iter_orbitIncTrig, Trigger, uint32_t>::_Iter;
    value_type operator*() const { return mIndex ? mData[mIndex].getOrbit() - mData[mIndex - 1].getOrbit() : 0; }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return id ? mData[id].getOrbit() - mData[id - 1].getOrbit() : 0;
    }
  };

  //_______________________________________________
  // Number of digits for trigger
  class Iter_entriesDig : public _Iter<Iter_entriesDig, Trigger, uint32_t>
  {
   public:
    using _Iter<Iter_entriesDig, Trigger, uint32_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getNumberOfObjects(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getNumberOfObjects(); }
  };

  //_______________________________________________
  class Iter_ChID : public _Iter<Iter_ChID, Digit, uint8_t>
  {
   private:
    const std::vector<bool>* mTrigStart{nullptr};

   public:
    using _Iter<Iter_ChID, Digit, uint8_t>::_Iter;
    Iter_ChID(const std::vector<bool>* ts, const gsl::span<const Digit>& data, bool end) : mTrigStart(ts), _Iter(data, end) {}
    Iter_ChID() = default;

    // assume sorting in ChID: for the 1st digit of the trigger return the abs ChID, for the following ones: difference to previous ChID
    value_type operator*() const
    {
      return (*mTrigStart)[mIndex] ? mData[mIndex].getCh() : mData[mIndex].getCh() - mData[mIndex - 1].getCh();
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return (*mTrigStart)[id] ? mData[id].getCh() : mData[id].getCh() - mData[id - 1].getCh();
    }
  };

  //_______________________________________________
  class Iter_Q : public _Iter<Iter_Q, Digit, uint16_t>
  {
   public:
    using _Iter<Iter_Q, Digit, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getQ(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getQ(); }
  };

  //_______________________________________________
  class Iter_Ph : public _Iter<Iter_Ph, Digit, uint8_t>
  {
   public:
    using _Iter<Iter_Ph, Digit, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPh(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getPh(); }
  };

  //_______________________________________________
  class Iter_X : public _Iter<Iter_X, Digit, uint8_t>
  {
   public:
    using _Iter<Iter_X, Digit, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getX(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getX(); }
  };

  //_______________________________________________
  class Iter_Y : public _Iter<Iter_Y, Digit, uint8_t>
  {
   public:
    using _Iter<Iter_Y, Digit, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getY(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getY(); }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncTrig begin_bcIncTrig() const { return Iter_bcIncTrig(mTrigRec, false); }
  Iter_bcIncTrig end_bcIncTrig() const { return Iter_bcIncTrig(mTrigRec, true); }

  Iter_orbitIncTrig begin_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigRec, false); }
  Iter_orbitIncTrig end_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigRec, true); }

  Iter_entriesDig begin_entriesDig() const { return Iter_entriesDig(mTrigRec, false); }
  Iter_entriesDig end_entriesDig() const { return Iter_entriesDig(mTrigRec, true); }

  Iter_ChID begin_ChID() const { return Iter_ChID(&mDigStart, mDigData, false); }
  Iter_ChID end_ChID() const { return Iter_ChID(&mDigStart, mDigData, true); }

  Iter_Q begin_Q() const { return Iter_Q(mDigData, false); }
  Iter_Q end_Q() const { return Iter_Q(mDigData, true); }

  Iter_Ph begin_Ph() const { return Iter_Ph(mDigData, false); }
  Iter_Ph end_Ph() const { return Iter_Ph(mDigData, true); }

  Iter_X begin_X() const { return Iter_X(mDigData, false); }
  Iter_X end_X() const { return Iter_X(mDigData, true); }

  Iter_Y begin_Y() const { return Iter_Y(mDigData, false); }
  Iter_Y end_Y() const { return Iter_Y(mDigData, true); }

 private:
  const gsl::span<const o2::hmpid::Trigger> mTrigRec;
  const gsl::span<const o2::hmpid::Digit> mDigData;
  std::vector<bool> mDigStart;
};

} // namespace hmpid
} // namespace o2

#endif
