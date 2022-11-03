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
/// \brief  Helper for CTP CTF creation

#ifndef O2_CTP_CTF_HELPER_H
#define O2_CTP_CTF_HELPER_H

#include "DataFormatsCTP/CTF.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/LumiInfo.h"
#include <gsl/span>

namespace o2
{
namespace ctp
{

class CTFHelper
{

 public:
  CTFHelper(const gsl::span<const CTPDigit>& data) : mData(data) {}

  static constexpr int CTPInpNBytes = CTP_NINPUTS / 8 + (CTP_NINPUTS % 8 > 0);
  static constexpr int CTPClsNBytes = CTP_NCLASSES / 8 + (CTP_NCLASSES % 8 > 0);

  CTFHeader createHeader(const LumiInfo& lumi)
  {
    CTFHeader h{o2::detectors::DetID::CTP, 0, 1, 0, // dummy timestamp, version 1.0
                lumi.counts, lumi.nHBFCounted, lumi.orbit,
                uint32_t(mData.size()), 0, 0};
    if (mData.size()) {
      h.firstOrbit = mData[0].intRecord.orbit;
      h.firstBC = mData[0].intRecord.bc;
    }
    return h;
  }

  size_t getSize() const { return mData.size() * sizeof(CTPDigit); }

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
  class Iter_bcIncTrig : public _Iter<Iter_bcIncTrig, CTPDigit, uint16_t>
  {
   public:
    using _Iter<Iter_bcIncTrig, CTPDigit, uint16_t>::_Iter;
    value_type operator*() const
    {
      if (mIndex) {
        if (mData[mIndex].intRecord.orbit == mData[mIndex - 1].intRecord.orbit) {
          return mData[mIndex].intRecord.bc - mData[mIndex - 1].intRecord.bc;
        } else {
          return mData[mIndex].intRecord.bc;
        }
      }
      return 0;
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      if (id) {
        if (mData[id].intRecord.orbit == mData[id - 1].intRecord.orbit) {
          return mData[id].intRecord.bc - mData[id - 1].intRecord.bc;
        } else {
          return mData[id].intRecord.bc;
        }
      }
      return 0;
    }
  };

  //_______________________________________________
  // Orbit difference wrt previous. For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_orbitIncTrig : public _Iter<Iter_orbitIncTrig, CTPDigit, uint32_t>
  {
   public:
    using _Iter<Iter_orbitIncTrig, CTPDigit, uint32_t>::_Iter;
    value_type operator*() const { return mIndex ? mData[mIndex].intRecord.orbit - mData[mIndex - 1].intRecord.orbit : 0; }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return id ? mData[id].intRecord.orbit - mData[id - 1].intRecord.orbit : 0;
    }
  };

  //_______________________________________________
  class Iter_bytesInput : public _Iter<Iter_bytesInput, CTPDigit, uint8_t, CTPInpNBytes>
  {
   public:
    using _Iter<Iter_bytesInput, CTPDigit, uint8_t, CTPInpNBytes>::_Iter;
    value_type operator*() const
    {
      return static_cast<uint8_t>(((mData[mIndex / CTPInpNBytes].CTPInputMask.to_ullong()) >> (8 * (mIndex % CTPInpNBytes))) & 0xff);
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return static_cast<uint8_t>(((mData[id / CTPInpNBytes].CTPInputMask.to_ullong()) >> (8 * (id % CTPInpNBytes))) & 0xff);
    }
  };

  //_______________________________________________
  class Iter_bytesClass : public _Iter<Iter_bytesClass, CTPDigit, uint8_t, CTPClsNBytes>
  {
   public:
    using _Iter<Iter_bytesClass, CTPDigit, uint8_t, CTPClsNBytes>::_Iter;
    value_type operator*() const
    {
      return static_cast<uint8_t>(((mData[mIndex / CTPClsNBytes].CTPClassMask.to_ullong()) >> (8 * (mIndex % CTPClsNBytes))) & 0xff);
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return static_cast<uint8_t>(((mData[id / CTPClsNBytes].CTPClassMask.to_ullong()) >> (8 * (id % CTPClsNBytes))) & 0xff);
    }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncTrig begin_bcIncTrig() const { return Iter_bcIncTrig(mData, false); }
  Iter_bcIncTrig end_bcIncTrig() const { return Iter_bcIncTrig(mData, true); }

  Iter_orbitIncTrig begin_orbitIncTrig() const { return Iter_orbitIncTrig(mData, false); }
  Iter_orbitIncTrig end_orbitIncTrig() const { return Iter_orbitIncTrig(mData, true); }

  Iter_bytesInput begin_bytesInput() const { return Iter_bytesInput(mData, false); }
  Iter_bytesInput end_bytesInput() const { return Iter_bytesInput(mData, true); }

  Iter_bytesClass begin_bytesClass() const { return Iter_bytesClass(mData, false); }
  Iter_bytesClass end_bytesClass() const { return Iter_bytesClass(mData, true); }

 private:
  const gsl::span<const o2::ctp::CTPDigit> mData;
};

} // namespace ctp
} // namespace o2

#endif
