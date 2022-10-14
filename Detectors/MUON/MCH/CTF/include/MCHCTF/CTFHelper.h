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
/// \brief  Helper for MCH CTF creation

#ifndef O2_MCH_CTF_HELPER_H
#define O2_MCH_CTF_HELPER_H

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/CTF.h"
#include <gsl/span>

namespace o2
{
namespace mch
{

class CTFHelper
{

 public:
  CTFHelper(const gsl::span<const o2::mch::ROFRecord>& rofData, const gsl::span<const o2::mch::Digit>& digData)
    : mROFData(rofData), mDigData(digData) {}

  CTFHeader createHeader()
  {
    CTFHeader h{o2::detectors::DetID::MCH, 0, 1, 0, // dummy timestamp, version 1.0
                uint32_t(mROFData.size()), uint32_t(mDigData.size()), 0, 0};
    if (mROFData.size()) {
      h.firstOrbit = mROFData[0].getBCData().orbit;
      h.firstBC = mROFData[0].getBCData().bc;
    }
    return h;
  }

  size_t getSize() const { return mROFData.size() * sizeof(o2::mch::ROFRecord) + mDigData.size() * sizeof(o2::mch::Digit); }

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
        if (mData[mIndex].getBCData().orbit == mData[mIndex - 1].getBCData().orbit) {
          return mData[mIndex].getBCData().bc - mData[mIndex - 1].getBCData().bc;
        } else {
          return mData[mIndex].getBCData().bc;
        }
      }
      return 0;
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      if (id) {
        if (mData[id].getBCData().orbit == mData[id - 1].getBCData().orbit) {
          return mData[id].getBCData().bc - mData[id - 1].getBCData().bc;
        } else {
          return mData[id].getBCData().bc;
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
    value_type operator*() const { return mIndex ? mData[mIndex].getBCData().orbit - mData[mIndex - 1].getBCData().orbit : 0; }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return id ? mData[id].getBCData().orbit - mData[id - 1].getBCData().orbit : 0;
    }
  };

  //_______________________________________________
  // Number of entries in the ROF
  class Iter_nDigitsROF : public _Iter<Iter_nDigitsROF, ROFRecord, uint32_t>
  {
   public:
    using _Iter<Iter_nDigitsROF, ROFRecord, uint32_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getNEntries(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getNEntries(); }
  };

  //_______________________________________________
  class Iter_tfTime : public _Iter<Iter_tfTime, Digit, int32_t>
  {
   public:
    using _Iter<Iter_tfTime, Digit, int32_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getTime(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getTime(); }
  };

  //_______________________________________________
  class Iter_nSamples : public _Iter<Iter_nSamples, Digit, uint16_t>
  {
   public:
    using _Iter<Iter_nSamples, Digit, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getNofSamples(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getNofSamples(); }
  };

  //_______________________________________________
  class Iter_isSaturated : public _Iter<Iter_isSaturated, Digit, uint8_t>
  {
   public:
    using _Iter<Iter_isSaturated, Digit, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].isSaturated(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].isSaturated(); }
  };

  //_______________________________________________
  class Iter_detID : public _Iter<Iter_detID, Digit, int16_t>
  {
   public:
    using _Iter<Iter_detID, Digit, int16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getDetID(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getDetID(); }
  };

  //_______________________________________________
  class Iter_padID : public _Iter<Iter_padID, Digit, int16_t>
  {
   public:
    using _Iter<Iter_padID, Digit, int16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPadID(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getPadID(); }
  };

  //_______________________________________________
  class Iter_ADC : public _Iter<Iter_ADC, Digit, uint32_t>
  {
   public:
    using _Iter<Iter_ADC, Digit, uint32_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getADC(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].getADC(); }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncROF begin_bcIncROF() const { return Iter_bcIncROF(mROFData, false); }
  Iter_bcIncROF end_bcIncROF() const { return Iter_bcIncROF(mROFData, true); }

  Iter_orbitIncROF begin_orbitIncROF() const { return Iter_orbitIncROF(mROFData, false); }
  Iter_orbitIncROF end_orbitIncROF() const { return Iter_orbitIncROF(mROFData, true); }

  Iter_nDigitsROF begin_nDigitsROF() const { return Iter_nDigitsROF(mROFData, false); }
  Iter_nDigitsROF end_nDigitsROF() const { return Iter_nDigitsROF(mROFData, true); }

  Iter_tfTime begin_tfTime() const { return Iter_tfTime(mDigData, false); }
  Iter_tfTime end_tfTime() const { return Iter_tfTime(mDigData, true); }

  Iter_nSamples begin_nSamples() const { return Iter_nSamples(mDigData, false); }
  Iter_nSamples end_nSamples() const { return Iter_nSamples(mDigData, true); }

  Iter_isSaturated begin_isSaturated() const { return Iter_isSaturated(mDigData, false); }
  Iter_isSaturated end_isSaturated() const { return Iter_isSaturated(mDigData, true); }

  Iter_detID begin_detID() const { return Iter_detID(mDigData, false); }
  Iter_detID end_detID() const { return Iter_detID(mDigData, true); }

  Iter_padID begin_padID() const { return Iter_padID(mDigData, false); }
  Iter_padID end_padID() const { return Iter_padID(mDigData, true); }

  Iter_ADC begin_ADC() const { return Iter_ADC(mDigData, false); }
  Iter_ADC end_ADC() const { return Iter_ADC(mDigData, true); }

 private:
  const gsl::span<const o2::mch::ROFRecord> mROFData;
  const gsl::span<const o2::mch::Digit> mDigData;
};

} // namespace mch
} // namespace o2

#endif
