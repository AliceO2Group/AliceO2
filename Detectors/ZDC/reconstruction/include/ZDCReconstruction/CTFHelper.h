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
/// \brief  Helper for ZDC CTF creation

#ifndef O2_ZDC_CTF_HELPER_H
#define O2_ZDC_CTF_HELPER_H

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "DataFormatsZDC/CTF.h"
#include <gsl/span>

namespace o2
{
namespace zdc
{

class CTFHelper
{

 public:
  CTFHelper(const gsl::span<const BCData>& trgData,
            const gsl::span<const ChannelData>& chanData,
            const gsl::span<const OrbitData>& pedData)
    : mTrigData(trgData), mChanData(chanData), mEOData(pedData) {}

  CTFHeader createHeader()
  {
    CTFHeader h{o2::detectors::DetID::ZDC, 0, 1, 0, // dummy timestamp, version 1.0
                uint32_t(mTrigData.size()), uint32_t(mChanData.size()), uint32_t(mEOData.size()), 0, 0, 0};
    if (mTrigData.size()) {
      h.firstOrbit = mTrigData[0].ir.orbit;
      h.firstBC = mTrigData[0].ir.bc;
    }
    if (mEOData.size()) {
      h.firstOrbitEOData = mEOData[0].ir.orbit;
      h.firstScaler = mEOData[0].scaler; // then we store increments
    }
    return h;
  }

  size_t getSize() const { return mTrigData.size() * sizeof(BCData) + mChanData.size() * sizeof(ChannelData) + mEOData.size() * sizeof(OrbitData); }

  //>>> =========================== ITERATORS ========================================

  template <typename I, typename D, typename T, int M = 1>
  class _Iter
  {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = const T*;
    using reference = const T&;
    using iterator_category = std::random_access_iterator_tag;

    _Iter(const gsl::span<const D>& data, bool end = false) : mData(data), mIndex(end ? M * data.size() : 0){};
    _Iter() = default;

    inline I& operator++() noexcept
    {
      ++mIndex;
      return static_cast<I&>(*this);
    }

    inline I operator++(int)
    {
      I res = *(static_cast<I*>(this));
      ++mIndex;
      return res;
    }

    inline I& operator--() noexcept
    {
      mIndex--;
      return static_cast<I&>(*this);
    }

    inline I operator--(int)
    {
      I res = *(static_cast<I*>(this));
      --mIndex;
      return res;
    }

    I& operator+=(difference_type i) noexcept
    {
      mIndex += i;
      return static_cast<I&>(*this);
    }

    I operator+(difference_type i) const
    {
      I res = *(const_cast<I*>(static_cast<const I*>(this)));
      return res += i;
    }

    I& operator-=(difference_type i) noexcept
    {
      mIndex -= i;
      return static_cast<I&>(*this);
    }

    I operator-(difference_type i) const
    {
      I res = *(const_cast<I*>(static_cast<const I*>(this)));
      return res -= i;
    }

    difference_type operator-(const I& other) const noexcept { return mIndex - other.mIndex; }

    inline friend I operator+(difference_type i, const I& iter) { return iter + i; };

    bool operator!=(const I& other) const noexcept { return mIndex != other.mIndex; }
    bool operator==(const I& other) const noexcept { return mIndex == other.mIndex; }
    bool operator>(const I& other) const noexcept { return mIndex > other.mIndex; }
    bool operator<(const I& other) const noexcept { return mIndex < other.mIndex; }
    bool operator>=(const I& other) const noexcept { return mIndex >= other.mIndex; }
    bool operator<=(const I& other) const noexcept { return mIndex <= other.mIndex; }

   protected:
    gsl::span<const D> mData{};
    difference_type mIndex = 0;
  };

  //_______________________________________________
  // BC difference wrt previous if in the same orbit, otherwise the abs.value.
  // For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_bcIncTrig : public _Iter<Iter_bcIncTrig, BCData, int16_t>
  {
   public:
    using _Iter<Iter_bcIncTrig, BCData, int16_t>::_Iter;
    value_type operator*() const
    {
      if (mIndex) {
        if (mData[mIndex].ir.orbit == mData[mIndex - 1].ir.orbit) {
          return value_type(mData[mIndex].ir.bc - mData[mIndex - 1].ir.bc);
        } else {
          return value_type(mData[mIndex].ir.bc);
        }
      }
      return 0;
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      if (id) {
        if (mData[id].ir.orbit == mData[id - 1].ir.orbit) {
          return value_type(mData[id].ir.bc - mData[id - 1].ir.bc);
        } else {
          return value_type(mData[id].ir.bc);
        }
      }
      return 0;
    }
  };

  /////////////////////////////////// BCData iterators ////////////////////////////////////////
  //_______________________________________________
  // Orbit difference wrt previous. For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_orbitIncTrig : public _Iter<Iter_orbitIncTrig, BCData, int32_t>
  {
   public:
    using _Iter<Iter_orbitIncTrig, BCData, int32_t>::_Iter;
    value_type operator*() const { return value_type(mIndex ? mData[mIndex].ir.orbit - mData[mIndex - 1].ir.orbit : 0); }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return value_type(id ? mData[id].ir.orbit - mData[id - 1].ir.orbit : 0);
    }
  };

  //_______________________________________________
  // Modules trigger words, NModules words per trigger, iterate over the BCData and its modules
  class Iter_moduleTrig : public _Iter<Iter_moduleTrig, BCData, uint16_t, NModules>
  {
   public:
    using _Iter<Iter_moduleTrig, BCData, uint16_t, NModules>::_Iter;
    value_type operator*() const { return mData[mIndex / NModules].moduleTriggers[mIndex % NModules]; }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return mData[id / NModules].moduleTriggers[id % NModules];
    }
  };

  //_______________________________________________
  // ZDC channels pattern word: 32b word is saved as 2 16 bit words
  class Iter_channelsHL : public _Iter<Iter_channelsHL, BCData, uint16_t, 2>
  {
   public:
    using _Iter<Iter_channelsHL, BCData, uint16_t, 2>::_Iter;
    value_type operator*() const { return uint16_t(mIndex & 0x1 ? mData[mIndex / 2].channels : mData[mIndex / 2].channels >> 16); }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return uint16_t(id & 0x1 ? mData[id / 2].channels : mData[id / 2].channels >> 16);
    }
  };

  //_______________________________________________
  // ZDC trigger word: 32b word is saved as 2 16 bit words
  class Iter_triggersHL : public _Iter<Iter_triggersHL, BCData, uint16_t, 2>
  {
   public:
    using _Iter<Iter_triggersHL, BCData, uint16_t, 2>::_Iter;
    value_type operator*() const { return uint16_t(mIndex & 0x1 ? mData[mIndex / 2].triggers : mData[mIndex / 2].triggers >> 16); }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return uint16_t(id & 0x1 ? mData[id / 2].triggers : mData[id / 2].triggers >> 16);
    }
  };

  //_______________________________________________
  // Alice external trigger word
  class Iter_extTriggers : public _Iter<Iter_extTriggers, BCData, uint8_t>
  {
   public:
    using _Iter<Iter_extTriggers, BCData, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].ext_triggers; }
    value_type operator[](difference_type i) const { return mData[mIndex + i].ext_triggers; }
  };

  //_______________________________________________
  // Number of channels for trigger
  class Iter_nchanTrig : public _Iter<Iter_nchanTrig, BCData, uint16_t>
  {
   public:
    using _Iter<Iter_nchanTrig, BCData, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].ref.getEntries(); }
    value_type operator[](difference_type i) const { return mData[mIndex + i].ref.getEntries(); }
  };

  ////////////////////////// ChannelData iterators /////////////////////////////

  //_______________________________________________
  class Iter_chanID : public _Iter<Iter_chanID, ChannelData, uint8_t>
  {
   public:
    using _Iter<Iter_chanID, ChannelData, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].id; }
    value_type operator[](difference_type i) const { return mData[mIndex + i].id; }
  };

  //_______________________________________________
  class Iter_chanData : public _Iter<Iter_chanData, ChannelData, uint16_t, NTimeBinsPerBC>
  {
   public:
    using _Iter<Iter_chanData, ChannelData, uint16_t, NTimeBinsPerBC>::_Iter;
    value_type operator*() const { return mData[mIndex / NTimeBinsPerBC].data[mIndex % NTimeBinsPerBC]; }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return mData[id / NTimeBinsPerBC].data[id % NTimeBinsPerBC];
    }
  };

  ////////////////////////// OrbitData iterators /////////////////////////////

  //_______________________________________________
  // Orbit difference wrt previous. For the very 1st entry return 0 (diff wrt 1st BC in the CTF header)
  class Iter_orbitIncEOD : public _Iter<Iter_orbitIncEOD, OrbitData, int32_t>
  {
   public:
    using _Iter<Iter_orbitIncEOD, OrbitData, int32_t>::_Iter;
    value_type operator*() const { return value_type(mIndex ? mData[mIndex].ir.orbit - mData[mIndex - 1].ir.orbit : 0); }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return value_type(id ? mData[id].ir.orbit - mData[id - 1].ir.orbit : 0);
    }
  };

  //_______________________________________________
  class Iter_pedData : public _Iter<Iter_pedData, OrbitData, uint16_t, NChannels>
  {
   public:
    using _Iter<Iter_pedData, OrbitData, uint16_t, NChannels>::_Iter;
    value_type operator*() const { return mData[mIndex / NChannels].data[mIndex % NChannels]; }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      return mData[id / NChannels].data[id % NChannels];
    }
  };

  //_______________________________________________
  class Iter_sclInc : public _Iter<Iter_sclInc, OrbitData, int16_t, NChannels>
  {
   public:
    using _Iter<Iter_sclInc, OrbitData, int16_t, NChannels>::_Iter;
    value_type operator*() const
    {
      // define with respect to previous orbit
      int slot = mIndex / NChannels, chan = mIndex % NChannels;
      return value_type(slot ? mData[slot].scaler[chan] - mData[slot - 1].scaler[chan] : 0);
    }
    value_type operator[](difference_type i) const
    {
      size_t id = mIndex + i;
      int slot = id / NChannels, chan = id % NChannels;
      return value_type(slot ? mData[slot].scaler[chan] - mData[slot - 1].scaler[chan] : 0);
    }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncTrig begin_bcIncTrig() const { return Iter_bcIncTrig(mTrigData, false); }
  Iter_bcIncTrig end_bcIncTrig() const { return Iter_bcIncTrig(mTrigData, true); }

  Iter_orbitIncTrig begin_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigData, false); }
  Iter_orbitIncTrig end_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigData, true); }

  Iter_moduleTrig begin_moduleTrig() const { return Iter_moduleTrig(mTrigData, false); }
  Iter_moduleTrig end_moduleTrig() const { return Iter_moduleTrig(mTrigData, true); }

  Iter_channelsHL begin_channelsHL() const { return Iter_channelsHL(mTrigData, false); }
  Iter_channelsHL end_channelsHL() const { return Iter_channelsHL(mTrigData, true); }

  Iter_triggersHL begin_triggersHL() const { return Iter_triggersHL(mTrigData, false); }
  Iter_triggersHL end_triggersHL() const { return Iter_triggersHL(mTrigData, true); }

  Iter_extTriggers begin_extTriggers() const { return Iter_extTriggers(mTrigData, false); }
  Iter_extTriggers end_extTriggers() const { return Iter_extTriggers(mTrigData, true); }

  Iter_nchanTrig begin_nchanTrig() const { return Iter_nchanTrig(mTrigData, false); }
  Iter_nchanTrig end_nchanTrig() const { return Iter_nchanTrig(mTrigData, true); }

  Iter_chanID begin_chanID() const { return Iter_chanID(mChanData, false); }
  Iter_chanID end_chanID() const { return Iter_chanID(mChanData, true); }

  Iter_chanData begin_chanData() const { return Iter_chanData(mChanData, false); }
  Iter_chanData end_chanData() const { return Iter_chanData(mChanData, true); }

  Iter_orbitIncEOD begin_orbitIncEOD() const { return Iter_orbitIncEOD(mEOData, false); }
  Iter_orbitIncEOD end_orbitIncEOD() const { return Iter_orbitIncEOD(mEOData, true); }

  Iter_pedData begin_pedData() const { return Iter_pedData(mEOData, false); }
  Iter_pedData end_pedData() const { return Iter_pedData(mEOData, true); }

  Iter_sclInc begin_sclInc() const { return Iter_sclInc(mEOData, false); }
  Iter_sclInc end_sclInc() const { return Iter_sclInc(mEOData, true); }

 private:
  const gsl::span<const o2::zdc::BCData> mTrigData;
  const gsl::span<const o2::zdc::ChannelData> mChanData;
  const gsl::span<const o2::zdc::OrbitData> mEOData;
};

} // namespace zdc
} // namespace o2

#endif
