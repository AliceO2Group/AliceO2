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
/// \brief  Helper for TRD CTF creation

#ifndef O2_TRD_CTF_HELPER_H
#define O2_TRD_CTF_HELPER_H

#include "DataFormatsTRD/CTF.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Digit.h"
#include <gsl/span>
#include <bitset>

namespace o2
{
namespace trd
{

class CTFHelper
{

 public:
  CTFHelper(const gsl::span<const TriggerRecord>& trgRec,
            const gsl::span<const Tracklet64>& trkData, const gsl::span<const Digit>& digData)
    : mTrigRec(trgRec), mTrkData(trkData), mDigData(digData), mTrkStart(trkData.size()), mDigStart(digData.size())
  {
    // flag start of new trigger for tracklets and digits
    for (const auto& trg : mTrigRec) {
      if (trg.getNumberOfTracklets()) {
        mTrkStart[trg.getFirstTracklet()] = true;
      }
      if (trg.getNumberOfDigits()) {
        mDigStart[trg.getFirstDigit()] = true;
      }
    }
  }

  CTFHeader createHeader()
  {
    CTFHeader h{o2::detectors::DetID::TRD, 0, 1, 0, // dummy timestamp, version 1.0
                uint32_t(mTrigRec.size()), uint32_t(mTrkData.size()), uint32_t(mDigData.size()), 0, 0, 0};
    if (mTrigRec.size()) {
      h.firstOrbit = mTrigRec[0].getBCData().orbit;
      h.firstBC = mTrigRec[0].getBCData().bc;
    }
    if (mTrkData.size()) {
      h.format = (uint16_t)mTrkData[0].getFormat();
    }
    return h;
  }

  size_t getSize() const { return mTrigRec.size() * sizeof(TriggerRecord) + mTrkData.size() * sizeof(Tracklet64) + mDigData.size() * sizeof(Digit); }

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
  class Iter_orbitIncTrig : public _Iter<Iter_orbitIncTrig, TriggerRecord, int32_t>
  {
   public:
    using _Iter<Iter_orbitIncTrig, TriggerRecord, int32_t>::_Iter;
    value_type operator*() const { return mIndex ? mData[mIndex].getBCData().orbit - mData[mIndex - 1].getBCData().orbit : 0; }
  };

  //_______________________________________________
  // Number of tracklets for trigger
  class Iter_entriesTrk : public _Iter<Iter_entriesTrk, TriggerRecord, uint32_t>
  {
   public:
    using _Iter<Iter_entriesTrk, TriggerRecord, uint32_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getNumberOfTracklets(); }
  };

  //_______________________________________________
  // Number of digits for trigger
  class Iter_entriesDig : public _Iter<Iter_entriesDig, TriggerRecord, uint32_t>
  {
   public:
    using _Iter<Iter_entriesDig, TriggerRecord, uint32_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getNumberOfDigits(); }
  };

  //_______________________________________________
  class Iter_HCIDTrk : public _Iter<Iter_HCIDTrk, Tracklet64, uint16_t>
  {
   private:
    const std::vector<bool>* mTrigStart{nullptr};

   public:
    using _Iter<Iter_HCIDTrk, Tracklet64, uint16_t>::_Iter;
    Iter_HCIDTrk(const std::vector<bool>* ts, const gsl::span<const Tracklet64>& data, bool end) : mTrigStart(ts), _Iter(data, end) {}
    Iter_HCIDTrk() = default;

    // assume sorting in HCID: for the 1st tracklet of the trigger return the abs HCID, for the following ones: difference to previous HCID
    value_type operator*() const
    {
      return (*mTrigStart)[mIndex] ? mData[mIndex].getHCID() : mData[mIndex].getHCID() - mData[mIndex - 1].getHCID();
    }
  };

  //_______________________________________________
  class Iter_padrowTrk : public _Iter<Iter_padrowTrk, Tracklet64, uint8_t>
  {
   public:
    using _Iter<Iter_padrowTrk, Tracklet64, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPadRow(); }
  };

  //_______________________________________________
  class Iter_colTrk : public _Iter<Iter_colTrk, Tracklet64, uint8_t>
  {
   public:
    using _Iter<Iter_colTrk, Tracklet64, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getColumn(); }
  };

  //_______________________________________________
  class Iter_posTrk : public _Iter<Iter_posTrk, Tracklet64, uint16_t>
  {
   public:
    using _Iter<Iter_posTrk, Tracklet64, uint16_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPosition(); }
  };

  //_______________________________________________
  class Iter_slopeTrk : public _Iter<Iter_slopeTrk, Tracklet64, uint8_t>
  {
   public:
    using _Iter<Iter_slopeTrk, Tracklet64, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getSlope(); }
  };

  //_______________________________________________
  class Iter_pidTrk : public _Iter<Iter_pidTrk, Tracklet64, uint32_t>
  {
   public:
    using _Iter<Iter_pidTrk, Tracklet64, uint32_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getPID(); }
  };

  //_______________________________________________
  class Iter_CIDDig : public _Iter<Iter_CIDDig, Digit, uint16_t>
  {
   private:
    const std::vector<bool>* mTrigStart{nullptr};

   public:
    using _Iter<Iter_CIDDig, Digit, uint16_t>::_Iter;
    Iter_CIDDig(const std::vector<bool>* ts, const gsl::span<const Digit>& data, bool end) : mTrigStart(ts), _Iter(data, end) {}
    Iter_CIDDig() = default;

    // assume sorting in CID: for the 1st digit of the trigger return the abs CID, for the following ones: difference to previous CID
    value_type operator*() const
    {
      return (*mTrigStart)[mIndex] ? mData[mIndex].getDetector() : mData[mIndex].getDetector() - mData[mIndex - 1].getDetector();
    }
  };

  //_______________________________________________
  class Iter_ROBDig : public _Iter<Iter_ROBDig, Digit, uint8_t>
  {
   public:
    using _Iter<Iter_ROBDig, Digit, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getROB(); }
  };

  //_______________________________________________
  class Iter_MCMDig : public _Iter<Iter_MCMDig, Digit, uint8_t>
  {
   public:
    using _Iter<Iter_MCMDig, Digit, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getMCM(); }
  };

  //_______________________________________________
  class Iter_chanDig : public _Iter<Iter_chanDig, Digit, uint8_t>
  {
   public:
    using _Iter<Iter_chanDig, Digit, uint8_t>::_Iter;
    value_type operator*() const { return mData[mIndex].getChannel(); }
  };

  //_______________________________________________
  class Iter_ADCDig : public _Iter<Iter_ADCDig, Digit, uint16_t, constants::TIMEBINS>
  {
   public:
    using _Iter<Iter_ADCDig, Digit, uint16_t, constants::TIMEBINS>::_Iter;
    value_type operator*() const { return mData[mIndex / constants::TIMEBINS].getADC()[mIndex % constants::TIMEBINS]; }
  };

  //<<< =========================== ITERATORS ========================================

  Iter_bcIncTrig begin_bcIncTrig() const { return Iter_bcIncTrig(mTrigRec, false); }
  Iter_bcIncTrig end_bcIncTrig() const { return Iter_bcIncTrig(mTrigRec, true); }

  Iter_orbitIncTrig begin_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigRec, false); }
  Iter_orbitIncTrig end_orbitIncTrig() const { return Iter_orbitIncTrig(mTrigRec, true); }

  Iter_entriesTrk begin_entriesTrk() const { return Iter_entriesTrk(mTrigRec, false); }
  Iter_entriesTrk end_entriesTrk() const { return Iter_entriesTrk(mTrigRec, true); }

  Iter_entriesDig begin_entriesDig() const { return Iter_entriesDig(mTrigRec, false); }
  Iter_entriesDig end_entriesDig() const { return Iter_entriesDig(mTrigRec, true); }

  Iter_HCIDTrk begin_HCIDTrk() const { return Iter_HCIDTrk(&mTrkStart, mTrkData, false); }
  Iter_HCIDTrk end_HCIDTrk() const { return Iter_HCIDTrk(&mTrkStart, mTrkData, true); }

  Iter_padrowTrk begin_padrowTrk() const { return Iter_padrowTrk(mTrkData, false); }
  Iter_padrowTrk end_padrowTrk() const { return Iter_padrowTrk(mTrkData, true); }

  Iter_colTrk begin_colTrk() const { return Iter_colTrk(mTrkData, false); }
  Iter_colTrk end_colTrk() const { return Iter_colTrk(mTrkData, true); }

  Iter_posTrk begin_posTrk() const { return Iter_posTrk(mTrkData, false); }
  Iter_posTrk end_posTrk() const { return Iter_posTrk(mTrkData, true); }

  Iter_slopeTrk begin_slopeTrk() const { return Iter_slopeTrk(mTrkData, false); }
  Iter_slopeTrk end_slopeTrk() const { return Iter_slopeTrk(mTrkData, true); }

  Iter_pidTrk begin_pidTrk() const { return Iter_pidTrk(mTrkData, false); }
  Iter_pidTrk end_pidTrk() const { return Iter_pidTrk(mTrkData, true); }

  Iter_CIDDig begin_CIDDig() const { return Iter_CIDDig(&mDigStart, mDigData, false); }
  Iter_CIDDig end_CIDDig() const { return Iter_CIDDig(&mDigStart, mDigData, true); }

  Iter_ROBDig begin_ROBDig() const { return Iter_ROBDig(mDigData, false); }
  Iter_ROBDig end_ROBDig() const { return Iter_ROBDig(mDigData, true); }

  Iter_MCMDig begin_MCMDig() const { return Iter_MCMDig(mDigData, false); }
  Iter_MCMDig end_MCMDig() const { return Iter_MCMDig(mDigData, true); }

  Iter_chanDig begin_chanDig() const { return Iter_chanDig(mDigData, false); }
  Iter_chanDig end_chanDig() const { return Iter_chanDig(mDigData, true); }

  Iter_ADCDig begin_ADCDig() const { return Iter_ADCDig(mDigData, false); }
  Iter_ADCDig end_ADCDig() const { return Iter_ADCDig(mDigData, true); }

 private:
  const gsl::span<const o2::trd::TriggerRecord> mTrigRec;
  const gsl::span<const o2::trd::Tracklet64> mTrkData;
  const gsl::span<const o2::trd::Digit> mDigData;
  std::vector<bool> mTrkStart;
  std::vector<bool> mDigStart;
};

} // namespace trd
} // namespace o2

#endif
