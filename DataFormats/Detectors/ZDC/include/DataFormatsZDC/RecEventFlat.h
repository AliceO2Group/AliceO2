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

#ifndef _ZDC_RECEVENTFLAT_H_
#define _ZDC_RECEVENTFLAT_H_

#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsZDC/BCRecData.h"
#include "DataFormatsZDC/ZDCEnergy.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "ZDCBase/Constants.h"
#include "MathUtils/Cartesian.h"
#include <Rtypes.h>
#include <array>
#include <vector>
#include <map>

/// \file RecEventFlat.h
/// \brief Class to decode the reconstructed ZDC event (single BC with signal in one of detectors)
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{
using FirstEntry = int;
using NElem = int;

struct RecEventFlat { //NOLINT: false positive in clang-tidy !!
  o2::InteractionRecord ir;
  uint32_t channels = 0;                           /// pattern of channels acquired
  uint32_t triggers = 0;                           /// pattern of channels with autotrigger bit
  std::map<uint8_t, float> ezdc;                   /// signal in ZDCs
  std::vector<int16_t> TDCVal[NTDCChannels];       /// TdcChannels
  std::vector<int16_t> TDCAmp[NTDCChannels];       /// TdcAmplitudes
  std::vector<o2::zdc::BCRecData>* mRecBC;         //! Interaction record and references to data
  std::vector<o2::zdc::ZDCEnergy>* mEnergy;        //! ZDC energy
  std::vector<o2::zdc::ZDCTDCData>* mTDCData;      //! ZDC TDC
  std::vector<uint16_t>* mInfo;                    //! Event quality information
  uint64_t mEntry = 0;                             //! Current entry
  uint64_t mNEntries = 0;                          //! Number of entries
  FirstEntry mFirstE = 0;                          //! First energy
  FirstEntry mFirstT = 0;                          //! First TDC
  FirstEntry mFirstI = 0;                          //! First info
  FirstEntry mStopE = 0;                           //! Last + 1 energy
  FirstEntry mStopT = 0;                           //! Last + 1 TDC
  FirstEntry mStopI = 0;                           //! Last + 1 info
  NElem mNE = 0;                                   //! N energy
  NElem mNT = 0;                                   //! N TDC
  NElem mNI = 0;                                   //! N info
  o2::zdc::BCRecData mCurB;                        //! Current BC
  std::array<bool, NChannels> tdcPedEv = {0};      /// Event pedestal for TDC
  std::array<bool, NChannels> tdcPedOr = {0};      /// Orbit pedestal for TDC
  std::array<bool, NChannels> tdcPedQC = {0};      /// QC pedestal for TDC
  std::array<bool, NChannels> tdcPedMissing = {0}; /// Missing pedestal for ADC
  std::array<bool, NChannels> adcPedEv = {0};      /// Event pedestal for ADC
  std::array<bool, NChannels> adcPedOr = {0};      /// Orbit pedestal for ADC
  std::array<bool, NChannels> adcPedQC = {0};      /// QC pedestal for ADC
  std::array<bool, NChannels> adcPedMissing = {0}; /// Missing pedestal for ADC
  uint8_t mVerbosity = DbgZero;                    //! Verbosity level
  uint32_t mTriggerMask = 0;                       //! Trigger mask for printout

  void init(std::vector<o2::zdc::BCRecData>* RecBC, std::vector<o2::zdc::ZDCEnergy>* Energy, std::vector<o2::zdc::ZDCTDCData>* TDCData, std::vector<uint16_t>* Info);

  int next();

  inline NElem getNEnergy() const
  {
    return mNE;
  }

  inline NElem getNTDC() const
  {
    return mNT;
  }

  inline NElem getNInfo() const
  {
    return mNI;
  }

  float tdcV(uint8_t ich, uint64_t ipos) const
  {
    if (ich < NTDCChannels) {
      if (ipos < TDCVal[ich].size()) {
        return FTDCVal * TDCVal[ich][ipos];
      }
    }
    return -std::numeric_limits<float>::infinity();
  }

  float tdcA(uint8_t ich, uint64_t ipos) const
  {
    if (ich < NTDCChannels) {
      if (ipos < TDCAmp[ich].size()) {
        return FTDCAmp * TDCAmp[ich][ipos];
      }
    }
    return -std::numeric_limits<float>::infinity();
  }

  int NtdcV(uint8_t ich) const
  {
    if (ich < NTDCChannels) {
      return TDCVal[ich].size();
    } else {
      return 0;
    }
  }

  int NtdcA(uint8_t ich) const
  {
    if (ich < NTDCChannels) {
      return TDCAmp[ich].size();
    } else {
      return 0;
    }
  }

  float EZDC(uint8_t ich) const
  {
    std::map<uint8_t, float>::const_iterator it = ezdc.find(ich);
    if (it != ezdc.end()) {
      return it->second;
    } else {
      return -std::numeric_limits<float>::infinity();
    }
  }

  float EZNAC() const { return EZDC(IdZNAC); }
  float EZNA1() const { return EZDC(IdZNA1); }
  float EZNA2() const { return EZDC(IdZNA2); }
  float EZNA3() const { return EZDC(IdZNA3); }
  float EZNA4() const { return EZDC(IdZNA4); }
  float EZNASum() const { return EZDC(IdZNASum); }

  float EZPAC() const { return EZDC(IdZPAC); }
  float EZPA1() const { return EZDC(IdZPA1); }
  float EZPA2() const { return EZDC(IdZPA2); }
  float EZPA3() const { return EZDC(IdZPA3); }
  float EZPA4() const { return EZDC(IdZPA4); }
  float EZPASum() const { return EZDC(IdZPASum); }

  float EZEM1() const { return EZDC(IdZEM1); }
  float EZEM2() const { return EZDC(IdZEM2); }

  float EZNCC() const { return EZDC(IdZNCC); }
  float EZNC1() const { return EZDC(IdZNC1); }
  float EZNC2() const { return EZDC(IdZNC2); }
  float EZNC3() const { return EZDC(IdZNC3); }
  float EZNC4() const { return EZDC(IdZNC4); }
  float EZNCSum() const { return EZDC(IdZNCSum); }

  float EZPCC() const { return EZDC(IdZPCC); }
  float EZPC1() const { return EZDC(IdZPC1); }
  float EZPC2() const { return EZDC(IdZPC2); }
  float EZPC3() const { return EZDC(IdZPC3); }
  float EZPC4() const { return EZDC(IdZPC4); }
  float EZPCSum() const { return EZDC(IdZPCSum); }

  void decodeInfo(uint8_t ch, uint16_t code);
  void decodeMapInfo(uint32_t ch, uint16_t code);

  void print() const;
  ClassDefNV(RecEventFlat, 1);
};

} // namespace zdc
} // namespace o2

#endif
