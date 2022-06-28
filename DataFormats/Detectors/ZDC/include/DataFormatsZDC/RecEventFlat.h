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
#include <gsl/span>
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

struct RecEventFlat { // NOLINT: false positive in clang-tidy !!
  o2::InteractionRecord ir;
  uint32_t channels = 0;                         /// pattern of channels acquired
  uint32_t ezdcDecoded = 0;                      /// pattern of decoded energies
  uint32_t triggers = 0;                         /// pattern of channels with autotrigger bit
  std::map<uint8_t, float> ezdc;                 /// signal in ZDCs
  std::vector<float> TDCVal[NTDCChannels];       /// TDC values
  std::vector<float> TDCAmp[NTDCChannels];       /// TDC signal amplitudes
  std::vector<bool> TDCPile[NTDCChannels];       /// TDC pile-up correction flag (TODO)
  gsl::span<const o2::zdc::BCRecData> mRecBC;    //! Interaction record and references to data
  gsl::span<const o2::zdc::ZDCEnergy> mEnergy;   //! ZDC energy
  gsl::span<const o2::zdc::ZDCTDCData> mTDCData; //! ZDC TDC
  gsl::span<const uint16_t> mInfo;               //! Event quality information
  std::vector<uint16_t> mDecodedInfo;            //! Event quality information (decoded)
  uint64_t mEntry = 0;                           //! Current entry
  uint64_t mNEntries = 0;                        //! Number of entries
  FirstEntry mFirstE = 0;                        //! First energy
  FirstEntry mFirstT = 0;                        //! First TDC
  FirstEntry mFirstI = 0;                        //! First info
  FirstEntry mStopE = 0;                         //! Last + 1 energy
  FirstEntry mStopT = 0;                         //! Last + 1 TDC
  FirstEntry mStopI = 0;                         //! Last + 1 info
  NElem mNE = 0;                                 //! N energy
  NElem mNT = 0;                                 //! N TDC
  NElem mNI = 0;                                 //! N info
  std::array<bool, NChannels> isBeg{};           //! Beginning of sequence
  std::array<bool, NChannels> isEnd{};           //! End of sequence
  BCRecData mCurB;                               //! Current BC
  std::vector<float> inter[NChannels];           //! Interpolated samples

  // Reconstruction messages
  std::array<bool, NChannels> genericE{};       ///  0 Generic error
  std::array<bool, NChannels> tdcPedEv{};       /// -- Event pedestal for TDC
  std::array<bool, NChannels> tdcPedOr{};       /// -- Orbit pedestal for TDC
  std::array<bool, NChannels> tdcPedQC{};       ///  1 QC pedestal for TDC
  std::array<bool, NChannels> tdcPedMissing{};  ///  2 Missing pedestal for ADC
  std::array<bool, NChannels> adcPedEv{};       /// -- Event pedestal for ADC
  std::array<bool, NChannels> adcPedOr{};       ///  3 Orbit pedestal for ADC
  std::array<bool, NChannels> adcPedQC{};       ///  4 QC pedestal for ADC
  std::array<bool, NChannels> adcPedMissing{};  ///  5 Missing pedestal for ADC
  std::array<bool, NChannels> offPed{};         ///  6 Anomalous offset from pedestal info
  std::array<bool, NChannels> pilePed{};        ///  7 Pile-up detection from pedestal info
  std::array<bool, NChannels> pileTM{};         ///  8 Pile-up detection from TM trigger bit
  std::array<bool, NChannels> adcMissingwTDC{}; ///  9 Missing ADC even if TDC is present
  std::array<bool, NChannels> tdcPileEvC{};     /// 10 TDC in-bunch pile-up corrected
  std::array<bool, NChannels> tdcPileEvE{};     /// 11 TDC in-bunch pile-up error
  std::array<bool, NChannels> tdcPileM1C{};     /// 12 TDC pile-up in bunch -1 corrected
  std::array<bool, NChannels> tdcPileM1E{};     /// 13 TDC pile-up in bunch -1 error
  std::array<bool, NChannels> tdcPileM2C{};     /// 14 TDC pile-up in bunch -2 corrected
  std::array<bool, NChannels> tdcPileM2E{};     /// 15 TDC pile-up in bunch -2 error
  std::array<bool, NChannels> tdcPileM3C{};     /// 16 TDC pile-up in bunch -3 corrected
  std::array<bool, NChannels> tdcPileM3E{};     /// 17 TDC pile-up in bunch -3 error
  std::array<bool, NChannels> tdcSigE{};        /// 18 Missing TDC signal correction
  // End_of_messages

  void clearBitmaps();

  uint8_t mVerbosity = DbgZero; //! Verbosity level
  uint32_t mTriggerMask = 0;    //! Trigger mask for printout

  void init(const std::vector<o2::zdc::BCRecData>* RecBC, const std::vector<o2::zdc::ZDCEnergy>* Energy, const std::vector<o2::zdc::ZDCTDCData>* TDCData, const std::vector<uint16_t>* Info);
  void init(const gsl::span<const o2::zdc::BCRecData> RecBC, const gsl::span<const o2::zdc::ZDCEnergy> Energy, const gsl::span<const o2::zdc::ZDCTDCData> TDCData, const gsl::span<const uint16_t> Info);

  int next();
  int at(int ientry);

  void allocate(int isig)
  {
    if (inter[isig].size() != NIS) {
      inter[isig].resize(NIS);
    }
  }

  BCRecData& getCurB()
  {
    return mCurB;
  }

  inline int getEntries() const
  {
    return mNEntries;
  }

  inline int getNextEntry() const
  {
    return mEntry;
  }

  inline NElem getNEnergy() const
  {
    return mNE;
  }

  inline bool getEnergy(int32_t i, uint8_t& key, float& val) const
  {
    if (i < mNE) {
      auto it = ezdc.begin();
      std::advance(it, i);
      key = it->first;
      val = it->second;
      return true;
    }
    return false;
  }

  inline NElem getNTDC() const
  {
    return mNT;
  }

  inline NElem getNInfo() const
  {
    return mNI;
  }

  const std::vector<uint16_t>& getDecodedInfo()
  {
    return mDecodedInfo;
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
  void printDecodedMessages() const;
  ClassDefNV(RecEventFlat, 1);
};

} // namespace zdc
} // namespace o2

#endif
