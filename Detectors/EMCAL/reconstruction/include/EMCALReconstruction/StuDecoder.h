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
#ifndef ALICEO2_EMCAL_STUDECODER_H
#define ALICEO2_EMCAL_STUDECODER_H

#include <exception>
#include <iosfwd>
#include <string>
#include "EMCALReconstruction/RawReaderMemory.h"
#include "DataFormatsEMCAL/Constants.h"

namespace o2
{
namespace emcal
{
/// \class StuDecoder
/// \brief Decoder of the EMCAL/DCAL STU data
/// \ingroup EMCALreconstruction
/// \author Martin Poghosyan <martin.poghosyan@cern.ch>, Oak Ridge National Laboratory
/// \since Apr. 27, 2022
///

class StuDecoder
{
 public:
  /// \brief Constructor
  /// \param reader Raw reader instance to be decoded
  StuDecoder(RawReaderMemory& reader);

  /// \brief Destructor
  ~StuDecoder() = default;

  /// \brief Decode the STU stream
  /// \throw StuDecoderError if the STU payload cannot be decoded
  void decode();

  int32_t getL1GammaHighThreshold() const { return mL1GammaHighThreshold; }
  int32_t getL1JetHighThreshold() const { return mL1JetHighThreshold; }
  int32_t getL1GammaLowThreshold() const { return mL1GammaLowThreshold; }
  int32_t getL1JetLowThreshold() const { return mL1JetLowThreshold; }
  int32_t getRho() const { return (mCFGword13 & 0x3FFFF); }
  int32_t getFrameReceivedSTU() const { return ((mCFGword13 >> 18) & 0x3); }
  int32_t getRegionEnable() const { return mRegionEnable; }
  int32_t getFrameReceived() const { return mFrameReceived; }
  int32_t getParchSize() const { return ((mCFGword16 >> 16) & 0xFFFF); }
  int32_t getFWversion() const { return (mCFGword16 & 0xFFFF); }

  STUtype_t getSTUtype() const { return mSTU; }
  int getFeeID() const { return STUparam::FeeID[mSTU]; }
  int getNumberOfTRUs() const { return STUparam::NTRU[mSTU]; }

  std::vector<int16_t> getL1JetHighPatchIndices() const { return mL1JetHighPatchIndex; }
  std::vector<int16_t> getL1JetLowPatchIndices() const { return mL1JetLowPatchIndex; }
  std::vector<int16_t> getL1GammaHighPatchIndices() const { return mL1GammaHighPatchIndex; }
  std::vector<int16_t> getL1GammaLowPatchIndics() const { return mL1GammaLowPatchIndex; }
  std::vector<int16_t> getFastOrADCs() const { return mFastOrADC; }

  int getNumberOfL1JetHighPatches() const { return mL1JetHighPatchIndex.size(); }
  int getNumberOfL1JetLowPatches() const { return mL1JetLowPatchIndex.size(); }
  int getNumberOfL1GammaHighPatches() const { return mL1GammaHighPatchIndex.size(); }
  int getNumberOfL1GammaLowPatches() const { return mL1GammaLowPatchIndex.size(); }

  int16_t getIndexOfL1JetHighPatch(int id) const { return mL1JetHighPatchIndex[id]; }
  int16_t getIndexOfL1JetLowPatch(int id) const { return mL1JetLowPatchIndex[id]; }
  int16_t getIndexOfL1GammaHighPatch(int id) const { return mL1GammaHighPatchIndex[id]; }
  int16_t getIndexOfL1GammaLowPatch(int id) const { return mL1GammaLowPatchIndex[id]; }

  int16_t getFastOrADC(int iTRU, int iCh) const { return mFastOrADC[iTRU + getNumberOfTRUs() * iCh]; }

  std::tuple<int, int, int> getL1GammaMaxPatch() const // std::tuple<TRUid,x,y>
  {
    return std::make_tuple(((mCFGWord0 >> 9) & 0x1F), ((mCFGWord0 >> 4) & 0x1F), (mCFGWord0 & 0xF));
  }

  bool isFullPayload() const { return mIsFullPayload; }
  bool isL1GammaLowFired() const { return ((mCFGWord0 >> 16) & 0x1); }
  bool isL1GammaHighFired() const { return ((mCFGWord0 >> 17) & 0x1); }
  bool isL1JetLowFired() const { return ((mCFGWord0 >> 18) & 0x1); }
  bool isL1JetHighFired() const { return ((mCFGWord0 >> 19) & 0x1); }
  bool isMedianMode() const { return ((mCFGWord0 >> 20) & 0x1); }

  void dumpSTUcfg() const;

  int mDebug = -3;

 private:
  RawReaderMemory& mRawReader; ///< underlying raw reader

  std::vector<int16_t> mL1JetHighPatchIndex;
  std::vector<int16_t> mL1JetLowPatchIndex;
  std::vector<int16_t> mL1GammaHighPatchIndex;
  std::vector<int16_t> mL1GammaLowPatchIndex;
  std::vector<int16_t> mFastOrADC;

  // data from payload
  int32_t mCFGWord0 = 0;             ///<
  int32_t mCFGWord1 = 0;             ///<
  int32_t mL0mask = 0;               ///<
  int32_t mL1GammaHighThreshold = 0; ///<
  int32_t mShortPayloadRate = 0;     ///<
  int32_t mL0bits = 0;               ///<
  int32_t mL1JetHighThreshold = 0;   ///<
  int32_t mL1GammaLowThreshold = 0;  ///<
  int32_t mL1JetLowThreshold = 0;    ///<
  int32_t mCFGword13 = 0;            ///<
  int32_t mRegionEnable = 0;         ///<
  int32_t mFrameReceived = 0;        ///<
  int32_t mCFGword16 = 0;            ///<

  STUtype_t mSTU = STUtype_t::ESTU;
  bool mIsFullPayload = true; ///<

  void init();
  void decodeL1JetPatchIndices(const uint32_t* buffer);
  void decodeL1GammaPatchIndices(const uint32_t* buffer);
  void decodeFastOrADC(const uint32_t* buffer);

  int getCFGWords() { return STUparam::CFG_nWords[mSTU]; }
  int getL1JetIndexWords() { return STUparam::L1JetIndex_nWords[mSTU]; }
  int getL0indexWords() { return STUparam::L0index_nWords[mSTU]; }
  int getL1GammaIndexWords() { return STUparam::L1GammaIndex_nWords[mSTU]; }
  int getRawWords() { return STUparam::Raw_nWords[mSTU]; }
  int getSubregionsEta() { return STUparam::SubregionsEta[mSTU]; }
  int getSubregionsPhi() { return STUparam::SubregionsPhi[mSTU]; }
  int getPaloadSizeFull() { return STUparam::PaloadSizeFull[mSTU]; }
  int getPaloadSizeShort() { return STUparam::PaloadSizeShort[mSTU]; }

  ClassDefNV(StuDecoder, 1);
};

} // namespace emcal
} // namespace o2

#endif
