// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RawPixelDecoder.h
/// \brief Definition of the Alpide pixel reader for raw data processing
#ifndef ALICEO2_ITSMFT_RAWPIXELDECODER_H_
#define ALICEO2_ITSMFT_RAWPIXELDECODER_H_

#include <array>
#include <TStopwatch.h>
#include "Framework/Logger.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Headers/RAWDataHeader.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSMFTReconstruction/GBTLink.h"
#include "ITSMFTReconstruction/RUDecodeData.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/PixelData.h"

namespace o2
{
namespace framework
{
class InputRecord;
}

namespace itsmft
{
class ChipPixelData;

template <class Mapping>
class RawPixelDecoder final : public PixelReader
{
  using RDH = o2::header::RAWDataHeader;

 public:
  RawPixelDecoder();
  ~RawPixelDecoder() final = default;

  GBTLink::Format getFormat() const { return mFormat; }
  void setFormat(GBTLink::Format f);

  void init() final {}
  bool getNextChipData(ChipPixelData& chipData) final;
  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) final;

  void startNewTF(o2::framework::InputRecord& inputs);

  int decodeNextTrigger() final;
  int decodeNextTrigger(int il);

  template <class DigitContainer, class ROFContainer>
  int fillDecodedDigits(DigitContainer& digits, ROFContainer& rofs);
  
  template <class DigitContainer, class ROFContainer>
  int fillDecodedDigitsHW(DigitContainer& digits, ROFContainer& rofs);


  const RUDecodeData* getRUDecode(int ruSW) const { return mRUEntry[ruSW] < 0 ? nullptr : &mRUDecodeVec[mRUEntry[ruSW]]; }
  const GBTLink* getGBTLink(int i) const { return i < 0 ? nullptr : &mGBTLinks[i]; }
  int getNLinks() const { return mGBTLinks.size(); }

  auto getUserDataOrigin() const { return mUserDataOrigin; }
  void setUserDataOrigin(header::DataOrigin orig) { mUserDataOrigin = orig; }

  auto getUserDataDescription() const { return mUserDataDescription; }
  void setUserDataDescription(header::DataDescription desc) { mUserDataDescription = desc; }

  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }

  void setVerbosity(int v);
  int getVerbosity() const { return mVerbosity; }

  void printReport(bool decstat = false, bool skipEmpty = true) const;

  void clearStat();

  TStopwatch& getTimerTFStart() { return mTimerTFStart; }
  TStopwatch& getTimerDecode() { return mTimerDecode; }
  TStopwatch& getTimerExtract() { return mTimerFetchData; }
  uint32_t getNChipsFiredROF() const { return mNChipsFiredROF; }
  uint32_t getNPixelsFiredROF() const { return mNPixelsFiredROF; }
  size_t getNChipsFired() const { return mNChipsFired; }
  size_t getNPixelsFired() const { return mNPixelsFired; }

  struct LinkEntry {
    int entry = -1;
  };

 private:
  void setupLinks(o2::framework::InputRecord& inputs);
  int getRUEntrySW(int ruSW) const { return mRUEntry[ruSW]; }
  RUDecodeData* getRUDecode(int ruSW) { return &mRUDecodeVec[mRUEntry[ruSW]]; }
  GBTLink* getGBTLink(int i) { return i < 0 ? nullptr : &mGBTLinks[i]; }
  RUDecodeData& getCreateRUDecode(int ruSW);

  static constexpr uint16_t NORUDECODED = 0xffff; // this must be > than max N RUs

  std::vector<GBTLink> mGBTLinks;                           // active links pool
  std::unordered_map<uint32_t, LinkEntry> mSubsSpec2LinkID; // link subspec to link entry in the pool mapping
  std::vector<RUDecodeData> mRUDecodeVec;                   // set of active RUs
  std::array<short, Mapping::getNRUs()> mRUEntry;           // entry of the RU with given SW ID in the mRUDecodeVec
  std::string mSelfName;                        // self name
  header::DataOrigin mUserDataOrigin = o2::header::gDataOriginInvalid; // alternative user-provided data origin to pick
  header::DataDescription mUserDataDescription = o2::header::gDataDescriptionInvalid; // alternative user-provided description to pick
  uint16_t mCurRUDecodeID = NORUDECODED;        // index of currently processed RUDecode container
  int mLastReadChipID = -1;                     // chip ID returned by previous getNextChipData call, used for ordering checks
  Mapping mMAP;                                 // chip mapping
  int mVerbosity = 0;
  int mNThreads = 1; // number of decoding threads
  GBTLink::Format mFormat = GBTLink::NewFormat; // ITS Data Format (old: 1 ROF per CRU page)
  // statistics
  o2::itsmft::ROFRecord::ROFtype mROFCounter = 0; // RSTODO is this needed? eliminate from ROFRecord ?
  uint32_t mNChipsFiredROF = 0;                   // counter within the ROF
  uint32_t mNPixelsFiredROF = 0;                  // counter within the ROF
  uint32_t mNLinksDone = 0;                       // number of links reached end of data
  size_t mNChipsFired = 0;                        // global counter
  size_t mNPixelsFired = 0;                       // global counter
  TStopwatch mTimerTFStart;
  TStopwatch mTimerDecode;
  TStopwatch mTimerFetchData;

  ClassDefOverride(RawPixelDecoder, 1);
};

///______________________________________________________________
/// Fill decoded digits to global vector
template <class Mapping>
template <class DigitContainer, class ROFContainer>
int RawPixelDecoder<Mapping>::fillDecodedDigits(DigitContainer& digits, ROFContainer& rofs)
{
  if (mInteractionRecord.isDummy()) {
    return 0; // nothing was decoded
  }
  mTimerFetchData.Start(false);
  int ref = digits.size();
  for (unsigned int iru = 0; iru < mRUDecodeVec.size(); iru++) {
    for (int ic = 0; ic < mRUDecodeVec[iru].nChipsFired; ic++) {
      const auto& chip = mRUDecodeVec[iru].chipsData[ic];
      for (const auto& hit : mRUDecodeVec[iru].chipsData[ic].getData()) {
                digits.emplace_back(chip.getChipID(), hit.getRow(), hit.getCol());
      }
    }
  }
  int nFilled = digits.size() - ref;
  rofs.emplace_back(mInteractionRecord, mROFCounter, ref, nFilled);
  mTimerFetchData.Stop();
  return nFilled;
}

template <class Mapping>
template <class DigitContainer, class ROFContainer>
int RawPixelDecoder<Mapping>::fillDecodedDigitsHW(DigitContainer& digits, ROFContainer& rofs)
{
  if (mInteractionRecord.isDummy()) {
    return 0; // nothing was decoded
  }
  mTimerFetchData.Start(false);
  int ref = digits.size();
  for (unsigned int iru = 0; iru < mRUDecodeVec.size(); iru++) {
    uint16_t ninjection = mRUDecodeVec[iru].nInj;
    uint16_t chargeinjected = mRUDecodeVec[iru].chargeInj;
    uint16_t feeID = mMAP.RUSW2FEEId(mRUDecodeVec[iru].ruSWID,0);
    uint16_t half = (feeID >> 6) & 0x1;
    uint16_t disk = (feeID >> 3) & 0x7;
    uint16_t plane = (feeID >> 2) & 0x1;
    uint16_t zone = feeID & 0x3;
    for (int ic = 0; ic < mRUDecodeVec[iru].nChipsFired; ic++) {
      const auto& chip = mRUDecodeVec[iru].chipsData[ic];
      for (const auto& hit : mRUDecodeVec[iru].chipsData[ic].getData()) {
                digits.emplace_back(ninjection, chargeinjected, half, disk, plane, zone, chip.getCableHW(), chip.getChipID(), hit.getRow(), hit.getCol());

      }
    }
  }
  int nFilled = digits.size() - ref;
  rofs.emplace_back(mInteractionRecord, mROFCounter, ref, nFilled);
  mTimerFetchData.Stop();
  return nFilled;
}
  

using RawDecoderITS = RawPixelDecoder<ChipMappingITS>;
using RawDecoderMFT = RawPixelDecoder<ChipMappingMFT>;

} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_RAWPIXELDECODER_H */
