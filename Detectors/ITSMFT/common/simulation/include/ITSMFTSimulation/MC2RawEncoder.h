// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MC2RawEncoder.h
/// \brief Definition of the ITS/MFT Alpide pixel MC->raw converter

#ifndef ALICEO2_ITSMFT_MC2RAWENCODER_H_
#define ALICEO2_ITSMFT_MC2RAWENCODER_H_

#include <gsl/gsl>                               // for guideline support library; array_view
#include "ITSMFTReconstruction/RawPixelReader.h" // TODO : must be modified
#include "ITSMFTReconstruction/AlpideCoder.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "CommonUtils/HBFUtils.h"

namespace o2
{
namespace itsmft
{

template <class Mapping>
class MC2RawEncoder
{
  using Coder = o2::itsmft::AlpideCoder;
  using RDH = o2::header::RAWDataHeader;

 public:
  MC2RawEncoder()
  {
    mRUEntry.fill(-1);
  }
  ~MC2RawEncoder() = default;

  void digits2raw(gsl::span<const Digit> digits, const o2::InteractionRecord& bcData);
  void finalize();
  void init();

  RUDecodeData& getCreateRUDecode(int ruSW);

  RUDecodeData* getRUDecode(int ruSW) { return mRUEntry[ruSW] < 0 ? nullptr : &mRUDecodeVec[mRUEntry[ruSW]]; }

  void setBCData(const o2::InteractionRecord& bcData);

  /// number of orbits per HBF (max 255)
  void setNOrbitsPerHBF(uint8_t n) { mNOrbitsPerHBF = n; }
  int getNOrbitsPerHBF() const { return mNOrbitsPerHBF; }

  /// nominal size of the superpage in bytes
  void setSuperPageSize(int n)
  {
    constexpr int MinSize = 2 * MaxGBTPacketBytes;
    mSuperPageSize = n < MinSize ? MinSize : n;
  }
  int getSuperPageSize() const { return mSuperPageSize; }

  /// continuous or triggered readout data
  bool isContinuous() const { return mIsContinuous; }
  void setSetContinuous(bool v) { mIsContinuous = v; }

  /// do we treat CRU pages as having max size?
  bool isMaxPageImposed() const { return mImposeMaxPage; }
  /// CRU pages are of max size of 8KB
  void imposeMaxPage(bool v) { mImposeMaxPage = v; }

  /// are we starting new superpage for each TF
  bool isStartTFOnNewSPage(bool v) { return mStartTFOnNewSPage; }
  void setStartTFOnNewSPage(bool v) { mStartTFOnNewSPage = v; }

  void setVerbosity(int v) { mVerbose = v; }
  int getVerbosity() const { return mVerbose; }

  void setOutFile(FILE* outf) { mOutFile = outf; }

  Mapping& getMapping() { return mMAP; }

  void setMinMaxRUSW(uint8_t ruMin, uint8_t ruMax)
  {
    mRUSWMax = (ruMax < uint8_t(mMAP.getNRUs())) ? ruMax : mMAP.getNRUs() - 1;
    mRUSWMin = ruMin < mRUSWMax ? ruMin : mRUSWMax;
  }

  int getRUSWMin() const { return mRUSWMin; }
  int getRUSWMax() const { return mRUSWMax; }

 private:
  void convertEmptyChips(int fromChip, int uptoChip, RUDecodeData& ru);
  void convertChip(ChipPixelData& chipData, RUDecodeData& ru);
  void fillGBTLinks(RUDecodeData& ru);
  void flushLinkSuperPage(GBTLink& link, FILE* outFl = nullptr);
  void openPageLinkHBF(GBTLink& link);
  void addPageLinkHBF(GBTLink& link, bool stop = false);
  void closePageLinkHBF(GBTLink& link) { addPageLinkHBF(link, true); }
  void openHBF();
  void closeHBF();
  void flushAllLinks();

 private:
  o2::utils::HBFUtils mHBFUtils;
  std::vector<o2::InteractionRecord> mHBIRVec; // workspace for HB IR generation
  o2::InteractionRecord mLastIR;               // last IR processed (or 1st in TF if none)
  o2::InteractionRecord mCurrIR;               // currently processed int record
  FILE* mOutFile = nullptr;
  Mapping mMAP;
  Coder mCoder;
  RDH mRDH;                         // current RDH
  int mSuperPageSize = 1024 * 1024; // super page size
  int mHBFCounter = -1;
  int mVerbose = 0;                                          //! verbosity level
  uint8_t mNOrbitsPerHBF = 0xff;                             ///< number of orbitd per HB frame
  uint8_t mRUSWMin = 0;                                      ///< min RU (SW) to convert
  uint8_t mRUSWMax = 0xff;                                   ///< max RU (SW) to convert
  int mNRUs = 0;                                             /// total number of RUs seen
  int mNLinks = 0;                                           /// total number of GBT links seen
  std::array<RUDecodeData, Mapping::getNRUs()> mRUDecodeVec; /// decoding buffers for all active RUs
  std::array<int, Mapping::getNRUs()> mRUEntry;              /// entry of the RU with given SW ID in the mRUDecodeVec
  int mCurRUDecodeID = -1;
  bool mIsContinuous = true;
  bool mImposeMaxPage = true; /// force (pad) all CRU pages to be 8KB
  bool mStartTFOnNewSPage = true;
  ClassDefNV(MC2RawEncoder, 1);
};

// template specifications
using MC2RawEncoderITS = MC2RawEncoder<ChipMappingITS>;
using MC2RawEncoderMFT = MC2RawEncoder<ChipMappingMFT>;

} // namespace itsmft
} // namespace o2

#endif
