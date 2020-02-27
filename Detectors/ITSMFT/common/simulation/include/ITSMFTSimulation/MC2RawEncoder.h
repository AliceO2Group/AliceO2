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
#include "DetectorsRaw/RawFileWriter.h"

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

  ~MC2RawEncoder()
  {
    mWriter.close();
  }

  void digits2raw(gsl::span<const Digit> digits, const o2::InteractionRecord& bcData);
  void finalize();
  void init();

  RUDecodeData& getCreateRUDecode(int ruSW);

  RUDecodeData* getRUDecode(int ruSW) { return mRUEntry[ruSW] < 0 ? nullptr : &mRUDecodeVec[mRUEntry[ruSW]]; }

  void setVerbosity(int v)
  {
    mVerbosity = v;
    mWriter.setVerbosity(v);
  }
  int getVerbosity() const { return mVerbosity; }

  Mapping& getMapping() { return mMAP; }

  void setMinMaxRUSW(uint8_t ruMin, uint8_t ruMax)
  {
    mRUSWMax = (ruMax < uint8_t(mMAP.getNRUs())) ? ruMax : mMAP.getNRUs() - 1;
    mRUSWMin = ruMin < mRUSWMax ? ruMin : mRUSWMax;
  }

  int getRUSWMin() const { return mRUSWMin; }
  int getRUSWMax() const { return mRUSWMax; }

  void setContinuousReadout(bool v) { mROMode = v ? Continuous : Triggered; }
  bool isContinuousReadout() const { return mROMode == Continuous; }

  o2::raw::RawFileWriter& getWriter() { return mWriter; }

  std::string getDefaultSinkName() const { return mDefaultSinkName; }
  void setDefaultSinkName(const std::string& nm)
  {
    if (!nm.empty()) {
      mDefaultSinkName = nm;
    }
  }

  int carryOverMethod(const RDH& rdh, const gsl::span<char> data, const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const;

 private:
  void convertEmptyChips(int fromChip, int uptoChip, RUDecodeData& ru);
  void convertChip(ChipPixelData& chipData, RUDecodeData& ru);
  void fillGBTLinks(RUDecodeData& ru);

  enum RoMode_t { NotSet,
                  Continuous,
                  Triggered };
  o2::InteractionRecord mCurrIR;               // currently processed int record
  o2::raw::RawFileWriter mWriter;
  std::string mDefaultSinkName = "dataSink.raw";
  Mapping mMAP;
  Coder mCoder;
  int mVerbosity = 0;                                        //! verbosity level
  uint8_t mRUSWMin = 0;                                      ///< min RU (SW) to convert
  uint8_t mRUSWMax = 0xff;                                   ///< max RU (SW) to convert
  int mNRUs = 0;                                             /// total number of RUs seen
  int mNLinks = 0;                                           /// total number of GBT links seen
  std::array<RUDecodeData, Mapping::getNRUs()> mRUDecodeVec; /// decoding buffers for all active RUs
  std::array<int, Mapping::getNRUs()> mRUEntry;              /// entry of the RU with given SW ID in the mRUDecodeVec
  RoMode_t mROMode = NotSet;

  ClassDefNV(MC2RawEncoder, 1);
};

// template specifications
using MC2RawEncoderITS = MC2RawEncoder<ChipMappingITS>;
using MC2RawEncoderMFT = MC2RawEncoder<ChipMappingMFT>;

} // namespace itsmft
} // namespace o2

#endif
