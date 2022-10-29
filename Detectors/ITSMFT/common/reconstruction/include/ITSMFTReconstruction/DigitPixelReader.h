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

/// \file DigitPixelReader.h
/// \brief Definition of the Alpide pixel reader for MC digits processing

#ifndef ALICEO2_ITSMFT_DIGITPIXELREADER_H
#define ALICEO2_ITSMFT_DIGITPIXELREADER_H

#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TTree.h>
#include <vector>
#include <memory>
#include <gsl/span>

namespace o2
{
namespace itsmft
{

class DigitPixelReader : public PixelReader
{
 public:
  DigitPixelReader() = default;
  ~DigitPixelReader() override;

  const auto getMC2ROFRecords() const
  {
    return mMC2ROFRecVec;
  }

  void setMC2ROFRecords(const gsl::span<const o2::itsmft::MC2ROFRecord> a)
  {
    mMC2ROFRecVec = a;
  }

  void setROFRecords(const gsl::span<const o2::itsmft::ROFRecord> a)
  {
    mROFRecVec = a;
    mIdROF = -1;
  }

  void setDigits(const gsl::span<const o2::itsmft::Digit> a)
  {
    mDigits = a;
    mIdDig = 0;
    if (mSquashOverflowsDepth) {
      mSquashedDigitsMask.resize(a.size(), false);
      mBookmarkNextROFs.resize(mSquashOverflowsDepth, 0);
      mIdROFLast = 0;
    }
  }

  void setDigitsMCTruth(const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* m)
  {
    mDigitsMCTruth = m;
  }

  const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* getDigitsMCTruth() const override
  {
    return mDigitsMCTruth;
  }

  bool getNextChipData(ChipPixelData& chipData) override;
  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) override;

  void init() override
  {
    mIdDig = 0;
    mIdROF = -1;
  }

  // prepare next trigger
  int decodeNextTrigger() override;

  // methods for standalone reading
  void openInput(const std::string rawInput, o2::detectors::DetID det);
  bool readNextEntry();

  void clear();

  uint16_t getSquashingDepth() { return mSquashOverflowsDepth; }
  void setSquashingDepth(const int16_t v)
  {
    mSquashOverflowsDepth = v;
  }

  uint16_t getSquashingDist() { return mMaxSquashDist; }
  void setSquashingDist(const int16_t v)
  {
    mMaxSquashDist = v;
  }

  int getMaxBCSeparationToSquash() const { return mMaxBCSeparationToSquash; }
  void setMaxBCSeparationToSquash(int n)
  {
    mMaxBCSeparationToSquash = n;
  }

 private:
  void addPixel(ChipPixelData& chipData, const Digit* dig)
  {
    // add new fired pixel
    chipData.getData().emplace_back(dig);
  }

  // pointer for input containers in the self-managed mode: due to the requirements of the
  // fairroot the externally settable pointers must be const...
  std::vector<o2::itsmft::Digit>* mDigitsSelf = nullptr;
  std::vector<o2::itsmft::ROFRecord>* mROFRecVecSelf = nullptr;
  std::vector<o2::itsmft::MC2ROFRecord>* mMC2ROFRecVecSelf = nullptr;
  const o2::dataformats::IOMCTruthContainerView* mDigitsMCTruthSelf = nullptr;

  gsl::span<const o2::itsmft::Digit> mDigits;
  gsl::span<const o2::itsmft::ROFRecord> mROFRecVec;
  gsl::span<const o2::itsmft::MC2ROFRecord> mMC2ROFRecVec;

  const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* mDigitsMCTruth = nullptr;
  Int_t mIdDig = 0; // Digits slot read within ROF
  Int_t mIdROF = 0; // ROFRecord being red

  std::unique_ptr<TTree> mInputTree; // input tree for digits

  // Squashing datamembers
  int16_t mSquashOverflowsDepth = 0;     // merge overflow pixels in next N ROF(s) into first ROF and mask them
  std::vector<bool> mSquashedDigitsMask; // keep info of squashed pixels
  uint16_t mMaxSquashDist = 1;           // maximum pixel distance to allow for squashing
  int mMaxBCSeparationToSquash;          // frames can be separated by less than this amount of BCs
  std::vector<int> mBookmarkNextROFs;    // keep track of the position of the last processed digits in next rof
  int mIdROFLast = 0;                    // ROFRecord red last iteration

  ClassDefOverride(DigitPixelReader, 1);
};

} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITS_DIGITPIXELREADER_H */