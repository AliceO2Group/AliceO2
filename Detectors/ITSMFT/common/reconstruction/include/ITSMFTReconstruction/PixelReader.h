// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PixelReader.h
/// \brief Definition of the ITS pixel reader
#ifndef ALICEO2_ITSMFT_PIXELREADER_H
#define ALICEO2_ITSMFT_PIXELREADER_H

#include <Rtypes.h>
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>

namespace o2
{
namespace ITSMFT
{
/// \class PixelReader
/// \brief PixelReader class for the ITSMFT
///
class PixelReader
{
  using Label = o2::MCCompLabel;

 public:
  /// Transient data for single fired pixel

  PixelReader() = default;
  virtual ~PixelReader() = default;
  PixelReader(const PixelReader& cluster) = delete;

  PixelReader& operator=(const PixelReader& src) = delete;

  virtual void init() = 0;
  virtual bool getNextChipData(ChipPixelData& chipData) = 0;
  virtual ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) = 0;
  //
 protected:
  //
  ClassDef(PixelReader, 1);
};

/// \class DigitPixelReader
/// \brief DigitPixelReader class for the ITS. Feeds the MC digits to the Cluster Finder
///
class DigitPixelReader : public PixelReader
{
 public:
  DigitPixelReader() = default;
  ~DigitPixelReader() override = default;
  void setDigitArray(const std::vector<o2::ITSMFT::Digit>* a)
  {
    mDigitArray = a;
    mIdx = 0;
  }

  void init() override
  {
    mIdx = 0;
    mLastDigit = nullptr;
  }

  bool getNextChipData(ChipPixelData& chipData) override;
  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) override;

 private:
  void addPixel(ChipPixelData& chipData, const Digit* dig)
  {
    // add new fired pixel
    chipData.getData().emplace_back(dig);
  }

  const std::vector<o2::ITSMFT::Digit>* mDigitArray = nullptr;
  const Digit* mLastDigit = nullptr;
  Int_t mIdx = 0;

  ClassDefOverride(DigitPixelReader, 1);
};

/// \class RawPixelReader
/// \brief RawPixelReader class for the ITS. Feeds raw data to the Cluster Finder
///
class RawPixelReader : public PixelReader
{
 public:
  RawPixelReader() = default;
  ~RawPixelReader() override = default;
  bool getNextChipData(ChipPixelData& chipData) override;
  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) override;

  ClassDefOverride(RawPixelReader, 1);
};

} // namespace ITSMFT
} // namespace o2

#endif /* ALICEO2_ITS_PIXELREADER_H */
