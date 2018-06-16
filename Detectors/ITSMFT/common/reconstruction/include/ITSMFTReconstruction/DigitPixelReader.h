// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "ITSMFTBase/Digit.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TTree.h>
#include <vector>
#include <memory>

namespace o2
{
namespace ITSMFT
{

class DigitPixelReader : public PixelReader
{
 public:
  DigitPixelReader() = default;
  ~DigitPixelReader() override;

  void setDigits(const std::vector<o2::ITSMFT::Digit>* a)
  {
    mDigits = a;
    mIdx = 0;
  }

  void setDigitsMCTruth(const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* m)
  {
    mDigitsMCTruth = m;
  }

  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* getDigitsMCTruth() const override
  {
    return mDigitsMCTruth;
  }

  bool getNextChipData(ChipPixelData& chipData) override;
  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) override;

  void init() override
  {
    mLastDigit = nullptr;
    mIdx = 0;
  }

  // methods for standalone reading
  void openInput(const std::string rawInput, o2::detectors::DetID det);
  bool readNextEntry();

  void clear();

 private:
  void addPixel(ChipPixelData& chipData, const Digit* dig)
  {
    // add new fired pixel
    chipData.getData().emplace_back(dig);
  }

  // pointer for input containers in the self-managed mode: due to the requirements of the
  // fairroot the externally settable pointers must be const...
  std::vector<o2::ITSMFT::Digit>* mDigitsSelf = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mDigitsMCTruthSelf = nullptr;

  const std::vector<o2::ITSMFT::Digit>* mDigits = nullptr;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mDigitsMCTruth = nullptr;
  const Digit* mLastDigit = nullptr;
  Int_t mIdx = 0;

  std::unique_ptr<TTree> mInputTree; // input tree for digits

  ClassDefOverride(DigitPixelReader, 1);
};

} // namespace ITSMFT
} // namespace o2

#endif /* ALICEO2_ITS_DIGITPIXELREADER_H */
