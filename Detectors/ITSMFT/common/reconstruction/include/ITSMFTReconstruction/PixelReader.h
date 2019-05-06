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
/// \brief Abstract class for Alpide data reader class

#ifndef ALICEO2_ITSMFT_PIXELREADER_H
#define ALICEO2_ITSMFT_PIXELREADER_H

#include <Rtypes.h>
#include "ITSMFTReconstruction/PixelData.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <vector>

namespace o2
{

namespace itsmft
{
/// \class PixelReader
/// \brief PixelReader class for the ITSMFT
///
class PixelReader
{
 public:
  /// Transient data for single fired pixel

  PixelReader() = default;
  virtual ~PixelReader() = default;
  PixelReader(const PixelReader& cluster) = delete;

  PixelReader& operator=(const PixelReader& src) = delete;
  virtual void init() = 0;
  virtual bool getNextChipData(ChipPixelData& chipData) = 0;
  virtual ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) = 0;
  virtual const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* getDigitsMCTruth() const
  {
    return nullptr;
  }
  const o2::InteractionRecord& getInteractionRecordHB() const
  {
    return mInteractionRecordHB;
  }
  const o2::InteractionRecord& getInteractionRecord() const
  {
    return mInteractionRecord;
  }
  uint32_t getTrigger() const
  {
    return mTrigger;
  }
  //
 protected:
  //
  o2::InteractionRecord mInteractionRecordHB = {}; // interation record for the HB
  o2::InteractionRecord mInteractionRecord = {};   // interation record for the trigger
  uint32_t mTrigger = 0;

  ClassDef(PixelReader, 1);
};

} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITS_PIXELREADER_H */
