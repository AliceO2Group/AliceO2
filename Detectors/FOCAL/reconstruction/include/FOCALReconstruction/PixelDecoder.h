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
#ifndef QC_MODULE_FOCAL_PIXELDECODER_H
#define QC_MODULE_FOCAL_PIXELDECODER_H

#include <memory>
#include <unordered_map>
#include <gsl/span>
#include <CommonDataFormat/InteractionRecord.h>
#include <ITSMFTReconstruction/GBTWord.h>
#include <DataFormatsFOCAL/PixelChip.h>
#include <DataFormatsFOCAL/PixelHit.h>
#include <FOCALReconstruction/PixelWords.h>
#include <FOCALReconstruction/PixelLaneData.h>

namespace o2::focal
{

class PixelDecoder
{
 public:
  PixelDecoder() = default;
  ~PixelDecoder() = default;

  void reset();
  void decodeEvent(gsl::span<const o2::itsmft::GBTWord> payload);

  const std::unordered_map<o2::InteractionRecord, std::vector<PixelChip>>& getChipData() const { return mChipData; }

 private:
  std::vector<PixelChip> decodeLane(uint8_t laneID, gsl::span<const uint8_t> laneWords);
  PixelWord::PixelWordType getWordType(uint8_t payloadword);
  uint16_t AlpideX(uint8_t region, uint8_t encoder, uint16_t address);
  uint16_t AlpideY(uint16_t address);

  std::unordered_map<o2::InteractionRecord, std::shared_ptr<PixelLaneHandler>> mPixelData;
  std::unordered_map<o2::InteractionRecord, std::vector<PixelChip>> mChipData;
};

} // namespace o2::focal
#endif // QC_MODULE_FOCAL_PIXELDECODER_H