// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/Digitizer.h"
#include "MCHMappingInterface/Segmentation.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <algorithm>

namespace o2::mch
{

Digitizer::Digitizer(geo::TransformationCreator transformationCreator)
{
  mapping::forEachDetectionElement([&transformation = transformationCreator, &digitizers = this->mDEDigitizers](int deId) {
    digitizers[deId] = std::make_unique<DEDigitizer>(deId, transformation(deId));
  });
}

void Digitizer::addNoise(float noiseProba)
{
  if (noiseProba > 0) {
    for (auto& d : mDEDigitizers) {
      d.second->addNoise(noiseProba);
    }
  }
}

void Digitizer::startCollision(o2::InteractionRecord collisionTime)
{
  for (auto& d : mDEDigitizers) {
    d.second->startCollision(collisionTime);
  }
}

void Digitizer::processHits(gsl::span<Hit> hits, int evID, int srcID)
{
  for (const auto& hit : hits) {
    mDEDigitizers[hit.detElemId()]->process(hit, evID, srcID);
  }
}

void Digitizer::extractDigitsAndLabels(std::vector<Digit>& digits,
                                       o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels)
{
  for (auto& d : mDEDigitizers) {
    d.second->extractDigitsAndLabels(digits, labels);
  }
}

void Digitizer::clear()
{
  for (auto& d : mDEDigitizers) {
    d.second->clear();
  }
}

std::map<o2::InteractionRecord, std::vector<int>> groupIR(gsl::span<const o2::InteractionTimeRecord> records, uint32_t width)
{
  std::vector<o2::InteractionRecord> irs;
  for (const auto& ir : records) {
    irs.emplace_back(ir.bc, ir.orbit);
  }
  return groupIR(irs, width);
}

std::map<o2::InteractionRecord, std::vector<int>> groupIR(gsl::span<const o2::InteractionRecord> records, uint32_t width)
{
  if (!std::is_sorted(records.begin(), records.end())) {
    throw std::invalid_argument("input records must be sorted");
  }
  std::map<o2::InteractionRecord, std::vector<int>> binned;
  auto ir0 = records[0];
  for (auto i = 0; i < records.size(); ++i) {
    auto ir = records[i];
    auto mchIR = ir;
    if (ir.differenceInBC(ir0) < width) {
      mchIR = ir0;
    } else {
      ir0 = ir;
    }
    binned[mchIR].emplace_back(i);
  }
  return binned;
}

} // namespace o2::mch
