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

#include "MCHSimulation/Digitizer.h"
#include "MCHMappingInterface/Segmentation.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <algorithm>
#include <set>

namespace o2::mch
{

Digitizer::Digitizer(geo::TransformationCreator transformationCreator,
                     float timeSpread,
                     float noiseChargeMean,
                     float noiseChargeSigma,
                     int seed)
{
  mapping::forEachDetectionElement([&transformation = transformationCreator,
                                    &digitizers = this->mDEDigitizers,
                                    timeSpread,
                                    noiseChargeMean,
                                    noiseChargeSigma,
                                    seed](int deId) {
    digitizers[deId] = std::make_unique<DEDigitizer>(deId, transformation(deId),
                                                     timeSpread,
                                                     noiseChargeMean,
                                                     noiseChargeSigma,
                                                     seed);
  });
}

void Digitizer::addNoise(float noiseProba,
                         const o2::InteractionRecord& firstIR,
                         const o2::InteractionRecord& lastIR)
{
  if (noiseProba > 0) {
    for (auto& d : mDEDigitizers) {
      d.second->addNoise(noiseProba, firstIR, lastIR);
    }
  }
}

void Digitizer::processHits(const o2::InteractionRecord& collisionTime,
                            gsl::span<Hit> hits, int evID, int srcID)
{
  for (const auto& hit : hits) {
    mDEDigitizers[hit.detElemId()]->process(collisionTime, hit, evID, srcID);
  }
}

void Digitizer::extract(std::vector<o2::mch::ROFRecord>& rofs,
                        std::vector<Digit>& digits,
                        o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels)
{
  // get unique rofs
  std::set<o2::InteractionRecord> irs;
  for (auto& d : mDEDigitizers) {
    d.second->extractRofs(irs);
  }

  // loop over those unique rofs and find out the digits (and labels)
  // associated to them
  for (auto ir : irs) {
    std::vector<o2::mch::Digit> irDigits;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> irLabels;
    for (auto& d : mDEDigitizers) {
      d.second->extractDigitsAndLabels(ir, irDigits, irLabels);
    }
    rofs.emplace_back(ROFRecord(ir, digits.size(), irDigits.size()));
    digits.insert(digits.end(), irDigits.begin(), irDigits.end());
    labels.mergeAtBack(irLabels);
  }
}

void Digitizer::clear()
{
  for (auto& d : mDEDigitizers) {
    d.second->clear();
  }
}

} // namespace o2::mch
