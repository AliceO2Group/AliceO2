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

#include "MCHSimulation/DigitizerParam.h"

namespace o2::mch
{

Digitizer::Digitizer(geo::TransformationCreator transformationCreator)
  : mRandom{DigitizerParam::Instance().seed == 0 ? std::random_device{}() : DigitizerParam::Instance().seed}
{
  mapping::forEachDetectionElement([&](int deId) {
    mDEDigitizers[deId] = std::make_unique<DEDigitizer>(deId, transformationCreator(deId), mRandom);
  });
}

void Digitizer::processHits(gsl::span<const Hit> hits, const InteractionRecord& collisionTime, int evID, int srcID)
{
  for (const auto& hit : hits) {
    mDEDigitizers[hit.detElemId()]->processHit(hit, collisionTime, evID, srcID);
  }
}

void Digitizer::addNoise(const InteractionRecord& firstIR, const InteractionRecord& lastIR)
{
  for (auto& d : mDEDigitizers) {
    d.second->addNoise(firstIR, lastIR);
  }
}

size_t Digitizer::digitize(std::vector<ROFRecord>& rofs,
                           std::vector<Digit>& digits,
                           dataformats::MCLabelContainer& labels)
{
  // digitize every DE and store digits and labels ordered per IR
  size_t nPileup = 0;
  std::map<InteractionRecord, DEDigitizer::DigitsAndLabels> irDigitsAndLabels{};
  for (auto& d : mDEDigitizers) {
    nPileup += d.second->digitize(irDigitsAndLabels);
  }

  // fill the external containers
  for (const auto& [ir, digitsAndLabels] : irDigitsAndLabels) {
    rofs.emplace_back(ROFRecord(ir, digits.size(), digitsAndLabels.first.size()));
    digits.insert(digits.end(), digitsAndLabels.first.begin(), digitsAndLabels.first.end());
    labels.mergeAtBack(digitsAndLabels.second);
  }

  return nPileup;
}

void Digitizer::clear()
{
  for (auto& d : mDEDigitizers) {
    d.second->clear();
  }
}

} // namespace o2::mch
