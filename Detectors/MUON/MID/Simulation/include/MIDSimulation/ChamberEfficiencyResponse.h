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

/// \file   MIDSimulation/ChamberEfficiencyResponse.h
/// \brief  MID RPC effciency response
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2019

#ifndef O2_MID_CHAMBEREFFICIENCYRESPONSE_H
#define O2_MID_CHAMBEREFFICIENCYRESPONSE_H

#include <random>
#include "DataFormatsMID/ChEffCounter.h"
#include "MIDEfficiency/ChamberEfficiency.h"

namespace o2
{
namespace mid
{
class ChamberEfficiencyResponse
{
 public:
  ChamberEfficiencyResponse(const ChamberEfficiency& efficiencyMap);
  virtual ~ChamberEfficiencyResponse() = default;

  bool isEfficient(int deId, int columnId, int line, bool& isEfficientBP, bool& isEfficientNBP);

  /// Sets the seed
  void setSeed(unsigned int seed) { mGenerator.seed(seed); }

  /// Sets the chamber efficiency from the counters
  void setFromCounters(const std::vector<ChEffCounter>& counters) { mEfficiency.setFromCounters(counters); }

 private:
  ChamberEfficiency mEfficiency;                  ///< Measured chamber efficiencies
  std::default_random_engine mGenerator;          ///< Random numbers generator
  std::uniform_real_distribution<double> mRandom; ///< Uniform distribution
};

ChamberEfficiencyResponse createDefaultChamberEfficiencyResponse();

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHAMBERRESPONSE_H */
