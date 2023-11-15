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

#ifndef O2_MCH_SIMULATION_DIGITIZER_H
#define O2_MCH_SIMULATION_DIGITIZER_H

#include <map>
#include <memory>
#include <random>

#include <gsl/span>

#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MCHSimulation/Hit.h"
#include "MCHSimulation/DEDigitizer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2::mch
{
/** MCH Digitizer.
 * This class is just steering the usage of o2::mch::DEDigitizer
 */
class Digitizer
{
 public:
  /** Constructor.
   * @param transformationCreator is a function that is able to create a o2::math_utils::Transform3D
   */
  Digitizer(geo::TransformationCreator transformationCreator);

  /// @see DEDigitizer::processHit
  void processHits(gsl::span<const Hit> hits, const InteractionRecord& collisionTime, int evID, int srcID);

  /// @see DEDigitizer::addNoise
  void addNoise(const InteractionRecord& firstIR, const InteractionRecord& lastIR);

  /** @see DEDigitizer::digitize
   * Fill the given containers with the result of the digitization.
   * Return the number of signal pileup in overlapping readout windows.
   */
  size_t digitize(std::vector<ROFRecord>& rofs, std::vector<Digit>& digits, dataformats::MCLabelContainer& labels);

  /// @see DEDigitizer::clear
  void clear();

 private:
  std::mt19937 mRandom; ///< random number generator

  std::map<int, std::unique_ptr<DEDigitizer>> mDEDigitizers; ///< list of digitizers per DE
};

} // namespace o2::mch
#endif
