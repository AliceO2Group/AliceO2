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

#include "DataFormatsMCH/ROFRecord.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MCHSimulation/DEDigitizer.h"
#include "MCHSimulation/Hit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <gsl/span>
#include <map>

namespace o2::mch
{
/** MCH Digitizer.
 *
 * This class is just steering the usage of o2::mch::DEDigitizer
 *
 */
class Digitizer
{
 public:
  /** Constructor.
   * @param transformationCreator is a function that is able to create
   *        a geo::TransformationCreator
   *
   * for the other parameters @see DEDigitizer::DEDigitizer
   */
  Digitizer(geo::TransformationCreator transformationCreator,
            float timeSpread,
            float noiseChargeMean,
            float noiseChargeSigma,
            int seed);

  /** @see DEDigitizer::addNoise */
  void addNoise(float noiseProba,
                const o2::InteractionRecord& firstIR,
                const o2::InteractionRecord& lastIR);

  /** @see DEDigitizer::processHit */
  void processHits(const o2::InteractionRecord& collisionTime,
                   gsl::span<Hit> hits, int evID, int srcID);

  /** fills (adds to) the given vectors of rofs, digits and labels with our internal
   *  information so far.
   */
  void extract(std::vector<o2::mch::ROFRecord>& rofs,
               std::vector<Digit>& digits,
               o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels);

  /** Clear our internal storage. */
  void clear();

 private:
  std::map<int, std::unique_ptr<DEDigitizer>> mDEDigitizers; // list of workers
};

} // namespace o2::mch
#endif
