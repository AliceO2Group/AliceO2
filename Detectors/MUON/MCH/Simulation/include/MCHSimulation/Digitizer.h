// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_DIGITIZER_H
#define O2_MCH_SIMULATION_DIGITIZER_H

#include "MCHSimulation/Hit.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MCHSimulation/DEDigitizer.h"
#include <map>
#include <gsl/span>
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

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
   * a geo::TransformationCreator
   */
  Digitizer(geo::TransformationCreator transformationCreator);

  // @see DEDigitizer::addNoise
  void addNoise(float noiseProba);

  // @see DEDigitizer::startCollision
  void startCollision(o2::InteractionRecord collisionTime);

  // @see DEDigitizer::processHit
  void processHits(gsl::span<Hit> hits, int evID, int srcID);

  // @see DEDigitizer::extractDigitsAndLabels
  void extractDigitsAndLabels(std::vector<Digit>& digits,
                              o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels);

  // @see DEDigitizer:clear
  void clear();

 private:
  std::map<int, std::unique_ptr<DEDigitizer>> mDEDigitizers; // list of workers
};

/** Group Interaction Record that are "too close" in time (BC).
 *
 * @param records : a list of input IR to group
 * @param width (in BC unit) : all IRs within this distance will be considered
 * to be a single group
 *
 * @returns a map of IRs->{index} where index is relative to input records
 *
 */
std::map<o2::InteractionRecord, std::vector<int>> groupIR(gsl::span<const o2::InteractionRecord> records, uint32_t width = 4);

/** Same as above for InteractionTimeRecord. */
std::map<o2::InteractionRecord, std::vector<int>> groupIR(gsl::span<const o2::InteractionTimeRecord> records, uint32_t width = 4);

} // namespace o2::mch
#endif
