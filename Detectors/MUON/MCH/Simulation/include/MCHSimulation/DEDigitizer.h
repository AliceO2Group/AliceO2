// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_DE_DIGITIZER_H
#define O2_MCH_SIMULATION_DE_DIGITIZER_H

#include "DataFormatsMCH/Digit.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHSimulation/Hit.h"
#include "MCHSimulation/Response.h"
#include "MathUtils/Cartesian.h"
#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2::mch
{

/** MCH Digitizer dealing with a single detection element. */

class DEDigitizer
{
 public:
  DEDigitizer(int deId, o2::math_utils::Transform3D transformation);

  /** startCollision sets the current IR under consideration.
   *
   * All the digits created between a call to this one and clear()
   * will be associated to a MCH ROFRecord == collisionTime
  */
  void startCollision(o2::InteractionRecord collisionTime);

  /** Add some noise to the current collision */
  void addNoise(float noiseProba);

  /** process one MCH Hit.
   *
   * This will convert the hit eloss into a charge and spread (according
   * to a Mathieson 2D distribution) that charge among several pads,
   * that will become digits eventually
   *
   * @param hit the input hit to be digitized
   * @param evID the event identifier of the hit in the sim chain within srcID
   * @param srcID the origin (signal, background) of the hit
   * @see o2::steer::EventPart
   */
  void process(const Hit& hit, int evID, int srcID);

  /** extractDigitsAndLabels appends digits and labels to given containers.
   *
   * This function copies our internal information into the parameter vectors
   *
   * @param digits vector of digits where we append our internal digits.
   * @param labels MCTruthContainer where we append our internal labels.
   */
  void extractDigitsAndLabels(std::vector<Digit>& digits,
                              o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels);

  /** Clear resets our internal lists of digits and labels. */
  void clear();

 private:
  int mDeId;                                         // detection element id
  Response mResponse;                                // response function (Mathieson parameters, ...)
  o2::math_utils::Transform3D mTransformation;       // from local to global and reverse
  mapping::Segmentation mSegmentation;               // mapping of this detection element
  o2::InteractionRecord mIR;                         // interaction record to associate charges and labels to
  std::vector<float> mCharges;                       // pad charges (fixed size = number of pads in this DE)
  std::vector<std::vector<o2::MCCompLabel>> mLabels; // mLabels.size()==mCharges.size()
};

} // namespace o2::mch

#endif
