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

#ifndef O2_MCH_SIMULATION_DE_DIGITIZER_H
#define O2_MCH_SIMULATION_DE_DIGITIZER_H

#include "DataFormatsMCH/Digit.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHSimulation/Hit.h"
#include "MCHSimulation/Response.h"
#include "MathUtils/Cartesian.h"
#include <set>
#include <vector>
#include <map>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <random>

namespace o2::mch
{

/** MCH Digitizer dealing with a single detection element. */

class DEDigitizer
{
 public:
  /** Constructor.
   *
   * @param deId detection element
   * @param transformation a geo transformation to convert global coordinates
   *        (of this hits) into local (to the detection element) ones
   * @param timeSpread the sigma of the time resolution to apply to digit
   *        times (in nanoseconds, typically O(100ns))
   * @param noiseChargeMean the mean of the charge to assign to noise digits
   *        (in ADC counts, typically O(10))
   * @param noiseChargeSigma the sigma of the charge to assign to noise digits
   *        (in ADC counts, typically O(1))
   * @param seed for random number generation
   */
  DEDigitizer(int deId,
              o2::math_utils::Transform3D transformation,
              float timeSpread,
              float noiseChargeMean,
              float noiseChargeSigma,
              int seed);

  /** Add some noise-only digits.
   *
   * @param noiseProba the probability that a given pad is above threshold
   * (upon a given duration = one ADC sample = 100ns)
   *
   * Noise digits will be generated for all IR between `firstIR`
   * and `lastIR`. Note that depending on the actual value of noiseProba
   * some IR might not get any noisy digit, and that's fine.
   *
   * The noise digits will be added to any pre-existing digits (signal
   * digits).
   */
  void addNoise(float noiseProba,
                const o2::InteractionRecord& firstIR,
                const o2::InteractionRecord& lastIR);

  /** process one MCH Hit.
   *
   * This will convert the hit eloss into a charge and spread (according
   * to a Mathieson 2D distribution) that charge among several pads,
   * that will become digits eventually
   *
   * @param collisionTime the IR this hit is associated to
   * @param hit the input hit to be digitized
   * @param evID the event identifier of the hit in the sim chain within srcID
   * @param srcID the origin (signal, background) of the hit
   * @see o2::steer::EventPart
   */
  void process(const o2::InteractionRecord& collisionTime, const Hit& hit, int evID, int srcID);

  /** extractDigitsAndLabels appends digits and labels corresponding
   * to a given ROF to the given containers.
   *
   * This function copies our internal information into the parameter vectors
   *
   * @param rofs the ROF for which we want the digits and labels.
   * @param digits vector of digits where we append our internal digits.
   * @param labels MCTruthContainer where we append our internal labels.
   */
  void extractDigitsAndLabels(const o2::InteractionRecord& rof,
                              std::vector<Digit>& digits,
                              o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels);

  void extractRofs(std::set<o2::InteractionRecord>& rofs);

  void appendDigit(o2::InteractionRecord ir,
                   int padid, float charge, const o2::MCCompLabel& label);

  /** Clear our internal storage (pad charges and so on). */
  void clear();

 private:
  /** get a random time shift (wrt to collision time), in bc unit.
   */
  int64_t drawRandomTimeShift();

  /** shift ir0 by some random number of BCs, thus
   * simulating the detector time resolution
   */
  o2::InteractionRecord shiftTime(o2::InteractionRecord ir);

  struct Pad {
    int padid;
    float charge;
    std::vector<o2::MCCompLabel> labels;
    Pad(int padid_, float charge_, const o2::MCCompLabel& label) : padid{padid_}, charge{charge_}, labels{label} {}
  };
  int mDeId;                                   // detection element id
  Response mResponse;                          // response function (Mathieson parameters, ...)
  o2::math_utils::Transform3D mTransformation; // from local to global and reverse
  mapping::Segmentation mSegmentation;         // mapping of this detection element
  std::map<o2::InteractionRecord, std::vector<Pad>> mPadMap;
  std::normal_distribution<float> mTimeDist;
  std::normal_distribution<float> mChargeDist;
  std::mt19937 mGene;
};

} // namespace o2::mch

#endif
