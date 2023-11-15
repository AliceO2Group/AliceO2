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

#include <map>
#include <random>
#include <utility>
#include <vector>

#include "DataFormatsMCH/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "MathUtils/Cartesian.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHSimulation/Hit.h"
#include "MCHSimulation/Response.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2::mch
{

/// MCH digitizer dealing with a single detection element.
class DEDigitizer
{
 public:
  using DigitsAndLabels = std::pair<std::vector<Digit>, dataformats::MCLabelContainer>;

  /** Constructor.
   * @param deId detection element ID
   * @param transformation a transformation to convert global coordinates
   *        (of this hits) into local (to the detection element) ones
   * @param random random number generator
   */
  DEDigitizer(int deId, math_utils::Transform3D transformation, std::mt19937& random);

  /** Process one MCH Hit.
   *
   * This will convert the hit eloss into a charge and spread (according
   * to a Mathieson 2D distribution) that charge among several pads,
   * that will become digits eventually
   *
   * @param hit the input hit to be processed
   * @param collisionTime the IR this hit is associated to
   * @param evID the event identifier of the hit in the sim chain within srcID
   * @param srcID the origin (signal, background) of the hit
   * @see o2::steer::EventPart
   */
  void processHit(const Hit& hit, const InteractionRecord& collisionTime, int evID, int srcID);

  /** Add noise-only signals.
   *
   * Noise will be generated for all IR between `firstIR` and `lastIR`.
   * Note that depending on the actual noise probability some IR
   * might not get any pad with noise-only signal, and that's fine.
   *
   * @param firstIR first IR for which noise will be generated
   * @param lastIR last IR for which noise will be generated
   */
  void addNoise(const InteractionRecord& firstIR, const InteractionRecord& lastIR);

  /** Do the digitization.
   *
   * The digitization accounts for charge dispersion (adding noise to physical
   * signals) and threshold, time dispersion and pileup within readout windows.
   * It fills the lists of digits and associated labels ordered per IR
   * and returns the number of signal pileup in overlapping readout windows.
   *
   * @param irDigitsAndLabels lists of digits and associated labels ordered per IR
   */
  size_t digitize(std::map<InteractionRecord, DigitsAndLabels>& irDigitsAndLabels);

  /// Clear the internal lists of signals.
  void clear();

 private:
  /// internal structure to hold signal informations
  struct Signal {
    InteractionRecord rofIR;
    uint8_t bcInROF;
    float charge;
    std::vector<MCCompLabel> labels;
    Signal(const InteractionRecord& ir, uint8_t bc, float q, const MCCompLabel& label)
      : rofIR{ir}, bcInROF{bc}, charge{q}, labels{label} {}
  };

  /// add a physical signal to the given pad at the given IR
  void addSignal(int padid, const InteractionRecord& collisionTime, float charge, const MCCompLabel& label);
  /// add a noise-only signal to the given pad at the given IR
  void addNoise(int padid, const InteractionRecord& rofIR);
  /// add noise to the given signal
  void addNoise(Signal& signal, uint32_t nSamples);
  /// add time dispersion to signal
  void addTimeDispersion(Signal& signal);
  /// test if the charge is above threshold
  bool isAboveThreshold(float charge);
  /// add a new digit and associated labels at the relevant IR and return a pointer to their container
  DigitsAndLabels* addNewDigit(std::map<InteractionRecord, DigitsAndLabels>& irDigitsAndLabels,
                               int padid, const Signal& signal, uint32_t nSamples) const;
  /// add signal and associated labels to the last created digit
  void appendLastDigit(DigitsAndLabels* digitsAndLabels, const Signal& signal, uint32_t nSamples) const;

  int mDeId;                                   ///< detection element ID
  Response mResponse;                          ///< response function (Mathieson parameters, ...)
  o2::math_utils::Transform3D mTransformation; ///< transformation from local to global and reverse
  mapping::Segmentation mSegmentation;         ///< mapping of this detection element

  std::mt19937& mRandom;                            ///< reference to the random number generator
  std::normal_distribution<float> mMinChargeDist;   ///< random lower charge threshold generator (gaussian distribution)
  std::normal_distribution<float> mTimeDist;        ///< random time dispersion generator (gaussian distribution)
  std::normal_distribution<float> mNoiseDist;       ///< random charge noise generator (gaussian distribution)
  std::normal_distribution<float> mNoiseOnlyDist;   ///< random noise-only signal generator (gaussian distribution)
  std::poisson_distribution<int> mNofNoisyPadsDist; ///< random number of noisy pads generator (poisson distribution)
  std::uniform_int_distribution<int> mPadIdDist;    ///< random pad ID generator (uniform distribution)
  std::uniform_int_distribution<int> mBCDist;       ///< random BC number inside ROF generator (uniform distribution)

  std::vector<std::vector<Signal>> mSignals; ///< list of signals per pad
};

} // namespace o2::mch

#endif
