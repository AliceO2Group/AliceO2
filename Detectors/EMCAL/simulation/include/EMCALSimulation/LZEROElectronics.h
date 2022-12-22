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

#ifndef ALICEO2_EMCAL_LZEROELECTRONICS_H_
#define ALICEO2_EMCAL_LZEROELECTRONICS_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <deque>
#include <list>
#include <optional>
#include <gsl/span>
#include "TRandom3.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALSimulation/DigitTimebin.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "EMCALSimulation/SimParam.h"
#include "EMCALSimulation/Patches.h"
#include "EMCALBase/TriggerMappingV2.h"

namespace o2
{
namespace emcal
{

/// @brief Trigger Inputs object, summary of the entire information needed for the L1 algorithm
/// \param mDelay Typical delay in BCs
/// \param mRollback Rollback from EMCALReconstruction/RecoParam.h
/// \param mInterRecord Last known interaction record
/// \param mLastTimesumAllFastOrs Vector of tuples with TRU ID, FastOrID with STU indexing, and their last Timesums
struct EMCALTriggerInputs {
  int mDelay = 8;                                                   ///< Typical delay in BCs
  int mRollback = 1;                                                ///< Rollback from EMCALReconstruction/RecoParam.h
  o2::InteractionRecord mInterRecord;                ///< Last known interaction record
  std::vector<std::tuple<int, int, double>> mLastTimesumAllFastOrs; ///< TRU ID, FastOrID with STU indexing, and its last Timesum
  ClassDefNV(EMCALTriggerInputs, 1);
};

/// \class LZEROElectronics
/// \brief Container class for Digits, MC lebels, and trigger records
/// \ingroup EMCALsimulation
/// \author Markus Fasel, ORNL
/// \author Simone Ragoni, Creighton
/// \date 22/11/2022

class LZEROElectronics
{

 public:
  /// Default constructor
  LZEROElectronics() = default;

  /// Destructor
  ~LZEROElectronics() = default;

  /// clear the L0 electronics
  void clear();

  /// Initialize the L0 electronics
  void init();

  /// Set Threshold for LZERO algorithm
  /// \param threshold LZERO algorithm threshold
  void setThreshold(double threshold) { mThreshold = threshold; }

  /// Implements the peak finder algorithm on the patch
  /// \param p Patches object
  /// \param patchID Patch ID to implement the peak finding algorithm
  bool peakFinderOnPatch(Patches& p, unsigned int patchID);

  /// Calls the peak finder algorithm on all patches
  /// \param p Patches object
  bool peakFinderOnAllPatches(Patches& p);

  /// Update patches
  /// \param p Patches object
  void updatePatchesADC(Patches& p);

  /// Add noise to this digit
  void addNoiseDigits(Digit& d1);

  /// Implements the fill of the patches. Runs the peak finding, and ships to L1 in case it finds something
  /// \param digitlist digits to be assigned to patches
  /// \param record interaction record time to be propagated
  /// \param patchesFromAllTRUs vector contained the patches of all TRUs
  void fill(std::deque<o2::emcal::DigitTimebinTRU>& digitlist, o2::InteractionRecord record, std::vector<Patches>& patchesFromAllTRUs);

  /// Getter for the pattern of peaks found by the LZERO algorithm
  /// \param p Patches object
  const std::vector<int>& getFiredPatches(Patches& p) const
  {
    return p.mFiredPatches;
  }

  // Getter for the threshold used for the integral of the ADC values in the LZERO algorithm
  const double getLZEROThreshold() const { return mThreshold; }

 private:
  double mThreshold = 0;
  TRandom3* mRandomGenerator = nullptr; ///< random number generator
  const SimParam* mSimParam = nullptr;  ///< SimParam object
  bool mSimulateNoiseDigits = true;     ///< simulate noise digits
  TriggerMappingV2 triggerMap;          ///< trigger map to properly assign an absolute FastOr to TRU FastOr

  ClassDefNV(LZEROElectronics, 2);
};

} // namespace emcal

} // namespace o2

#endif /* ALICEO2_EMCAL_LZEROELECTRONICS_H_ */
