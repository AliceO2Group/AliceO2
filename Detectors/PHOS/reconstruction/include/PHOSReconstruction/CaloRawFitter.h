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

/// \class CaloRawFitter
/// \brief  Raw data fitting: extraction amplitude and time
///
/// Extraction of amplitude and time
/// from CALO raw data using fast k-level approach or
/// least square fit with Gamma2 function
///
/// \author Dmitri Peresunko
/// \since Jan.2020
///

#ifndef PHOSRAWFITTER_H_
#define PHOSRAWFITTER_H_
#include "Rtypes.h"

namespace o2
{

namespace phos
{

class CaloRawFitter
{

 public:
  enum FitStatus { kOK,
                   kNotEvaluated,
                   kEmptyBunch,
                   kOverflow,
                   kSpike,
                   kNoTime,
                   kFitFailed,
                   kBadPedestal,
                   kManyBunches };

 public:
  /// \brief Constructor
  CaloRawFitter();

  /// \brief Destructor
  virtual ~CaloRawFitter() = default;

  /// \brief Evaluation Amplitude and TOF
  /// return status -1: not evaluated/empty bunch;
  ///                0: OK;
  ///                1: overflow;
  ///                4: single spikes
  ///                3: too large RMS;
  virtual FitStatus evaluate(gsl::span<short unsigned int> signal);

  /// \brief Set HighGain/LowGain channel to performe or not fit of saturated samples
  void setLowGain(bool isLow = false) { mLowGain = isLow; }

  /// \brief estimate and subtract pedestals from pre-samples
  void setPedSubtract(bool toSubtruct = false) { mPedSubtract = toSubtruct; }

  /// \brief amplitude in last fitted sample
  float getAmp() const { return mAmp; }

  /// \brief Chi2/NDF of last performed fit
  float getChi2() const { return mChi2; }

  /// \brief time in last fitted sample
  float getTime() const { return mTime; }

  /// \brief is last fitted sample has overflow
  bool isOverflow() const { return mOverflow; }

  /// \brief Forse perform fitting
  /// Make fit for any sample, not only saturated LowGain samples as by default
  void forseFitting(bool toRunFit = true) { makeFit = toRunFit; }

  /// \brief Set analysis of pedestal run
  /// Analyze pedestal run, i.e. calculate mean and RMS of pedestals instead of Amp and Time
  void setPedestal() { mPedestalRun = true; }

 protected:
  FitStatus evalKLevel(gsl::span<short unsigned int> signal);

 protected:
  bool makeFit = false;              ///< run (slow) fit with Gamma2 or use fast evaluation with k-level
  bool mLowGain = false;             ///< is current bunch from LowGain channel
  bool mPedSubtract = false;         ///< should one evaluate and subtract pedestals
  bool mPedestalRun = false;         ///< analyze as pedestal run
  bool mOverflow;                    ///< is last sample saturated
  FitStatus mStatus = kNotEvaluated; ///< status of last evaluated sample: -1: not yet evaluated; 0: OK; 1: overflow; 2: too large RMS; 3: single spikes
  short mMaxSample = 0;              ///< maximal sample
  float mAmp;                        ///< amplitude of last processed sample
  float mTime;                       ///< time of last processed sample
  float mChi2;                       ///< chi2 calculated in last fit
  short mSpikeThreshold;
  short mBaseLine;
  short mPreSamples;

  ClassDef(CaloRawFitter, 1);
}; // End of CaloRawFitter

} // namespace phos

} // namespace o2
#endif
