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

/// \file KrClusterFinder.h
/// \brief The TRD Krypton cluster finder from digits
/// \author Ole Schmidt

#ifndef O2_TRD_KRCLUSTERFINDER_H
#define O2_TRD_KRCLUSTERFINDER_H

#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/KrCluster.h"
#include "DataFormatsTRD/KrClusterTriggerRecord.h"

#include "Rtypes.h"
#include "TF1.h"
#include "Fit/Fitter.h"
#include "Fit/FitResult.h"

#include <memory>
#include <gsl/span>

namespace o2
{

namespace trd
{

class KrClusterFinder
{

 public:
  KrClusterFinder() = default;
  KrClusterFinder(const KrClusterFinder&) = delete;
  ~KrClusterFinder() = default;

  struct LandauChi2Functor {
    double operator()(const double* par) const;
    std::vector<float> x; ///< the number of the time bin
    std::vector<float> y; ///< ADC sum per time bin
    int xLowerBound;
    int xUpperBound;
  };

  /// Initialization
  void init();

  // Reset output containers
  void reset();

  /// Provide digits and trigger records as input
  void setInput(const gsl::span<const Digit>& digitsIn, const gsl::span<const TriggerRecord>& trigRecIn);

  /// Find clusters in the digit ADC data.
  /// We start with finding the maximum ADC value in a chamber. We store the ADC value, the time bin within the corresponding digit
  /// and the pad row and pad column of that digit. Afterwards we cluster around that maximum, discarding digits more than 1 pad row
  /// or more than 2 pad columns away. In time bin direction there are no limits for other ADC values to contribute to the cluster.
  /// Each ADC value is added maximum to a single cluster: ADCs connected to one cluster are flagged as used and disregarded
  /// in the following iterations for the given detector. Also ADC below a threshold (mMinAdcClContrib) are ignored.
  /// The cluster size is defined in three dimensions (row, column, time bin), for each dimension from the lowermost constituent ADC
  /// to the uppermost constituent ADC.
  /// Different ADC sums are calculated from the constituent ADCs.
  /// Afterwards the main Krypton peak is identified (maxAdcA and maxTbA) and also we try to find the second Kr peak (maxAdcB, maxTbB).
  /// Quality checks indicate the validity of those two maxima (shape, ordering, ADC size)
  /// A Landau fit to the ADC vs time bins values for a cluster provide the ADC sums for the two Kr peaks.
  /// The clusters don't necessarily need to fullfill all quality criteria (e.g. the Landau fit may fail). But these clusters are
  /// stored anyway, as they can be filtered out later at the analysis stage.
  void findClusters();

  /// Calculate some statistics for the given cluster constituent ADC values
  double getRms(const std::vector<uint64_t>& adcIndices, int itTrunc, double nRmsTrunc, int minAdc, double& rmsTime, uint32_t& sumAdc) const;

  /// Output
  const std::vector<KrCluster>& getKrClusters() const { return mKrClusters; }
  const std::vector<KrClusterTriggerRecord>& getKrTrigRecs() const { return mTrigRecs; }

 private:
  // input
  gsl::span<const Digit> mDigits;                 ///< the TRD digits
  gsl::span<const TriggerRecord> mTriggerRecords; ///< the TRD trigger records
  // output
  std::vector<KrCluster> mKrClusters{};            ///< the Kr clusters
  std::vector<KrClusterTriggerRecord> mTrigRecs{}; ///< number of Kr clusters and interaction record of first trigger per time frame
  // helpers
  LandauChi2Functor mLandauChi2Functor;                                         ///< stores the binned ADC data and provides a chi2 estimate
  std::shared_ptr<ROOT::Fit::FitResult> mFitResult{new ROOT::Fit::FitResult()}; ///< pointer to the results of the Landau fit
  ROOT::Fit::Fitter mFitter{mFitResult};                                        ///< an instance of the ROOT fitter
  std::array<double, 3> mInitialFitParams{1., 1., 1.};                          ///< initial fit parameters for the Landau fit
  std::unique_ptr<TF1> mFuncLandauFit;                                          ///< helper function to approximate the binned ADC data with a Landau distribution
  // settings
  const int mBaselineAdc{10};        ///< ADC baseline for each pad (can maybe be moved into Constants.h)
  const int mMinAdcForMax{70};       ///< minimum ADC value that can be considered for a maximum
  const int mMinAdcClContrib{40};    ///< minimum ADC value that can contribute to a cluster
  const int mMinAdcForSecondMax{50}; ///< threshold for a second maximum inside a cluster
  const int mMinAdcClEoverT{60};     ///< energy threshold for ADC value

  ClassDefNV(KrClusterFinder, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_KRCLUSTERFINDER_H
