// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackBasedCalib.h
/// \brief Provides information required for TRD calibration which is based on the global tracking
/// \author Ole Schmidt

#ifndef O2_TRD_TRACKBASEDCALIB_H
#define O2_TRD_TRACKBASEDCALIB_H

#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/AngularResidHistos.h"
#include "DetectorsBase/Propagator.h"
#include "TRDBase/RecoParam.h"

#include "Rtypes.h"

#include <gsl/span>

namespace o2
{

namespace globaltracking
{
class RecoContainer;
}

namespace trd
{

class TrackBasedCalib
{
  using MatCorrType = o2::base::Propagator::MatCorrType;

 public:
  TrackBasedCalib() = default;
  TrackBasedCalib(const TrackBasedCalib&) = delete;
  ~TrackBasedCalib() = default;

  /// Load geometry and apply magnetic field setting
  void init();

  /// Main processing function for creating angular residual histograms for vDrift and ExB calibration
  void calculateAngResHistos(const o2::globaltracking::RecoContainer& input);

  /// Main processing function for gathering information needed for gain calibration
  /// i.e. TRD tracklet ADC vs TPC track dEdx for given momentum slice
  void calculateGainCalibObjs(const o2::globaltracking::RecoContainer& input);

  /// Extrapolate track parameters to given layer and if requested perform update with tracklet
  bool propagateAndUpdate(TrackTRD& trk, int iLayer, bool doUpdate) const;

  const AngularResidHistos& getAngResHistos() const { return mAngResHistos; }

 private:
  float mMaxSnp{o2::base::Propagator::MAX_SIN_PHI};  ///< max snp when propagating tracks
  float mMaxStep{o2::base::Propagator::MAX_STEP};    ///< maximum step for propagation
  MatCorrType mMatCorr{MatCorrType::USEMatCorrNONE}; ///< if material correction should be done
  RecoParam mRecoParam;                              ///< parameters required for TRD reconstruction
  AngularResidHistos mAngResHistos;                  ///< aggregated data for the track based calibration
  // input arrays which should not be modified since they are provided externally
  gsl::span<const TrackTRD> mTracksIn;                 ///< TRD tracks reconstructed from TPC or ITS-TPC seeds
  gsl::span<const Tracklet64> mTrackletsRaw;           ///< array of raw tracklets needed for TRD refit
  gsl::span<const CalibratedTracklet> mTrackletsCalib; ///< array of calibrated tracklets needed for TRD refit

  ClassDefNV(TrackBasedCalib, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_TRACKBASEDCALIB_H
