// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackParam.h
/// \brief Definition of the MCH track parameters for internal use
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKPARAM_H_
#define ALICEO2_MCH_TRACKPARAM_H_

#include <TMatrixD.h>

#include "MCHBase/TrackBlock.h"

namespace o2
{
namespace mch
{

class Cluster;

/// track parameters for internal use
class TrackParam
{
 public:
  TrackParam() = default;
  ~TrackParam() = default;

  TrackParam(const TrackParam& tp);
  TrackParam& operator=(const TrackParam& tp);
  TrackParam(TrackParam&&) = delete;
  TrackParam& operator=(TrackParam&&) = delete;

  /// return Z coordinate (cm)
  Double_t getZ() const { return mZ; }
  /// set Z coordinate (cm)
  void setZ(Double_t z) { mZ = z; }
  /// return non bending coordinate (cm)
  Double_t getNonBendingCoor() const { return mParameters(0, 0); }
  /// set non bending coordinate (cm)
  void setNonBendingCoor(Double_t nonBendingCoor) { mParameters(0, 0) = nonBendingCoor; }
  /// return non bending slope (cm ** -1)
  Double_t getNonBendingSlope() const { return mParameters(1, 0); }
  /// set non bending slope (cm ** -1)
  void setNonBendingSlope(Double_t nonBendingSlope) { mParameters(1, 0) = nonBendingSlope; }
  /// return bending coordinate (cm)
  Double_t getBendingCoor() const { return mParameters(2, 0); }
  /// set bending coordinate (cm)
  void setBendingCoor(Double_t bendingCoor) { mParameters(2, 0) = bendingCoor; }
  /// return bending slope (cm ** -1)
  Double_t getBendingSlope() const { return mParameters(3, 0); }
  /// set bending slope (cm ** -1)
  void setBendingSlope(Double_t bendingSlope) { mParameters(3, 0) = bendingSlope; }
  /// return inverse bending momentum (GeV/c ** -1) times the charge (assumed forward motion)
  Double_t getInverseBendingMomentum() const { return mParameters(4, 0); }
  /// set inverse bending momentum (GeV/c ** -1) times the charge (assumed forward motion)
  void setInverseBendingMomentum(Double_t inverseBendingMomentum) { mParameters(4, 0) = inverseBendingMomentum; }
  /// return the charge (assumed forward motion)
  Double_t getCharge() const { return TMath::Sign(1., mParameters(4, 0)); }
  /// set the charge (assumed forward motion)
  void setCharge(Double_t charge)
  {
    if (charge * mParameters(4, 0) < 0.)
      mParameters(4, 0) *= -1.;
  }

  /// return track parameters
  const TMatrixD& getParameters() const { return mParameters; }
  /// set track parameters
  void setParameters(const TMatrixD& parameters) { mParameters = parameters; }
  /// add track parameters
  void addParameters(const TMatrixD& parameters) { mParameters += parameters; }

  Double_t px() const; // return px
  Double_t py() const; // return py
  Double_t pz() const; // return pz
  Double_t p() const;  // return total momentum

  /// return kTRUE if the covariance matrix exist, kFALSE if not
  Bool_t hasCovariances() const { return (mCovariances) ? kTRUE : kFALSE; }

  const TMatrixD& getCovariances() const;
  void setCovariances(const TMatrixD& covariances);
  void setCovariances(const Double_t matrix[5][5]);
  void setVariances(const Double_t matrix[5][5]);
  void deleteCovariances();

  const TMatrixD& getPropagator() const;
  void resetPropagator();
  void updatePropagator(const TMatrixD& propagator);

  const TMatrixD& getExtrapParameters() const;
  void setExtrapParameters(const TMatrixD& parameters);

  const TMatrixD& getExtrapCovariances() const;
  void setExtrapCovariances(const TMatrixD& covariances);

  const TMatrixD& getSmoothParameters() const;
  void setSmoothParameters(const TMatrixD& parameters);

  const TMatrixD& getSmoothCovariances() const;
  void setSmoothCovariances(const TMatrixD& covariances);

  /// get pointer to associated cluster
  const Cluster* getClusterPtr() const { return mClusterPtr; }
  /// set pointer to associated cluster
  void setClusterPtr(const Cluster* cluster) { mClusterPtr = cluster; }

  /// return true if the associated cluster can be removed from the track it belongs to
  Bool_t isRemovable() const { return mRemovable; }
  /// set the flag telling whether the associated cluster can be removed from the track it belongs to or not
  void setRemovable(Bool_t removable) { mRemovable = removable; }

  /// return the chi2 of the track when the associated cluster was attached
  Double_t getTrackChi2() const { return mTrackChi2; }
  /// set the chi2 of the track when the associated cluster was attached
  void setTrackChi2(Double_t chi2) { mTrackChi2 = chi2; }
  /// return the local chi2 of the associated cluster with respect to the track
  Double_t getLocalChi2() const { return mLocalChi2; }
  /// set the local chi2 of the associated cluster with respect to the track
  void setLocalChi2(Double_t chi2) { mLocalChi2 = chi2; }

  TrackParamStruct getTrackParamStruct() const;

  Bool_t isCompatibleTrackParam(const TrackParam& trackParam, Double_t sigma2Cut, Double_t& normChi2) const;

  void print() const;

  void clear();

 private:
  Double_t mZ = 0.; ///< Z coordinate (cm)

  /// Track parameters ordered as follow:      <pre>
  /// X       = Non bending coordinate   (cm)
  /// SlopeX  = Non bending slope        (cm ** -1)
  /// Y       = Bending coordinate       (cm)
  /// SlopeY  = Bending slope            (cm ** -1)
  /// InvP_yz = Inverse bending momentum (GeV/c ** -1) times the charge (assumed forward motion)  </pre>
  TMatrixD mParameters{5, 1}; ///< \brief Track parameters

  /// Covariance matrix of track parameters, ordered as follow:      <pre>
  ///    <X,X>      <X,SlopeX>        <X,Y>      <X,SlopeY>       <X,InvP_yz>
  /// <X,SlopeX>  <SlopeX,SlopeX>  <Y,SlopeX>  <SlopeX,SlopeY>  <SlopeX,InvP_yz>
  ///    <X,Y>      <Y,SlopeX>        <Y,Y>      <Y,SlopeY>       <Y,InvP_yz>
  /// <X,SlopeY>  <SlopeX,SlopeY>  <Y,SlopeY>  <SlopeY,SlopeY>  <SlopeY,InvP_yz>
  /// <X,InvP_yz> <SlopeX,InvP_yz> <Y,InvP_yz> <SlopeY,InvP_yz> <InvP_yz,InvP_yz>  </pre>
  mutable std::unique_ptr<TMatrixD> mCovariances{}; ///< \brief Covariance matrix of track parameters

  /// Jacobian used to extrapolate the track parameters and covariances to the actual z position
  mutable std::unique_ptr<TMatrixD> mPropagator{};
  /// Track parameters extrapolated to the actual z position (not filtered by Kalman)
  mutable std::unique_ptr<TMatrixD> mExtrapParameters{};
  /// Covariance matrix extrapolated to the actual z position (not filtered by Kalman)
  mutable std::unique_ptr<TMatrixD> mExtrapCovariances{};

  mutable std::unique_ptr<TMatrixD> mSmoothParameters{};  ///< Track parameters obtained using smoother
  mutable std::unique_ptr<TMatrixD> mSmoothCovariances{}; ///< Covariance matrix obtained using smoother

  const Cluster* mClusterPtr = nullptr; ///< Pointer to the associated cluster if any

  Bool_t mRemovable = false; ///< kTRUE if the associated cluster can be removed from the track it belongs to

  Double_t mTrackChi2 = 0.; ///< Chi2 of the track when the associated cluster was attached
  Double_t mLocalChi2 = 0.; ///< Local chi2 of the associated cluster with respect to the track
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACKPARAM_H_
