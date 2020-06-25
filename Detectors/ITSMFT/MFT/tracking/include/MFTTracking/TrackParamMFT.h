// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackParamMFT.h
/// \brief Definition of the MFT track parameters for internal use
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#ifndef ALICEO2_MFT_TRACKPARAMMFT_H_
#define ALICEO2_MFT_TRACKPARAMMFT_H_

#include <TMatrixD.h>
#include <TMath.h>

#include "MFTBase/Constants.h"
#include "MFTTracking/Cluster.h"

namespace o2
{
namespace mft
{


/// track parameters for internal use
class TrackParamMFT
{
 public:
  TrackParamMFT() = default;
  ~TrackParamMFT() = default;

  TrackParamMFT(const TrackParamMFT& tp);
  TrackParamMFT& operator=(const TrackParamMFT& tp);
  TrackParamMFT(TrackParamMFT&&) = delete;
  TrackParamMFT& operator=(TrackParamMFT&&) = delete;

  /// return Z coordinate (cm)
  Double_t getZ() const { return mZ; }
  /// set Z coordinate (cm)
  void setZ(Double_t z) { mZ = z; }
  Double_t getX() const { return mParameters(0, 0); }
  void setX(Double_t x) { mParameters(0, 0) = x; }

  Double_t getY() const { return mParameters(1, 0); }
  void setY(Double_t y) { mParameters(1, 0) = y; }

  void setPhi(Double_t phi) { mParameters(2, 0) = phi; }
  Double_t getPhi() const { return mParameters(2, 0); }

  void setTanl(Double_t tanl) { mParameters(3, 0) = tanl; }
  Double_t getTanl() const { return mParameters(3, 0); }

  void setInvQPt(Double_t invqpt) { mParameters(4, 0) = invqpt; }
  Double_t getInvQPt() const { return mParameters(4, 0); } // return Inverse charged pt
  Double_t getPt() const { return TMath::Abs(1.f / mParameters(4, 0)); }
  Double_t getInvPt() const { return TMath::Abs(mParameters(4, 0)); }

  Double_t getPx() const { return TMath::Cos(getPhi()) * getPt(); } // return px
  Double_t getInvPx() const { return 1. / getPx(); }                // return invpx

  Double_t getPy() const { return TMath::Sin(getPhi()) * getPt(); } // return py
  Double_t getInvPy() const { return 1. / getPx(); }                // return invpy

  Double_t getPz() const { return getTanl() * getPt(); } // return pz
  Double_t getInvPz() const { return 1. / getPz(); }     // return invpz

  Double_t getP() const { return getPt() * TMath::Sqrt(1. + getTanl() * getTanl()); } // return total momentum
  Double_t getInverseMomentum() const { return 1.f / getP(); }

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

  Bool_t isCompatibleTrackParamMFT(const TrackParamMFT& TrackParamMFT, Double_t sigma2Cut, Double_t& normChi2) const;

  void print() const;

  void clear();

 private:
  Double_t mZ = 0.; ///< Z coordinate (cm)

  /// Track parameters ordered as follow:      <pre>
  /// X       = X coordinate   (cm)
  /// Y       = Y coordinate   (cm)
  /// PHI     = azimutal angle
  /// TANL    = tangent of \lambda (dip angle)
  /// INVQPT    = Inverse transverse momentum (GeV/c ** -1) times charge (assumed forward motion)  </pre>
  TMatrixD mParameters{5, 1}; ///< \brief Track parameters

  /// Covariance matrix of track parameters, ordered as follows:    <pre>
  ///  <X,X>         <Y,X>           <PHI,X>       <TANL,X>        <INVQPT,X>
  ///  <X,Y>         <Y,Y>           <PHI,Y>       <TANL,Y>        <INVQPT,Y>
  /// <X,PHI>       <Y,PHI>         <PHI,PHI>     <TANL,PHI>      <INVQPT,PHI>
  /// <X,TANL>      <Y,TANL>       <PHI,TANL>     <TANL,TANL>     <INVQPT,TANL>
  /// <X,INVQPT>   <Y,INVQPT>     <PHI,INVQPT>   <TANL,INVQPT>   <INVQPT,INVQPT>  </pre>
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

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_TRACKPARAMMFT_H_
