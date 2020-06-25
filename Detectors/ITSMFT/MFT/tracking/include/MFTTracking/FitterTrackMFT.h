// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FitterTrackMFT.h
/// \brief Definition of the MFT track for internal use by the fitter
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#ifndef ALICEO2_MFT_FITTERTRACK_H_
#define ALICEO2_MFT_FITTERTRACK_H_

#include <list>
#include <memory>

#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackParamMFT.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace mft
{

/// track for internal use byt the MFT Track Fitter
class FitterTrackMFT
{
 public:
  FitterTrackMFT() = default;
  ~FitterTrackMFT() = default;

  FitterTrackMFT(const FitterTrackMFT& track);
  FitterTrackMFT& operator=(const FitterTrackMFT& track) = delete;
  //FitterTrackMFT(FitterTrackMFT&&) = delete;
  FitterTrackMFT& operator=(FitterTrackMFT&&) = delete;

  /// Return a reference to the track parameters at vertex
  const TrackParamMFT& getParamAtVertex() const { return mParamAtVertex; }

  /// Return the number of attached clusters
  const int getNClusters() const { return mParamAtClusters.size(); }

  // Set and Get MCCompLabels
  const std::array<MCCompLabel, 10>& getMCCompLabels() const { return mMCCompLabels; } // constants::mft::LayersNumber = 10
  void setMCCompLabels(const std::array<MCCompLabel, 10>& labels, int nPoints)
  {
    mMCCompLabels = labels;
    mNPoints = nPoints;
  }

  /// Return a reference to the track parameters at first cluster
  const TrackParamMFT& first() const { return mParamAtClusters.front(); }
  /// Return a reference to the track parameters at last cluster
  const TrackParamMFT& last() const { return mParamAtClusters.back(); }

  /// Return an iterator to the track parameters at clusters (point to the first one)
  auto begin() { return mParamAtClusters.begin(); }
  auto begin() const { return mParamAtClusters.begin(); }
  /// Return an iterator passing the track parameters at last cluster
  auto end() { return mParamAtClusters.end(); }
  auto end() const { return mParamAtClusters.end(); }
  /// Return a reverse iterator to the track parameters at clusters (point to the last one)
  auto rbegin() { return mParamAtClusters.rbegin(); }
  auto rbegin() const { return mParamAtClusters.rbegin(); }
  /// Return a reverse iterator passing the track parameters at first cluster
  auto rend() { return mParamAtClusters.rend(); }
  auto rend() const { return mParamAtClusters.rend(); }

  TrackParamMFT& createParamAtCluster(const Cluster& cluster);
  void addParamAtCluster(const TrackParamMFT& param);
  /// Remove the given track parameters from the internal list and return an iterator to the parameters that follow
  auto removeParamAtCluster(std::list<TrackParamMFT>::iterator& itParam) { return mParamAtClusters.erase(itParam); }

  bool isBetter(const FitterTrackMFT& track) const;

  void tagRemovableClusters(uint8_t requestedStationMask);

  void setCurrentParam(const TrackParamMFT& param, int chamber);
  TrackParamMFT& getCurrentParam();
  /// get a reference to the current chamber on which the current parameters are given
  const int& getCurrentChamber() const { return mCurrentLayer; }
  /// check whether the current track parameters exist
  bool hasCurrentParam() const { return mCurrentParam ? true : false; }
  /// check if the current parameters are valid
  bool areCurrentParamValid() const { return (mCurrentLayer > -1); }
  /// invalidate the current parameters
  void invalidateCurrentParam() { mCurrentLayer = -1; }

  /// set the flag telling if this track shares cluster(s) with another
  void connected(bool connected = true) { mConnected = connected; }
  /// return the flag telling if this track shares cluster(s) with another
  bool isConnected() const { return mConnected; }

  /// set the flag telling if this track should be deleted
  void removable(bool removable = true) { mRemovable = removable; }
  /// return the flag telling if this track should be deleted
  bool isRemovable() const { return mRemovable; }

  const Int_t getNPoints() const { return mNPoints; }

  // Charge and momentum from quadratic regression of clusters X,Y positions
  void setInvQPtQuadtratic(Double_t invqpt) { mInvQPtQuadtratic = invqpt; }
  const Double_t getInvQPtQuadtratic() const { return mInvQPtQuadtratic; } // Inverse charged pt
  const Double_t getPtQuadtratic() const { return TMath::Abs(1.f / getInvQPtQuadtratic()); }
  const Double_t getChargeQuadratic() const { return TMath::Sign(1., getInvQPtQuadtratic()); }
  void setChi2QPtQuadtratic(Double_t chi2) { mQuadraticFitChi2 = chi2; }
  const Double_t getChi2QPtQuadtratic() const { return mQuadraticFitChi2; }

 private:
  TrackParamMFT mParamAtVertex{};                 ///< track parameters at vertex
  std::list<TrackParamMFT> mParamAtClusters{};    ///< list of track parameters at each cluster
  std::unique_ptr<TrackParamMFT> mCurrentParam{}; ///< current track parameters used during tracking
  int mCurrentLayer = -1;                         ///< current chamber on which the current parameters are given
  bool mConnected = false;                        ///< flag telling if this track shares cluster(s) with another
  bool mRemovable = false;                        ///< flag telling if this track should be deleted
  Int_t mNPoints{0};                              // Number of clusters
  std::array<MCCompLabel, 10> mMCCompLabels;      // constants::mft::LayersNumber = 10

  // Results from quadratic regression of clusters X,Y positions
  // Chi2 of the quadratic regression used to estimate track pT and charge
  Double_t mQuadraticFitChi2 = 0.;
  // inversed charged momentum from quadratic regression
  Double_t mInvQPtQuadtratic;
};

} // namespace mft
} // namespace o2

#endif // ALICEO2_MFT_FITTERTRACK_H_
