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

/// \file Track.h
/// \brief Definition of the MFT track
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 8, 2018

#ifndef ALICEO2_MFT_TRACKMFT_H
#define ALICEO2_MFT_TRACKMFT_H

#include <vector>
#include <TMath.h>
#include "Math/SMatrix.h"

#include "CommonDataFormat/RangeReference.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ReconstructionDataFormats/TrackFwd.h"

namespace o2
{

namespace mft
{
class TrackMFT : public o2::track::TrackParCovFwd
{
  using ClusRefs = o2::dataformats::RangeRefComp<4>;

 public:
  TrackMFT() = default;
  TrackMFT(const TrackMFT& t) = default;
  ~TrackMFT() = default;

  // Track finding method
  void setCA(Bool_t method = true) { mIsCA = method; }
  const Bool_t isCA() const { return mIsCA; }   ///< Track found by CA algorithm
  const Bool_t isLTF() const { return !mIsCA; } ///< Track found by Linear Track Finder

  // Tracking seed charge and momentum from Fast Circle Fit of clusters X,Y positions
  void setInvQPtSeed(Double_t invqpt) { mInvQPtSeed = invqpt; }
  const Double_t getInvQPtSeed() const { return mInvQPtSeed; } // Inverse charged pt
  const Double_t getPtSeed() const { return TMath::Abs(1.f / getInvQPtSeed()); }
  const Double_t getChargeSeed() const { return TMath::Sign(1., getInvQPtSeed()); }
  void setChi2QPtSeed(Double_t chi2) { mSeedinvQPtFitChi2 = chi2; }
  const Double_t getChi2QPtSeed() const { return mSeedinvQPtFitChi2; }

  const int getNumberOfPoints() const { return mClusRef.getEntries(); } //< Get number of clusters

  const int getExternalClusterIndexOffset() const { return mClusRef.getFirstEntry(); }

  void setExternalClusterIndexOffset(int offset = 0) { mClusRef.setFirstEntry(offset); }

  void setNumberOfPoints(int n) { mClusRef.setEntries(n); } ///< Set number of clusters

  void print() const;

  const o2::track::TrackParCovFwd& getOutParam() const { return mOutParameters; }       ///< Returns track parameters fitted outwards
  void setOutParam(const o2::track::TrackParCovFwd parcov) { mOutParameters = parcov; } ///< Set track out parameters

 private:
  Bool_t mIsCA = false; ///< Track finding method CA vs. LTF

  ClusRefs mClusRef; ///< Clusters references

  o2::track::TrackParCovFwd mOutParameters; ///< Outward parameters for MCH matching

  Double_t mSeedinvQPtFitChi2 = 0.; ///< Seed InvQPt Chi2 from FCF clusters X,Y positions
  Double_t mInvQPtSeed;             ///< Seed InvQPt from FCF clusters X,Y positions

  ClassDefNV(TrackMFT, 2);
};

class TrackMFTExt : public TrackMFT
{
  ///< heavy version of TrackMFT, with clusters embedded
 public:
  TrackMFTExt() = default;
  TrackMFTExt(const TrackMFTExt& t) = default;
  ~TrackMFTExt() = default;
  static constexpr int MaxClusters = 10;
  using TrackMFT::TrackMFT; // inherit base constructors

  int getExternalClusterIndex(int i) const { return mExtClsIndex[i]; }

  void setExternalClusterIndex(int np, int idx)
  {
    mExtClsIndex[np] = idx;
  }

 protected:
  std::array<int, MaxClusters> mExtClsIndex = {-1}; ///< External indices of associated clusters

  ClassDefNV(TrackMFTExt, 1);
};
} // namespace mft
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::mft::TrackMFT> : std::true_type {
};
} // namespace framework
} // namespace o2

#endif
