// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
  using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

 public:
  TrackMFT() = default;
  TrackMFT(const TrackMFT& t) = default;
  ~TrackMFT() = default;

  // Track finding method
  void setCA(Bool_t method = true) { mIsCA = method; }
  const Bool_t isCA() const { return mIsCA; }
  const Bool_t isLTF() const { return !mIsCA; }

  // Tracking seed charge and momentum from Fast Circle Fit of clusters X,Y positions
  void setInvQPtSeed(Double_t invqpt) { mInvQPtSeed = invqpt; }
  const Double_t getInvQPtSeed() const { return mInvQPtSeed; } // Inverse charged pt
  const Double_t getPtSeed() const { return TMath::Abs(1.f / getInvQPtSeed()); }
  const Double_t getChargeSeed() const { return TMath::Sign(1., getInvQPtSeed()); }
  void setChi2QPtSeed(Double_t chi2) { mSeedinvQPtFitChi2 = chi2; }
  const Double_t getChi2QPtSeed() const { return mSeedinvQPtFitChi2; }

  // Other functions
  int getNumberOfClusters() const { return mClusRef.getEntries(); }
  int getFirstClusterEntry() const { return mClusRef.getFirstEntry(); }
  int getClusterEntry(int i) const { return getFirstClusterEntry() + i; }
  void shiftFirstClusterEntry(int bias)
  {
    mClusRef.setFirstEntry(mClusRef.getFirstEntry() + bias);
  }
  void setFirstClusterEntry(int offs)
  {
    mClusRef.setFirstEntry(offs);
  }
  void setNumberOfClusters(int n)
  {
    mClusRef.setEntries(n);
  }
  void setClusterRefs(int firstEntry, int n)
  {
    mClusRef.set(firstEntry, n);
  }

  const std::array<MCCompLabel, 10>& getMCCompLabels() const { return mMCCompLabels; } // constants::mft::LayersNumber = 10
  void setMCCompLabels(const std::array<MCCompLabel, 10>& labels, int nPoints)
  {
    mMCCompLabels = labels;
    mNPoints = nPoints;
  }

  const ClusRefs& getClusterRefs() const { return mClusRef; }
  ClusRefs& getClusterRefs() { return mClusRef; }

  std::uint32_t getROFrame() const { return mROFrame; }
  void setROFrame(std::uint32_t f) { mROFrame = f; }

  const Int_t getNPoints() const { return mNPoints; }

  void print() const;
  void printMCCompLabels() const;

  const o2::track::TrackParCovFwd& GetOutParam() const { return mOutParameters; }
  void SetOutParam(const o2::track::TrackParCovFwd parcov) { mOutParameters = parcov; }

 private:
  std::uint32_t mROFrame = 0;                ///< RO Frame
  Int_t mNPoints{0};                         // Number of clusters
  std::array<MCCompLabel, 10> mMCCompLabels; // constants::mft::LayersNumber = 10
  Bool_t mIsCA = false;                      // Track finding method CA vs. LTF

  ClusRefs mClusRef; ///< references on clusters

  // Outward parameters for MCH matching
  o2::track::TrackParCovFwd mOutParameters;

  // Seed InveQPt and Chi2 from fitting clusters X,Y positions
  Double_t mSeedinvQPtFitChi2 = 0.;
  Double_t mInvQPtSeed;

  ClassDefNV(TrackMFT, 1);
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

  void setClusterIndex(int l, int i, int ncl)
  {
    //int ncl = getNumberOfClusters();
    mIndex[ncl] = (l << 28) + i;
    getClusterRefs().setEntries(ncl);
  }

  int getClusterIndex(int lr) const { return mIndex[lr]; }

  void setExternalClusterIndex(int layer, int idx, bool newCluster = false)
  {
    if (newCluster) {
      getClusterRefs().setEntries(getNumberOfClusters() + 1);
    }
    mIndex[layer] = idx;
  }

 private:
  std::array<int, MaxClusters> mIndex = {-1}; ///< Indices of associated clusters
  ClassDefNV(TrackMFTExt, 1);
};
} // namespace mft
} // namespace o2

#endif
