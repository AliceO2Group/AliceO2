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

  // Charge and momentum from quadratic regression of clusters X,Y positions
  void setInvQPtQuadtratic(Double_t invqpt) { mInvQPtQuadtratic = invqpt; }
  const Double_t getInvQPtQuadtratic() const { return mInvQPtQuadtratic; } // Inverse charged pt
  const Double_t getPtQuadtratic() const { return TMath::Abs(1.f / getInvQPtQuadtratic()); }
  const Double_t getChargeQuadratic() const { return TMath::Sign(1., getInvQPtQuadtratic()); }
  void setChi2QPtQuadtratic(Double_t chi2) { mQuadraticFitChi2 = chi2; }
  const Double_t getChi2QPtQuadtratic() const { return mQuadraticFitChi2; }

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

  // Parameters and Covariances on last track clusters
  const SMatrix5& getParametersLast() const { return mParametersLast; }
  void setParametersLast(const SMatrix5& parameters) { mParametersLast = parameters; } // Last cluster
  const SMatrix55& getCovariancesLast() const;
  void setCovariancesLast(const SMatrix55& covariances);

  Double_t getZLast() const { return mZLast; }
  void setZLast(Double_t z) { mZLast = z; }
  Double_t getXLast() const { return mParametersLast(0); }
  Double_t getYLast() const { return mParametersLast(1); }
  Double_t getPhiLast() const { return mParametersLast(2); }
  Double_t getTanlLast() const { return mParametersLast(3); }
  Double_t getInvQPtLast() const { return mParametersLast(4); }
  Double_t getPtLast() const { return TMath::Abs(1.f / mParametersLast(4)); }
  Double_t getInvPtLast() const { return TMath::Abs(mParametersLast(4)); }
  Double_t getPLast() const { return getPtLast() * TMath::Sqrt(1. + getTanlLast() * getTanlLast()); }                  // return total momentum last cluster
  Double_t getEtaLast() const { return -TMath::Log(TMath::Tan((TMath::PiOver2() - TMath::ATan(getTanlLast())) / 2)); } // return total momentum

 private:
  std::uint32_t mROFrame = 0;                ///< RO Frame
  Int_t mNPoints{0};                         // Number of clusters
  std::array<MCCompLabel, 10> mMCCompLabels; // constants::mft::LayersNumber = 10
  Bool_t mIsCA = false;                      // Track finding method CA vs. LTF

  ClusRefs mClusRef; ///< references on clusters

  Double_t mZLast = 0.; ///< Z coordinate (cm) of Last cluster

  SMatrix5 mParametersLast; ///< \brief Track parameters at last cluster
  SMatrix55 mCovariancesLast; ///< \brief Covariance matrix of track parameters at last cluster


  // Results from quadratic regression of clusters X,Y positions
  // Chi2 of the quadratic regression used to estimate track pT and charge
  Double_t mQuadraticFitChi2 = 0.;
  // inversed charged momentum from quadratic regression
  Double_t mInvQPtQuadtratic;

  ClassDefNV(TrackMFT, 1);
};

class TrackMFTExt : public TrackMFT
{
  ///< heavy version of TrackMFT, with clusters embedded
 public:
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
