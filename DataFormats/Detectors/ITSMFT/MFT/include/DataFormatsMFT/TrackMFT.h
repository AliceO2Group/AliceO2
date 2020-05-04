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

namespace o2
{

namespace itsmft
{
class Cluster;
}

namespace mft
{
class TrackMFT
{
  using Cluster = o2::itsmft::Cluster;
  using ClusRefs = o2::dataformats::RangeRefComp<4>;
  using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
  using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

 public:
  TrackMFT() = default;
  TrackMFT(const TrackMFT& t) = default;
  TrackMFT(const Double_t Z, const SMatrix5 parameters, const SMatrix55 covariances, const Double_t chi2);

  ~TrackMFT() = default;

  /// return Z coordinate (cm)
  Double_t getZ() const { return mZ; }
  /// set Z coordinate (cm)
  void setZ(Double_t z) { mZ = z; }
  Double_t getX() const { return mParameters(0); }
  void setX(Double_t x) { mParameters(0) = x; }
  Double_t getSigmaX() const { return mCovariances(0, 0); }

  Double_t getY() const { return mParameters(1); }
  void setY(Double_t y) { mParameters(1) = y; }
  Double_t getSigmaY() const { return mCovariances(1, 1); }

  void setPhi(Double_t phi) { mParameters(2) = phi; }
  Double_t getPhi() const { return mParameters(2); }
  Double_t getSigmaPhi() const { return mCovariances(2, 2); }

  void setTanl(Double_t tanl) { mParameters(3) = tanl; }
  Double_t getTanl() const { return mParameters(3); }
  Double_t getSigmaTanl() const { return mCovariances(3, 3); }

  void setInvQPt(Double_t invqpt) { mParameters(4) = invqpt; }
  Double_t getInvQPt() const { return mParameters(4); } // return Inverse charged pt
  Double_t getPt() const { return TMath::Abs(1.f / mParameters(4)); }
  Double_t getInvPt() const { return TMath::Abs(mParameters(4)); }
  Double_t getSigmaInvQPt() const { return mCovariances(4, 4); }

  // Charge and momentum from quadratic regression of clusters X,Y positions
  void setInvQPtQuadtratic(Double_t invqpt) { mInvQPtQuadtratic = invqpt; }
  const Double_t getInvQPtQuadtratic() const { return mInvQPtQuadtratic; } // Inverse charged pt
  const Double_t getPtQuadtratic() const { return TMath::Abs(1.f / getInvQPtQuadtratic()); }
  const Double_t getChargeQuadratic() const { return TMath::Sign(1., getInvQPtQuadtratic()); }
  void setChi2QPtQuadtratic(Double_t chi2) { mQuadraticFitChi2 = chi2; }
  const Double_t getChi2QPtQuadtratic() const { return mQuadraticFitChi2; }

  Double_t getPx() const { return TMath::Cos(getPhi()) * getPt(); } // return px
  Double_t getInvPx() const { return 1. / getPx(); }                // return invpx

  Double_t getPy() const { return TMath::Sin(getPhi()) * getPt(); } // return py
  Double_t getInvPy() const { return 1. / getPx(); }                // return invpy

  Double_t getPz() const { return getTanl() * getPt(); } // return pz
  Double_t getInvPz() const { return 1. / getPz(); }     // return invpz

  Double_t getP() const { return getPt() * TMath::Sqrt(1. + getTanl() * getTanl()); } // return total momentum
  Double_t getInverseMomentum() const { return 1.f / getP(); }

  Double_t getEta() const { return -TMath::Log(TMath::Tan((TMath::PiOver2() - TMath::ATan(getTanl())) / 2)); } // return total momentum

  /// return the charge (assumed forward motion)
  Double_t getCharge() const { return TMath::Sign(1., mParameters(4)); }
  /// set the charge (assumed forward motion)
  void setCharge(Double_t charge)
  {
    if (charge * mParameters(4) < 0.)
      mParameters(4) *= -1.;
  }

  /// return track parameters
  const SMatrix5& getParameters() const { return mParameters; }
  /// set track parameters
  void setParameters(const SMatrix5& parameters) { mParameters = parameters; }

  const SMatrix55& getCovariances() const;
  void setCovariances(const SMatrix55& covariances);

  /// return the chi2 of the track when the associated cluster was attached
  Double_t getTrackChi2() const { return mTrackChi2; }
  /// set the chi2 of the track when the associated cluster was attached
  void setTrackChi2(Double_t chi2) { mTrackChi2 = chi2; }

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

  // Extrapolate this track to
  void extrapHelixToZ(double zEnd, double Field);

 private:
  std::uint32_t mROFrame = 0;                ///< RO Frame
  Int_t mNPoints{0};                         // Number of clusters
  std::array<MCCompLabel, 10> mMCCompLabels; // constants::mft::LayersNumber = 10

  ClusRefs mClusRef; ///< references on clusters

  Double_t mZ = 0.; ///< Z coordinate (cm)

  /// Track parameters ordered as follow:      <pre>
  /// X       = X coordinate   (cm)
  /// Y       = Y coordinate   (cm)
  /// PHI     = azimutal angle
  /// TANL    = tangent of \lambda (dip angle)
  /// INVQPT    = Inverse transverse momentum (GeV/c ** -1) times charge (assumed forward motion)  </pre>
  SMatrix5 mParameters; ///< \brief Track parameters

  /// Covariance matrix of track parameters, ordered as follows:    <pre>
  ///  <X,X>         <Y,X>           <PHI,X>       <TANL,X>        <INVQPT,X>
  ///  <X,Y>         <Y,Y>           <PHI,Y>       <TANL,Y>        <INVQPT,Y>
  /// <X,PHI>       <Y,PHI>         <PHI,PHI>     <TANL,PHI>      <INVQPT,PHI>
  /// <X,TANL>      <Y,TANL>       <PHI,TANL>     <TANL,TANL>     <INVQPT,TANL>
  /// <X,INVQPT>   <Y,INVQPT>     <PHI,INVQPT>   <TANL,INVQPT>   <INVQPT,INVQPT>  </pre>
  SMatrix55 mCovariances;   ///< \brief Covariance matrix of track parameters
  Double_t mTrackChi2 = 0.; ///< Chi2 of the track when the associated cluster was attached

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

  void setClusterIndex(int l, int i)
  {
    int ncl = getNumberOfClusters();
    mIndex[ncl++] = (l << 28) + i;
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
