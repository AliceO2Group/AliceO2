/// \file CookedTrack.h
/// \brief Definition of the ITS cooked track
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACK_H
#define ALICEO2_ITS_COOKEDTRACK_H

#include <vector>

#include <Rtypes.h>

#include "DetectorsBase/Track.h"

using AliceO2::Base::Track::TrackParCov;

namespace AliceO2
{
namespace ITS
{
class Cluster;

class CookedTrack : public TrackParCov
{
 public:
  CookedTrack();
  CookedTrack(float x, float alpha, const float* par, const float* cov);
  CookedTrack(const CookedTrack& t);
  CookedTrack& operator=(const CookedTrack& tr);
  virtual ~CookedTrack();

  // These functions must be provided
  Double_t getPredictedChi2(const Cluster* c) const;
  Bool_t propagate(Double_t alpha, Double_t x, Double_t bz);
  Bool_t correctForMeanMaterial(Double_t x2x0, Double_t xrho, Bool_t anglecorr = kTRUE);
  Bool_t update(const Cluster* c, Double_t chi2, Int_t idx);

  // Other functions
  Int_t getChi2() const { return mChi2; }
  Int_t getNumberOfClusters() const { return mIndex.size(); }
  Int_t getClusterIndex(Int_t i) const { return mIndex[i]; }
  bool operator<(const CookedTrack& o) const;
  void getImpactParams(Double_t x, Double_t y, Double_t z, Double_t bz, Double_t ip[2]) const;
  // Bool_t getPhiZat(Double_t r,Double_t &phi,Double_t &z) const;

  void setClusterIndex(Int_t layer, Int_t index);
  void resetClusters();

  void setLabel(Int_t lab) { mLabel = lab; }
  Int_t getLabel() const { return mLabel; }
  Bool_t isBetter(const CookedTrack& best, Double_t maxChi2) const;

  // The fuctions below address the naming conventions in the base class
  Double_t getX() const { return GetX(); }
  Double_t getY() const { return GetY(); }
  Double_t getZ() const { return GetZ(); }
  void resetCovariance(Float_t s2 = 0.) { ResetCovariance(s2); }
 private:
  Int_t mLabel;              ///< Monte Carlo label for this track
  Double_t mMass;            ///< Assumed mass for this track
  Double_t mChi2;            ///< Chi2 for this track
  std::vector<Int_t> mIndex; ///< Indices of associated clusters

  ClassDef(CookedTrack, 1)
};
}
}
#endif /* ALICEO2_ITS_COOKEDTRACK_H */
