/// \file CookedTrack.h
/// \brief Definition of the ITS cooked track
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACK_H
#define ALICEO2_ITS_COOKEDTRACK_H

#include <vector>

#include <TObject.h>

#include "DetectorsBase/Track.h"

namespace o2
{
namespace ITS
{
class Cluster;

class CookedTrack : public TObject 
{
 public:
  CookedTrack();
  CookedTrack(float x, float alpha, const std::array<float,o2::Base::Track::kNParams> &par, const std::array<float,o2::Base::Track::kCovMatSize> &cov);
  CookedTrack(const CookedTrack& t);
  CookedTrack& operator=(const CookedTrack& tr);
  ~CookedTrack() override;

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
  void setExternalClusterIndex(Int_t layer, Int_t idx);
  void resetClusters();

  void setLabel(Int_t lab) { mLabel = lab; }
  Int_t getLabel() const { return mLabel; }
  Bool_t isBetter(const CookedTrack& best, Double_t maxChi2) const;

  Double_t getCurvature(Double_t bz) const { return mTrack.GetCurvature(float(bz)); }
  Double_t getAlpha() const { return mTrack.GetAlpha(); }
  Double_t getX() const { return mTrack.GetX(); }
  Double_t getY() const { return mTrack.GetY(); }
  Double_t getZ() const { return mTrack.GetZ(); }
  Double_t getSnp() const { return mTrack.GetSnp(); }
  Double_t getTgl() const { return mTrack.GetTgl(); }
  Double_t getPt() const { return mTrack.GetPt(); }
  Bool_t getPxPyPz(std::array<float,3> &pxyz) const { return mTrack.GetPxPyPz(pxyz); }
  void resetCovariance(Double_t s2 = 0.) { mTrack.ResetCovariance(float(s2)); }
  
 private:
  o2::Base::Track::TrackParCov mTrack; ///< Base track
  Int_t mLabel;              ///< Monte Carlo label for this track
  Double_t mMass;            ///< Assumed mass for this track
  Double_t mChi2;            ///< Chi2 for this track
  std::vector<Int_t> mIndex; ///< Indices of associated clusters

  ClassDefOverride(CookedTrack, 1)
};
}
}
#endif /* ALICEO2_ITS_COOKEDTRACK_H */
