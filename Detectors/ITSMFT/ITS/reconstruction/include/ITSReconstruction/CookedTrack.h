// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CookedTrack.h
/// \brief Definition of the ITS cooked track
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACK_H
#define ALICEO2_ITS_COOKEDTRACK_H

#include <vector>

#include "DetectorsBase/Track.h"

namespace o2
{

namespace ITSMFT {
class Cluster;
}

namespace ITS
{

class CookedTrack : public o2::Base::Track::TrackParCov
{
  using Cluster = o2::ITSMFT::Cluster;

 public:

  using o2::Base::Track::TrackParCov::TrackParCov; // inherit base constructors
  static constexpr int MaxClusters = 7;

  CookedTrack() = default;
  CookedTrack(const CookedTrack& t) = default;
  CookedTrack& operator=(const CookedTrack& tr) = default;
  ~CookedTrack()=default;

  // These functions must be provided
  Bool_t propagate(Float_t alpha, Float_t x, Float_t bz);
  Bool_t update(const Cluster& c, Float_t chi2, Int_t idx);

  // Other functions
  Int_t getChi2() const { return mChi2; }
  Int_t getNumberOfClusters() const { return mNClusters; }
  Int_t getClusterIndex(Int_t i) const { return mIndex[i]; }
  bool operator<(const CookedTrack& o) const;
  void getImpactParams(Float_t x, Float_t y, Float_t z, Float_t bz, Float_t ip[2]) const;
  // Bool_t getPhiZat(Float_t r,Float_t &phi,Float_t &z) const;

  void setClusterIndex(Int_t layer, Int_t index);
  void setExternalClusterIndex(Int_t layer, Int_t idx);
  void resetClusters();

  std::uint32_t getROFrame() const {return mROFrame;}
  void setROFrame(std::uint32_t f) {mROFrame = f;}
  
  Bool_t isBetter(const CookedTrack& best, Float_t maxChi2) const;

  o2::Base::Track::TrackParCov &getParamOut() {return mParamOut;}
  const o2::Base::Track::TrackParCov &getParamOut() const {return mParamOut;}
  
 private:

  short mNClusters = 0;
  float mMass = 0.14;            ///< Assumed mass for this track
  float mChi2 = 0.;            ///< Chi2 for this track
  std::uint32_t mROFrame=0;    ///< RO Frame
  o2::Base::Track::TrackParCov mParamOut; // parameter at largest radius
  std::array<Int_t,MaxClusters> mIndex; ///< Indices of associated clusters
  
  ClassDef(CookedTrack, 2)
};
}
}
#endif /* ALICEO2_ITS_COOKEDTRACK_H */
