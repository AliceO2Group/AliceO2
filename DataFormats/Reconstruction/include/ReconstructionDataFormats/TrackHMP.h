// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRACKHMP_H
#define ALICEO2_TRACKHMP_H

#include "TMath.h"
#include "TVector2.h"

#include "ReconstructionDataFormats/TrackParametrizationWithError.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

// #include "HMPIDBase/Param.h"                 // for param

namespace o2
{
namespace dataformats
{

class TrackHMP : public o2::track::TrackParCov
{
 public:
  TrackHMP();
  TrackHMP(const TrackHMP& t) = default;
  TrackHMP(const o2::track::TrackParCov& t) : o2::track::TrackParCov{t} {};
  //  TrackHMP(const o2::track::TrackParCov& t);
  TrackHMP& operator=(const o2::track::TrackParCov& t);
  ~TrackHMP() = default;

  Bool_t intersect(Double_t pnt[3], Double_t norm[3], double bz) const; //
  void propagate(Double_t len, std::array<float, 3>& x, std::array<float, 3>& p, double bz) const;

  // protected:
  //    Bool_t   Update(const AliCluster */*c*/, Double_t /*chi2*/, Int_t /*idx*/) {return 0;}
  //    Double_t GetPredictedChi2(const AliCluster */*c*/) const {return 0.;}
 private:
  ClassDef(TrackHMP, 1) // HMPID reconstructed tracks
};

} // namespace dataformats
} // namespace o2

#endif
