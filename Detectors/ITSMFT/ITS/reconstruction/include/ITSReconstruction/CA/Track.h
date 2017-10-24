// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CA/Track.h
/// \brief Definition of the ITS CA track

#ifndef O2_ITSMFT_RECONSTRUCTION_CA_TRACK_H_
#define O2_ITSMFT_RECONSTRUCTION_CA_TRACK_H_

#include <TObject.h>

#include "ITSReconstruction/CA/Cluster.h"
#include "DetectorsBase/Track.h"

namespace o2
{
namespace ITS
{
namespace CA
{
  class Track
  {
  public:
    Track() {}
    Track(const Base::Track::TrackParCov& param, float chi2, const std::array<int,7>& clusters);
    Base::Track::TrackParCov mParam; ///< Standard barrel track parameterisation
    float mChi2 = 1.e27;                    ///< Chi2
    std::array<int,7> mClusters = {-1};     ///< Cluster index on the ITS layers
  };

  class TrackObject : public TObject
  { /// Just for storage
  public:
    TrackObject(const Track& track);
    ~TrackObject() override;
    Track mTrack;
    ClassDefOverride(TrackObject,1)
  };

}
}
}
#endif /* O2_ITSMFT_RECONSTRUCTION_CA_TRACK_H_ */