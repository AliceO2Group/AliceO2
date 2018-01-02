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

#include "ITSReconstruction/CA/Cluster.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace ITS
{
namespace CA
{
  struct Track
  {
    Track();
    Track(const track::TrackParCov& param, float chi2, \
        const std::array<int,7>& clusters);

    track::TrackParCov mParam;    ///< Barrel track parameterisation
    float mChi2 = 1.e27;                ///< Chi2
    std::array<int,7> mClusters = {-1}; ///< Cluster index on the ITS layers
    int mMClabel;                       ///< Monte Carlo label [temp]
  };

  inline Track::Track()
  {
    // Nothing to do
  }

  inline Track::Track(const track::TrackParCov& param, float chi2, \
      const std::array<int,7>& clusters) :
    mParam{param},
    mChi2{chi2},
    mClusters{clusters}
  {
    // Nothing to do
  }

}
}
}

#endif /* O2_ITSMFT_RECONSTRUCTION_CA_TRACK_H_ */

