// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Use a hash to sort tracks into a 2D histogram. The hash is used to
//         create pairs of tracks from the same hash bin with function selfCombinations.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/ASoAHelpers.h"

namespace o2::aod
{
namespace hash
{
DECLARE_SOA_COLUMN(Bin, bin, int);
} // namespace hash
DECLARE_SOA_TABLE(Hashes, "AOD", "HASH", hash::Bin);

} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::soa;

struct TrackCombinations {
  void process(aod::Tracks const& tracks)
  {
    // Strictly upper tracks
    for (auto& [t0, t1] : combinations(tracks, tracks)) {
      LOGF(info, "Tracks pair: %d %d", t0.index(), t1.index());
    }
  }
};

struct HashTask {
  std::vector<float> xBins{-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  std::vector<float> yBins{-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  Produces<aod::Hashes> hashes;

  // Calculate hash for an element based on 2 properties and their bins.
  int getHash(const std::vector<float>& xBins, const std::vector<float>& yBins, float colX, float colY)
  {
    for (int i = 0; i < xBins.size(); i++) {
      if (colX < xBins[i]) {
        for (int j = 0; j < yBins.size(); j++) {
          if (colY < yBins[j]) {
            return i + j * (xBins.size() + 1);
          }
        }
        // overflow for yBins only
        return i + yBins.size() * (xBins.size() + 1);
      }
    }

    // overflow for xBins only
    for (int j = 0; j < yBins.size(); j++) {
      if (colY < yBins[j]) {
        return xBins.size() + j * (xBins.size() + 1);
      }
    }

    // overflow for both bins
    return (yBins.size() + 1) * (xBins.size() + 1) - 1;
  }

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      int hash = getHash(xBins, yBins, track.x(), track.y());
      LOGF(info, "Track: %d (%f, %f, %f) hash: %d", track.index(), track.x(), track.y(), track.z(), hash);
      hashes(hash);
    }
  }
};

struct BinnedTrackCombinations {
  void process(soa::Join<aod::Hashes, aod::Tracks> const& hashedTracks)
  {
    // Strictly upper categorised tracks
    for (auto& [t0, t1] : selfCombinations("fBin", 5, -1, hashedTracks, hashedTracks)) {
      LOGF(info, "Tracks bin: %d pair: %d (%f, %f, %f), %d (%f, %f, %f)", t0.bin(), t0.index(), t0.x(), t0.y(), t0.z(), t1.index(), t1.x(), t1.y(), t1.z());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<TrackCombinations>(cfgc),
    adaptAnalysisTask<HashTask>(cfgc),
    adaptAnalysisTask<BinnedTrackCombinations>(cfgc),
  };
}
