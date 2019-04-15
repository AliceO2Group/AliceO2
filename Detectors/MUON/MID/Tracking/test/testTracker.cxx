// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Tracking/test/testTracker.cxx
/// \brief  Test tracking device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   15 March 2018

#define BOOST_TEST_MODULE midTracking
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <cstdint>
#include <sstream>
#include <vector>
#include <random>
#include "DataFormatsMID/Cluster2D.h"
#include "DataFormatsMID/Track.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/MpArea.h"
#include "MIDBase/HitFinder.h"
#include "MIDTestingSimTools/TrackGenerator.h"
#include "MIDTracking/Tracker.h"

namespace o2
{
namespace mid
{

struct TrackClusters {
  Track track;
  std::vector<Cluster2D> clusters;
  int nFiredChambers;
  bool isReconstructible() { return nFiredChambers > 2; }
};

struct MyFixture {
  static GeometryTransformer geoTrans;
  static HitFinder hitFinder;
  static TrackGenerator trackGen;
  static Tracker tracker;
  static Mapping mapping;
};

GeometryTransformer MyFixture::geoTrans = createDefaultTransformer();
HitFinder MyFixture::hitFinder(geoTrans);
TrackGenerator MyFixture::trackGen;
Tracker MyFixture::tracker(geoTrans);
Mapping MyFixture::mapping;

TrackClusters getTrackClusters(const Track& track)
{
  TrackClusters trCl;
  trCl.track = track;
  trCl.nFiredChambers = 0;
  Mapping::MpStripIndex stripIndex;
  MpArea area;
  Cluster2D cl;
  for (int ich = 0; ich < 4; ++ich) {
    std::vector<std::pair<int, Point3D<float>>> pairs = MyFixture::hitFinder.getLocalPositions(track, ich);
    bool isFired = false;
    for (auto& pair : pairs) {
      int deId = pair.first;
      float xPos = pair.second.x();
      float yPos = pair.second.y();
      stripIndex = MyFixture::mapping.stripByPosition(xPos, yPos, 0, deId, false);
      if (!stripIndex.isValid()) {
        continue;
      }
      cl.deId = deId;
      area = MyFixture::mapping.stripByLocation(stripIndex.strip, 0, stripIndex.line, stripIndex.column, deId);
      cl.yCoor = area.getCenterY();
      float halfSize = area.getHalfSizeY();
      cl.sigmaY2 = halfSize * halfSize / 3.;
      stripIndex = MyFixture::mapping.stripByPosition(xPos, yPos, 1, deId, false);
      area = MyFixture::mapping.stripByLocation(stripIndex.strip, 1, stripIndex.line, stripIndex.column, deId);
      cl.xCoor = area.getCenterX();
      halfSize = area.getHalfSizeX();
      cl.sigmaX2 = halfSize * halfSize / 3.;
      trCl.clusters.push_back(cl);
      isFired = true;
    } // loop on fired pos
    if (isFired) {
      ++(trCl.nFiredChambers);
    }
  }
  return trCl;
}

std::vector<TrackClusters> getTrackClusters(int nTracks)
{
  std::vector<TrackClusters> trackClusters;
  std::vector<Track> tracks = MyFixture::trackGen.generate(nTracks);
  for (auto& track : tracks) {
    trackClusters.push_back(getTrackClusters(track));
  }
  return trackClusters;
}

BOOST_DATA_TEST_CASE_F(MyFixture, TestMultipleTracks, boost::unit_test::data::xrange(1, 9), nTracksPerEvent)
{
  float chi2Cut = tracker.getSigmaCut() * tracker.getSigmaCut();
  int nTotFakes = 0, nTotReconstructible = 0;
  for (int ievt = 0; ievt < 1000; ++ievt) {
    std::vector<TrackClusters> trackClusters = getTrackClusters(nTracksPerEvent);
    std::vector<Cluster2D> clusters;

    // Fill string for debugging
    std::stringstream ss;
    ss << "\n";
    int itr = -1;
    for (auto& trCl : trackClusters) {
      ++itr;
      ss << "Track " << itr << ": " << trCl.track << "\n";
      for (auto& cl : trCl.clusters) {
        clusters.push_back(cl);
        ss << "  deId " << (int)cl.deId << " pos: (" << cl.xCoor << ", " << cl.yCoor << ")";
      }
      ss << "\n";
    }

    // Run tracker algorithm
    tracker.process(clusters);

    // Further strings for debugging
    ss << "  Reconstructed tracks:\n";
    for (size_t ireco = 0; ireco < tracker.getTracks().size(); ++ireco) {
      ss << "  " << tracker.getTracks()[ireco] << "\n";
    }

    // Check that all reconstructible tracks are reconstructed
    size_t nReconstructible = 0;
    itr = -1;
    for (auto& trCl : trackClusters) {
      ++itr;
      bool isReco = false;
      for (size_t ireco = 0; ireco < tracker.getTracks().size(); ++ireco) {
        if (tracker.getTracks()[ireco].isCompatible(trCl.track, chi2Cut)) {
          isReco = true;
          break;
        }
      }
      bool isReconstructible = trCl.isReconstructible();
      // If the number of tracks is small, we can check that:
      // 1) all reconstructible tracks are reconstructed
      // 2) all non-reconstructible tracks are not
      // Case 2), however, is not always valid when we have many tracks,
      // since we can combine clusters from different tracks to build fakes
      bool testTrack = (isReconstructible || nTracksPerEvent < 4);
      if (testTrack) {
        BOOST_TEST(isReco == isReconstructible, ss.str() << "  track " << itr << " reco " << isReco
                                                         << " != reconstructible " << isReconstructible);
      }
      if (isReconstructible) {
        ++nReconstructible;
      }
    } // loop on input tracks
    nTotReconstructible += nReconstructible;
    int nFakes = tracker.getTracks().size() - nReconstructible;
    if (nFakes > 0) {
      ++nTotFakes;
    }
  } // loop on events
  // To show the following message, run the test with: --log_level=message
  BOOST_TEST_MESSAGE("Fraction of fake tracks: " << (double)nTotFakes / (double)nTotReconstructible);
}

} // namespace mid
} // namespace o2
