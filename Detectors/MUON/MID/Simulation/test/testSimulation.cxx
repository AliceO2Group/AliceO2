// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE midSimulation
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/test/data/test_case.hpp>
#include <sstream>
#include "MathUtils/Cartesian3D.h"
#include "DataFormatsMID/Cluster2D.h"
#include "DataFormatsMID/Cluster3D.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/Track.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/GeometryTransformer.h"
#include "MIDSimulation/ChamberResponseParams.h"
#include "MIDSimulation/ClusterLabeler.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/Digitizer.h"
#include "MIDSimulation/DigitsMerger.h"
#include "MIDSimulation/DigitsPacker.h"
#include "MIDSimulation/Hit.h"
#include "MIDSimulation/PreClusterLabeler.h"
#include "MIDSimulation/TrackLabeler.h"
#include "MIDClustering/Clusterizer.h"
#include "MIDClustering/PreClusterHelper.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDTracking/Tracker.h"
#include "MIDTestingSimTools/TrackGenerator.h"
#include "MIDTestingSimTools/HitFinder.h"

namespace o2
{
namespace mid
{

Digitizer createDigitizerNoClusterSize()
{
  /// Returns the default chamber response
  ChamberResponseParams params = createDefaultChamberResponseParams();
  params.setParA(0., 0.);
  params.setParC(0., 0.);
  ChamberHV hv = createDefaultChamberHV();

  return Digitizer(ChamberResponse(params, hv), createDefaultChamberEfficiencyResponse(), createDefaultTransformer());
}

struct SimBase {
  Mapping mapping;
  GeometryTransformer geoTrans;
  SimBase() : mapping(), geoTrans(createDefaultTransformer()) {}
};

static SimBase simBase;

struct SimDigitizer {
  ChamberResponseParams params;
  Digitizer digitizer;
  Digitizer digitizerNoClusterSize;
  DigitsMerger digitsMerger;
  DigitsPacker digitsPacker;
  SimDigitizer() : params(createDefaultChamberResponseParams()), digitizer(createDefaultDigitizer()), digitizerNoClusterSize(createDigitizerNoClusterSize()), digitsMerger(), digitsPacker() {}
};

static SimDigitizer simDigitizer;

struct SimClustering {
  std::vector<std::array<size_t, 2>> correlation;
  PreClusterizer preClusterizer;
  Clusterizer clusterizer;
  PreClusterHelper preClusterHelper;
  PreClusterLabeler preClusterLabeler;
  ClusterLabeler clusterLabeler;
  SimClustering() : correlation(), preClusterizer(), clusterizer(), preClusterHelper(), preClusterLabeler(), clusterLabeler()
  {
    correlation.clear();
    preClusterizer.init();
    clusterizer.init([&](size_t baseIndex, size_t relatedIndex) { correlation.push_back({baseIndex, relatedIndex}); });
  }
};

static SimClustering simClustering;

struct SimTracking {
  Tracker tracker;
  TrackGenerator trackGen;
  HitFinder hitFinder;
  TrackLabeler trackLabeler;
  SimTracking() : tracker(simBase.geoTrans), trackGen(), hitFinder(simBase.geoTrans), trackLabeler()
  {
    trackGen.setSeed(123456789);
    tracker.init(true);
  }
};

static SimTracking simTracking;

struct GenTrack {
  Track track;
  std::vector<Hit> hits;
  int nFiredChambers{0};
  bool isReconstructible() { return nFiredChambers > 2; }
};

std::vector<Hit> generateHits(size_t nHits, int deId, const Mapping& mapping, const GeometryTransformer geoTrans)
{
  std::vector<Hit> hits;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> distX(-127.5, 127.5);
  std::uniform_real_distribution<double> distY(-40., 40.);
  while (hits.size() < nHits) {
    Point3D<float> point(distX(mt), distY(mt), 0.);
    if (!mapping.stripByPosition(point.x(), point.y(), 0, deId, false).isValid()) {
      continue;
    }
    Point3D<float> globalPoint = geoTrans.localToGlobal(deId, point);
    hits.emplace_back(hits.size(), deId, globalPoint, globalPoint);
  }
  return hits;
}

std::vector<GenTrack> generateTracks(int nTracks)
{
  Mapping::MpStripIndex stripIndex;
  auto tracks = simTracking.trackGen.generate(nTracks);
  std::vector<GenTrack> genTracks;
  for (auto& track : tracks) {
    int trackId = &track - &tracks[0];
    GenTrack genTrack;
    genTrack.track = track;
    for (int ich = 0; ich < 4; ++ich) {
      auto clusters = simTracking.hitFinder.getLocalPositions(track, ich);
      bool isFired = false;
      for (auto& cl : clusters) {
        stripIndex = simBase.mapping.stripByPosition(cl.xCoor, cl.yCoor, 0, cl.deId, false);
        if (!stripIndex.isValid()) {
          continue;
        }
        auto pos = simBase.geoTrans.localToGlobal(cl.deId, cl.xCoor, cl.yCoor);
        genTrack.hits.emplace_back(trackId, cl.deId, pos, pos);
        isFired = true;
      }
      if (isFired) {
        ++genTrack.nFiredChambers;
      }
    }
    genTracks.emplace_back(genTrack);
  }
  return genTracks;
}

bool checkLabel(const ColumnData& digit, MCLabel& label, std::string& errorMessage)
{
  std::stringstream ss;
  ss << digit << "\n";
  bool isOk = true;
  if (label.getDEId() != digit.deId) {
    isOk = false;
    ss << "label deId " << label.getDEId() << " != " << digit.deId << "\n";
  }
  if (label.getColumnId() != digit.columnId) {
    isOk = false;
    ss << "label columnId " << label.getColumnId() << " != " << digit.columnId << "\n";
  }
  int firstStrip = label.getFirstStrip();
  int lastStrip = label.getLastStrip();
  int cathode = label.getCathode();
  int nLines = (cathode == 0) ? 4 : 1;
  for (int iline = 0; iline < nLines; ++iline) {
    for (int istrip = 0; istrip < 16; ++istrip) {
      int currStrip = MCLabel::getStrip(istrip, iline);
      bool isFired = digit.isStripFired(istrip, cathode, iline);
      if (isFired != (currStrip >= firstStrip && currStrip <= lastStrip)) {
        isOk = false;
        ss << "Cathode: " << cathode << "  firstStrip: " << firstStrip << "  lastStrip: " << lastStrip;
        ss << "  line: " << iline << "  strip: " << istrip << "  fired: " << isFired << "\n";
      }
    }
  }
  errorMessage = ss.str();
  return isOk;
}

std::vector<PreCluster> getRelatedPreClusters(const Hit& hit, int cathode, const std::vector<PreCluster>& preClusters, const o2::dataformats::MCTruthContainer<MCCompLabel>& labels)
{
  std::vector<PreCluster> sortedPC;
  for (size_t ipc = 0; ipc < preClusters.size(); ++ipc) {
    for (auto& label : labels.getLabels(ipc)) {
      if (label.getTrackID() == hit.GetTrackID() && preClusters[ipc].cathode == cathode) {
        sortedPC.emplace_back(preClusters[ipc]);
      }
    }
  }
  std::sort(sortedPC.begin(), sortedPC.end(), [](const PreCluster& pc1, const PreCluster& pc2) { return (pc1.firstColumn <= pc2.firstColumn); });
  return std::move(sortedPC);
}

bool isContiguous(const std::vector<PreCluster>& sortedPC)
{
  int lastColumn = sortedPC.front().firstColumn;
  for (auto it = sortedPC.begin() + 1; it != sortedPC.end(); ++it) {
    if (it->firstColumn - (it - 1)->firstColumn != 1) {
      return false;
    }
  }
  return true;
}

bool isInside(double localX, double localY, const std::vector<PreCluster>& sortedPC, std::string& errorMsg)
{
  std::stringstream ss;
  ss << "Point: (" << localX << ", " << localY << ")  outside PC:";
  for (auto& pc : sortedPC) {
    MpArea area = simClustering.preClusterHelper.getArea(pc);
    ss << "  PC: " << pc;
    ss << "  area: x (" << area.getXmin() << ", " << area.getXmax() << ")";
    ss << " y (" << area.getYmin() << ", " << area.getYmax() << ")";
    if (localX >= area.getXmin() && localX <= area.getXmax() &&
        localY >= area.getYmin() && localY <= area.getYmax()) {
      return true;
    }
  }
  errorMsg = ss.str();
  return false;
}

std::string getDebugInfo(const std::vector<GenTrack>& genTracks, Tracker& tracker, TrackLabeler& trackLabeler)
{
  std::stringstream debug;
  for (size_t igen = 0; igen < genTracks.size(); ++igen) {
    debug << "Gen: " << genTracks[igen].track << "\n  hits:\n";
    for (auto& hit : genTracks[igen].hits) {
      debug << "    " << hit << "\n";
    }
    debug << "  clusters:\n";
    for (size_t icl = 0; icl < tracker.getClusters().size(); ++icl) {
      bool matches = false;
      for (auto& label : trackLabeler.getTrackClustersLabels().getLabels(icl)) {
        if (label.getTrackID() == igen) {
          matches = true;
          break;
        }
      }
      if (matches) {
        debug << "    icl: " << icl << "  " << tracker.getClusters()[icl] << "\n";
      }
    }
  }

  for (size_t itrack = 0; itrack < tracker.getTracks().size(); ++itrack) {
    debug << "reco: " << tracker.getTracks()[itrack] << "  matches:";
    for (auto& label : trackLabeler.getTracksLabels().getLabels(itrack)) {
      debug << "  " << label.getTrackID();
    }
    debug << "  clusters:\n";
    for (int ich = 0; ich < 4; ++ich) {
      int icl = tracker.getTracks()[itrack].getClusterMatched(ich);
      debug << "    icl: " << icl;
      if (icl >= 0) {
        debug << "    " << tracker.getClusters()[icl];
        for (auto& label : trackLabeler.getTrackClustersLabels().getLabels(icl)) {
          debug << "  ID: " << label.getTrackID() << "  fires: [" << label.isFiredBP() << ", " << label.isFiredNBP() << "]";
        }
        debug << "\n";
      }
    }
  }
  return debug.str();
}

std::vector<int> getDEList()
{
  // The algorithm should work in the same way on all detection elements.
  // Let us just sample the detection elements with different shape
  // (2,6 are long, 3, 5 have a cut geometry and 4 is shorter w.r.t.
  // the others in the same chamber plane)
  // in the 4 different chamber planes (different dimensions)
  std::vector<int> deList = {2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 47, 48, 49, 50, 51, 65, 66, 67, 68, 69};
  return deList;
}

BOOST_AUTO_TEST_SUITE(o2_mid_simulation)

BOOST_DATA_TEST_CASE(MID_Digitizer, boost::unit_test::data::make(getDEList()), deId)
{
  // In this test we generate hits, digitize them and test that the MC labels are correctly assigned
  auto hits = generateHits(10, deId, simBase.mapping, simBase.geoTrans);
  std::vector<ColumnDataMC> digitStoreMC;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC;
  simDigitizer.digitizer.process(hits, digitStoreMC, digitLabelsMC);
  // We check that we have as many sets of labels as digits
  BOOST_TEST(digitStoreMC.size() == digitLabelsMC.getIndexedSize());
  for (size_t idig = 0; idig < digitLabelsMC.getIndexedSize(); ++idig) {
    auto labels = digitLabelsMC.getLabels(idig);
    auto digit = digitStoreMC[idig];
    // Then for each label we check that the parameters correctly identify the digit
    for (auto label : labels) {
      std::string errorMessage;
      BOOST_TEST(checkLabel(digit, label, errorMessage), errorMessage.c_str());
    }
  }

  // We then test the merging of the hits
  std::vector<ColumnData> digitStore;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabels;
  simDigitizer.digitsMerger.process(digitStoreMC, digitLabelsMC, digitStore, digitLabels);
  // The number of merged digits should be smaller than the number of input digits
  BOOST_TEST(digitStore.size() <= digitStoreMC.size());
  // We check that we have as many sets of labels as digits
  BOOST_TEST(digitStore.size() == digitLabels.getIndexedSize());
  // We check that we do not discard any label in the merging
  BOOST_TEST(digitLabels.getNElements() == digitLabelsMC.getNElements());
  for (auto digit : digitStoreMC) {
    bool isMergedDigit = false;
    for (auto col : digitStore) {
      if (digit.deId == col.deId && digit.columnId == col.columnId) {
        /// Checks the merged pattern
        isMergedDigit = true;
        // Finally we check that all input patterns are contained in the merged ones
        for (int iline = 0; iline < 4; ++iline) {
          BOOST_TEST(((digit.getBendPattern(iline) & col.getBendPattern(iline)) == digit.getBendPattern(iline)));
        }
        BOOST_TEST(((digit.getNonBendPattern() & col.getNonBendPattern()) == digit.getNonBendPattern()));
      }
    }
    // Check that all digits get merged
    BOOST_TEST(isMergedDigit);
  }
}

BOOST_DATA_TEST_CASE(MID_DigitReader, boost::unit_test::data::make(getDEList()), deId)
{
  // Testing of the packing of digits

  size_t nEvents = 20;
  std::vector<std::vector<ColumnDataMC>> digitsCollection;
  std::vector<ColumnDataMC> digits, packedDigits;
  std::vector<o2::dataformats::MCTruthContainer<MCLabel>> mcContainerCollection;
  o2::dataformats::MCTruthContainer<MCLabel> mcContainer, packedMCContainer;
  for (size_t ievent = 0; ievent < nEvents; ++ievent) {
    // Generate digits per event. Each event has a different timestamp
    auto hits = generateHits(5, deId, simBase.mapping, simBase.geoTrans);
    digitsCollection.push_back({});
    mcContainerCollection.push_back({});
    simDigitizer.digitizer.setEventTime(10 * ievent);
    simDigitizer.digitizer.process(hits, digitsCollection.back(), mcContainerCollection.back());
    std::copy(digitsCollection.back().begin(), digitsCollection.back().end(), std::back_inserter(digits));
    mcContainer.mergeAtBack(mcContainerCollection.back());
  }
  // This should pack again the digits per event
  simDigitizer.digitsPacker.process(digits, mcContainer, 0);
  BOOST_TEST(simDigitizer.digitsPacker.getNGroups() == nEvents);
  for (size_t igroup = 0; igroup < simDigitizer.digitsPacker.getNGroups(); ++igroup) {
    simDigitizer.digitsPacker.getGroup(igroup, packedDigits, packedMCContainer);
    // Check that the number of digits per event is equal to the input one
    BOOST_TEST(packedDigits.size() == digitsCollection[igroup].size());
    for (size_t idigit = 0; idigit < packedDigits.size(); ++idigit) {
      // Check that we have the same digits as the input ones
      BOOST_TEST(packedDigits[idigit].getNonBendPattern() == digitsCollection[igroup][idigit].getNonBendPattern());
      // Check that we have the same labels as the input ones
      size_t nLabels = packedMCContainer.getLabels(idigit).size();
      BOOST_TEST(nLabels == mcContainerCollection[igroup].getLabels(idigit).size());
      for (size_t ilabel = 0; ilabel < nLabels; ++ilabel) {
        BOOST_TEST(packedMCContainer.getLabels(idigit)[ilabel] == mcContainerCollection[igroup].getLabels(idigit)[ilabel]);
      }
    }
  }
}

BOOST_DATA_TEST_CASE(MID_SingleCluster, boost::unit_test::data::make(getDEList()), deId)
{
  // In this test, we generate reconstruct one single hit.
  // If the hit is in the RPC, the digitizer will return a list of strips,
  // that are close to each other by construction.
  // We will therefore have exactly one reconstructed cluster.
  // It is worth noting, however, that the clustering algorithm is designed to
  // produce two clusters if the BP and NBP do not overlap.
  // The digitizer produces superposed BP and NBP strips only if the response parameters
  // are the same for both.
  // Otherwise we can have from 1 to 3 clusters produced.

  std::vector<ColumnDataMC> digitStoreMC;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC;
  std::vector<ColumnData> digitStore;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabels;

  for (int ievent = 0; ievent < 100; ++ievent) {
    auto hits = generateHits(1, deId, simBase.mapping, simBase.geoTrans);
    std::stringstream ss;
    int nGenClusters = 1, nRecoClusters = 0;
    simDigitizer.digitizer.process(hits, digitStoreMC, digitLabelsMC);
    simDigitizer.digitsMerger.process(digitStoreMC, digitLabelsMC, digitStore, digitLabels);
    simClustering.preClusterizer.process(digitStore);
    gsl::span<const PreCluster> preClusters(simClustering.preClusterizer.getPreClusters().data(), simClustering.preClusterizer.getPreClusters().size());
    simClustering.clusterizer.process(preClusters);
    nRecoClusters = simClustering.clusterizer.getClusters().size();
    ss << "nRecoClusters: " << nRecoClusters << "  nGenClusters: " << nGenClusters << "\n";
    for (auto& col : digitStore) {
      ss << col << "\n";
    }
    ss << "\n  Clusters:\n";
    for (auto& cl : simClustering.clusterizer.getClusters()) {
      ss << cl << "\n";
    }

    int nColumns = digitStore.size();

    if (simDigitizer.params.getParB(0, deId) == simDigitizer.params.getParB(1, deId) && nColumns <= 2) {
      BOOST_TEST((nRecoClusters == nGenClusters), ss.str());
    } else {
      BOOST_TEST((nRecoClusters >= nGenClusters && nRecoClusters <= nColumns), ss.str());
    }
  }
}

BOOST_DATA_TEST_CASE(MID_SimClusters, boost::unit_test::data::make(getDEList()), deId)
{
  // In this test, we generate few hits, reconstruct the clusters
  // and verify that the MC labels are correctly assigned to the clusters

  std::vector<ColumnDataMC> digitStoreMC;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC;
  std::vector<ColumnData> digitStore;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabels;

  for (int ievent = 0; ievent < 100; ++ievent) {
    auto hits = generateHits(10, deId, simBase.mapping, simBase.geoTrans);
    std::stringstream ss;
    int nGenClusters = 1, nRecoClusters = 0;
    simDigitizer.digitizer.process(hits, digitStoreMC, digitLabelsMC);
    simDigitizer.digitsMerger.process(digitStoreMC, digitLabelsMC, digitStore, digitLabels);
    simClustering.preClusterizer.process(digitStore);
    gsl::span<const PreCluster> preClusters(simClustering.preClusterizer.getPreClusters().data(), simClustering.preClusterizer.getPreClusters().size());
    simClustering.correlation.clear();
    simClustering.clusterizer.process(preClusters);
    simClustering.preClusterLabeler.process(preClusters, digitLabels);
    // Check that all pre-clusters have a label
    BOOST_TEST(preClusters.size() == simClustering.preClusterLabeler.getContainer().getIndexedSize());
    // Check that the pre-clusters contain the hits from which they were generated
    for (auto& hit : hits) {
      auto pt = simBase.geoTrans.globalToLocal(hit.GetDetectorID(), hit.middlePoint());
      for (int icath = 0; icath < 2; ++icath) {
        auto sortedPC = getRelatedPreClusters(hit, 1, simClustering.preClusterizer.getPreClusters(), simClustering.preClusterLabeler.getContainer());
        if (icath == 1) {
          // Check that there is only 1 pre-cluster in the NBP
          // CAVEAT: this is valid as far as we do not have masked strips
          BOOST_TEST(sortedPC.size() == 1);
        }
        std::string errorMsg;
        BOOST_TEST(isInside(pt.x(), pt.y(), sortedPC, errorMsg), errorMsg.c_str());
      }
    }

    gsl::span<const Cluster2D> clusters(simClustering.clusterizer.getClusters().data(), simClustering.clusterizer.getClusters().size());
    gsl::span<const std::array<size_t, 2>> correlation(simClustering.correlation.data(), simClustering.correlation.size());
    simClustering.clusterLabeler.process(preClusters, simClustering.preClusterLabeler.getContainer(), clusters, correlation);
    // Check that all clusters have a label
    BOOST_TEST(clusters.size() == simClustering.clusterLabeler.getContainer().getIndexedSize());

    for (auto pair : simClustering.correlation) {
      // std::string errorMsg;
      // const auto& cl(clusters[pair.first]);
      // std::vector<PreCluster> sortedPC{ preClusters[pair.second] };
      // Test that the cluster is inside the associated pre-cluster
      // BOOST_TEST(isInside(cl.xCoor, cl.yCoor, sortedPC, errorMsg), errorMsg.c_str());
      // Test that the cluster has all pre-clusters labels
      for (auto& pcLabel : simClustering.preClusterLabeler.getContainer().getLabels(pair[1])) {
        bool isInLabels = false;
        for (auto& label : simClustering.clusterLabeler.getContainer().getLabels(pair[0])) {
          if (label.compare(pcLabel) == 1) {
            isInLabels = true;
            bool isFired = (preClusters[pair[1]].cathode == 0) ? label.isFiredBP() : label.isFiredNBP();
            // Test that the fired flag is correctly set
            BOOST_TEST(isFired);
            break;
          }
        }
        BOOST_TEST(isInLabels);
      }
    }
  }
}

BOOST_DATA_TEST_CASE(MID_SimTracks, boost::unit_test::data::make({1, 2, 3, 4, 5, 6, 7, 8, 9}), nTracks)
{
  // In this test, we generate tracks and we reconstruct them.
  // It is worth noting that in this case we do not use the default digitizer
  // But rather a digitizer with a cluster size of 0.
  // The reason is that, with the current implementation of the digitizer,
  // which has to be further fine-tuned,
  // we can have very large clusters, of few local-boards/columns.
  // In this case the resolution is rather bad, and this impacts the tracking performance,
  // even when the tracking algorithm has no bugs.
  // The aim of this test is to check that the algorithms work.
  // The tracking performance and tuning of the MC will be done in the future via dedicated studies.

  std::vector<ColumnDataMC> digitStoreMC;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC;
  std::vector<ColumnData> digitStore;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabels;

  // In the tracking algorithm, if we have two tracks that are compatible within uncertainties
  // we keep only one of the two. This is done to avoid duplicated tracks.
  // In this test we can have many tracks in the same event.
  // If two tracks are close, they can give two reconstucted tracks compatible among each others,
  // within their uncertainties. One of the two is therefore rejected.
  // However, the track might not be compatible with the (rejected) generated track that has no uncertainty.
  // To avoid this, compare adding a factor 2 in the sigma cut.
  float chi2Cut = simTracking.tracker.getSigmaCut() * simTracking.tracker.getSigmaCut();

  unsigned long int nReco = 0, nGood = 0, nUntagged = 0, nTaggedNonCompatible = 0, nReconstructible = 0;

  for (int ievent = 0; ievent < 100; ++ievent) {
    auto genTracks = generateTracks(nTracks);
    std::vector<Hit> hits;
    for (auto& genTrack : genTracks) {
      std::copy(genTrack.hits.begin(), genTrack.hits.end(), std::back_inserter(hits));
      if (genTrack.isReconstructible()) {
        ++nReconstructible;
      }
    }

    simDigitizer.digitizerNoClusterSize.process(hits, digitStoreMC, digitLabelsMC);
    simDigitizer.digitsMerger.process(digitStoreMC, digitLabelsMC, digitStore, digitLabels);
    simClustering.preClusterizer.process(digitStore);
    gsl::span<const PreCluster> preClusters(simClustering.preClusterizer.getPreClusters().data(), simClustering.preClusterizer.getPreClusters().size());
    simClustering.correlation.clear();
    simClustering.clusterizer.process(preClusters);
    simClustering.preClusterLabeler.process(preClusters, digitLabels);
    gsl::span<const Cluster2D> clusters(simClustering.clusterizer.getClusters().data(), simClustering.clusterizer.getClusters().size());
    gsl::span<const std::array<size_t, 2>> correlation(simClustering.correlation.data(), simClustering.correlation.size());
    simClustering.clusterLabeler.process(preClusters, simClustering.preClusterLabeler.getContainer(), clusters, correlation);
    simTracking.tracker.process(clusters);
    gsl::span<const Track> tracks(simTracking.tracker.getTracks().data(), simTracking.tracker.getTracks().size());
    gsl::span<const Cluster3D> trClusters(simTracking.tracker.getClusters().data(), simTracking.tracker.getClusters().size());
    simTracking.trackLabeler.process(trClusters, tracks, simClustering.clusterLabeler.getContainer());

    // For the moment we save all clusters
    BOOST_TEST(trClusters.size() == clusters.size());
    BOOST_TEST(tracks.size() == simTracking.trackLabeler.getTracksLabels().getIndexedSize());
    BOOST_TEST(trClusters.size() == simTracking.trackLabeler.getTrackClustersLabels().getIndexedSize());

    std::string debugInfo = "";
    // Test that all reconstructible tracks are reconstructed
    for (size_t igen = 0; igen < genTracks.size(); ++igen) {
      bool isReco = false;
      for (size_t itrack = 0; itrack < tracks.size(); ++itrack) {
        for (auto& label : simTracking.trackLabeler.getTracksLabels().getLabels(itrack)) {
          if (label.getTrackID() == igen) {
            if (tracks[itrack].isCompatible(genTracks[igen].track, chi2Cut)) {
              isReco = true;
            }
          }
        }
      }
      std::stringstream ss;
      ss << "Gen ID: " << igen << "  isReconstructible: " << genTracks[igen].isReconstructible() << " != isReco " << isReco;
      if (debugInfo.empty()) {
        debugInfo = getDebugInfo(genTracks, simTracking.tracker, simTracking.trackLabeler);
      }
      ss << "\n"
         << debugInfo;
      BOOST_TEST(isReco == genTracks[igen].isReconstructible(), ss.str().c_str());
    }

    // Perform some statistics
    for (size_t itrack = 0; itrack < tracks.size(); ++itrack) {
      for (auto& label : simTracking.trackLabeler.getTracksLabels().getLabels(itrack)) {
        if (label.isEmpty()) {
          ++nUntagged;
          continue;
        }
        const Track& matchedGenTrack(genTracks[label.getTrackID()].track);
        if (tracks[itrack].isCompatible(matchedGenTrack, chi2Cut)) {
          ++nGood;
        } else {
          ++nTaggedNonCompatible;
        }
        int nMatched = 0;
        for (int ich = 0; ich < 4; ++ich) {
          int trClusIdx = tracks[itrack].getClusterMatched(ich);
          if (trClusIdx < 0) {
            continue;
          }
          for (auto& trClusterLabel : simTracking.trackLabeler.getTrackClustersLabels().getLabels(trClusIdx)) {
            if (trClusterLabel.getTrackID() == label.getTrackID()) {
              ++nMatched;
            }
          }
        }
        BOOST_TEST(nMatched >= 3);
      }
    }
  }
  std::stringstream outMsg;
  outMsg << "Tracks per event: " << nTracks << "  fraction of good: " << static_cast<double>(nGood) / static_cast<double>(nReconstructible) << "  untagged: " << static_cast<double>(nUntagged) / static_cast<double>(nReconstructible) << "  tagged but not compatible: " << static_cast<double>(nTaggedNonCompatible) / static_cast<double>(nReconstructible);

  BOOST_TEST_MESSAGE(outMsg.str().c_str());
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace mid
} // namespace o2
