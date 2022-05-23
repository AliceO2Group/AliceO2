// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE midSimulation
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/test/data/test_case.hpp>
#include <sstream>
#include "MathUtils/Cartesian.h"
#include "CommonConstants/LHCConstants.h"
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/Track.h"
#include "MIDBase/HitFinder.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/GeometryTransformer.h"
#include "MIDSimulation/ChamberResponseParams.h"
#include "MIDSimulation/ClusterLabeler.h"
#include "MIDSimulation/Digitizer.h"
#include "MIDSimulation/DigitsMerger.h"
#include "MIDSimulation/Hit.h"
#include "MIDSimulation/PreClusterLabeler.h"
#include "MIDSimulation/TrackLabeler.h"
#include "MIDClustering/Clusterizer.h"
#include "MIDClustering/PreClusterHelper.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDTracking/Tracker.h"
#include "MIDTestingSimTools/TrackGenerator.h"

namespace o2
{
namespace mid
{

std::vector<ColumnData> getColumnDataNonMC(const o2::mid::DigitsMerger& dm)
{
  std::vector<ColumnData> v;
  auto ref = dm.getColumnData();
  v.insert(v.begin(), ref.begin(), ref.end());
  return v;
}

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
  SimDigitizer() : params(createDefaultChamberResponseParams()), digitizer(createDefaultDigitizer()), digitizerNoClusterSize(createDigitizerNoClusterSize()), digitsMerger() {}
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
    math_utils::Point3D<float> point(distX(mt), distY(mt), 0.);
    if (!mapping.stripByPosition(point.x(), point.y(), 0, deId, false).isValid()) {
      continue;
    }
    math_utils::Point3D<float> globalPoint = geoTrans.localToGlobal(deId, point);
    hits.emplace_back(hits.size(), deId, globalPoint, globalPoint);
  }
  return hits;
}

std::vector<size_t> getCompatibleGenTrackIds(size_t igen, const std::vector<GenTrack>& genTracks)
{
  // Two generated tracks are indistinguishable in reconstruction if:
  // - they cross the same strip in all chamber planes
  // - they cross neighbour strips in all chamber planes
  // The first case is evident.
  // The second happens because the digits of the tracks will form a unique cluster.
  // The maximum strip pitch is 4 cm. So, to simplify,
  // we consider that the tracks will have digits in the same cluster
  // if the distance of their hits is smaller than twice the maximum strip pitch.
  std::vector<size_t> ids;
  auto& refHits = genTracks[igen].hits;
  for (auto itr = 0; itr < genTracks.size(); ++itr) {
    if (itr == igen) {
      ids.emplace_back(itr);
      continue;
    }
    auto& hits = genTracks[itr].hits;
    if (hits.size() == refHits.size()) {
      size_t nSame = 0;
      for (auto ihit = 0; ihit < refHits.size(); ++ihit) {
        if (hits[ihit].GetDetectorID() == refHits[ihit].GetDetectorID() && std::abs(hits[ihit].GetX() - refHits[ihit].GetX()) < 8. && std::abs(hits[ihit].GetY() - refHits[ihit].GetY()) < 8.) {
          ++nSame;
        } else {
          break;
        }
      }
      if (nSame == refHits.size()) {
        ids.emplace_back(itr);
      }
    }
  }
  return ids;
}

std::vector<GenTrack> generateTracks(int nTracks)
{
  auto tracks = simTracking.trackGen.generate(nTracks);
  std::vector<GenTrack> genTracks;
  for (auto trackIt = tracks.begin(), end = tracks.end(); trackIt != end; ++trackIt) {
    auto trackId = std::distance(tracks.begin(), trackIt);
    GenTrack genTrack;
    genTrack.track = *trackIt;
    for (int ich = 0; ich < 4; ++ich) {
      auto clusters = simTracking.hitFinder.getLocalPositions(*trackIt, ich);
      bool isFired = false;
      for (auto& cl : clusters) {
        auto stripIndex = simBase.mapping.stripByPosition(cl.xCoor, cl.yCoor, 0, cl.deId, false);
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

std::vector<PreCluster> getRelatedPreClusters(const Hit& hit, int cathode, const std::vector<PreCluster>& preClusters, const o2::dataformats::MCTruthContainer<MCCompLabel>& labels, const ROFRecord& rofRecord)
{
  std::vector<PreCluster> sortedPC;
  for (size_t ipc = rofRecord.firstEntry; ipc < rofRecord.firstEntry + rofRecord.nEntries; ++ipc) {
    for (auto& label : labels.getLabels(ipc)) {
      if (label.getTrackID() == hit.GetTrackID() && preClusters[ipc].cathode == cathode) {
        sortedPC.emplace_back(preClusters[ipc]);
      }
    }
  }
  std::sort(sortedPC.begin(), sortedPC.end(), [](const PreCluster& pc1, const PreCluster& pc2) { return (pc1.firstColumn <= pc2.firstColumn); });
  return sortedPC;
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

std::string getDebugInfo(const std::vector<GenTrack>& genTracks, Tracker& tracker, TrackLabeler& trackLabeler, const ROFRecord& rofTrack, const ROFRecord& rofCluster)
{
  std::stringstream debug;
  for (size_t igen = 0; igen < genTracks.size(); ++igen) {
    debug << "Gen: " << genTracks[igen].track << "\n  hits:\n";
    for (auto& hit : genTracks[igen].hits) {
      debug << "    " << hit << "\n";
    }
    debug << "  clusters:\n";
    for (size_t icl = rofCluster.firstEntry; icl < rofCluster.firstEntry + rofCluster.nEntries; ++icl) {
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

  for (size_t itrack = rofTrack.firstEntry; itrack < rofTrack.firstEntry + rofTrack.nEntries; ++itrack) {
    debug << "reco: " << tracker.getTracks()[itrack] << "  matches: " << trackLabeler.getTracksLabels()[itrack].getTrackID();
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

BOOST_DATA_TEST_CASE(MID_DigitMerger, boost::unit_test::data::make(getDEList()), deId)
{
  // Test the merging of the MC digits
  size_t nEvents = 20;
  std::vector<std::vector<ColumnData>> digitsCollection;
  std::vector<o2::dataformats::MCTruthContainer<MCLabel>> mcContainerCollection;
  std::vector<ColumnData> digits;
  o2::dataformats::MCTruthContainer<MCLabel> mcContainer;
  std::vector<ROFRecord> rofRecords;
  for (size_t ievent = 0; ievent < nEvents; ++ievent) {
    // Generate digits per event. Each event has a different timestamp
    auto hits = generateHits(1, deId, simBase.mapping, simBase.geoTrans);
    digitsCollection.push_back({});
    mcContainerCollection.push_back({});
    simDigitizer.digitizer.process(hits, digitsCollection.back(), mcContainerCollection.back());
    rofRecords.emplace_back(o2::constants::lhc::LHCBunchSpacingNS * ievent, EventType::Standard, digits.size(), digitsCollection.back().size());
    std::copy(digitsCollection.back().begin(), digitsCollection.back().end(), std::back_inserter(digits));
    mcContainer.mergeAtBack(mcContainerCollection.back());
  }
  simDigitizer.digitsMerger.process(digits, mcContainer, rofRecords);

  BOOST_TEST(simDigitizer.digitsMerger.getROFRecords().size() == rofRecords.size());
  auto rofMCIt = rofRecords.begin();
  auto rofIt = simDigitizer.digitsMerger.getROFRecords().begin();
  for (; rofIt != simDigitizer.digitsMerger.getROFRecords().end(); ++rofIt) {
    // Check that input and output event information are the same
    BOOST_TEST(rofIt->interactionRecord == rofMCIt->interactionRecord);
    BOOST_TEST(static_cast<int>(rofIt->eventType) == static_cast<int>(rofMCIt->eventType));
    // Check that the merged digits are less than the input ones
    BOOST_TEST(rofIt->nEntries <= rofMCIt->nEntries);
    ++rofMCIt;
  }
}

BOOST_DATA_TEST_CASE(MID_Digitizer, boost::unit_test::data::make(getDEList()), deId)
{
  // In this test we generate hits, digitize them and test that the MC labels are correctly assigned
  auto hits = generateHits(10, deId, simBase.mapping, simBase.geoTrans);
  std::vector<ColumnData> digitStoreMC;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC;
  std::vector<ROFRecord> rofRecords;
  simDigitizer.digitizer.process(hits, digitStoreMC, digitLabelsMC);
  rofRecords.emplace_back(1, EventType::Standard, 0, digitStoreMC.size());
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

  simDigitizer.digitsMerger.process(digitStoreMC, digitLabelsMC, rofRecords);
  // The number of merged digits should be smaller than the number of input digits
  BOOST_TEST(simDigitizer.digitsMerger.getColumnData().size() <= digitStoreMC.size());
  // We check that we have as many sets of labels as digits
  BOOST_TEST(simDigitizer.digitsMerger.getColumnData().size() == simDigitizer.digitsMerger.getMCContainer().getIndexedSize());
  // We check that we do not discard any label in the merging
  BOOST_TEST(simDigitizer.digitsMerger.getMCContainer().getNElements() == digitLabelsMC.getNElements());
  for (auto digit : digitStoreMC) {
    bool isMergedDigit = false;
    for (auto col : simDigitizer.digitsMerger.getColumnData()) {
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

  std::vector<ColumnData> digitStoreMC;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC;
  std::vector<ROFRecord> rofRecords;

  for (int ievent = 0; ievent < 100; ++ievent) {
    auto hits = generateHits(1, deId, simBase.mapping, simBase.geoTrans);
    std::stringstream ss;
    int nGenClusters = 1, nRecoClusters = 0;
    simDigitizer.digitizer.process(hits, digitStoreMC, digitLabelsMC);
    rofRecords.clear();
    rofRecords.emplace_back(o2::constants::lhc::LHCBunchSpacingNS * ievent, EventType::Standard, 0, digitStoreMC.size());
    simDigitizer.digitsMerger.process(digitStoreMC, digitLabelsMC, rofRecords);
    simClustering.preClusterizer.process(getColumnDataNonMC(simDigitizer.digitsMerger), simDigitizer.digitsMerger.getROFRecords());
    simClustering.clusterizer.process(simClustering.preClusterizer.getPreClusters(), simClustering.preClusterizer.getROFRecords());
    nRecoClusters = simClustering.clusterizer.getClusters().size();
    ss << "nRecoClusters: " << nRecoClusters << "  nGenClusters: " << nGenClusters << "\n";
    for (auto& col : simDigitizer.digitsMerger.getColumnData()) {
      ss << col << "\n";
    }
    ss << "\n  Clusters:\n";
    for (auto& cl : simClustering.clusterizer.getClusters()) {
      ss << cl << "\n";
    }

    BOOST_TEST(simClustering.clusterizer.getROFRecords().size() == rofRecords.size());

    int nColumns = simDigitizer.digitsMerger.getColumnData().size();

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

  std::vector<ColumnData> digitStoreMC, digitsAccum;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC, digitLabelsAccum;
  std::vector<ROFRecord> digitsROF;
  std::vector<std::vector<Hit>> hitsCollection;

  for (int ievent = 0; ievent < 100; ++ievent) {
    auto hits = generateHits(10, deId, simBase.mapping, simBase.geoTrans);
    hitsCollection.emplace_back(hits);
    simDigitizer.digitizer.process(hits, digitStoreMC, digitLabelsMC);
    digitsROF.emplace_back(o2::constants::lhc::LHCBunchSpacingNS * ievent, EventType::Standard, digitsAccum.size(), digitStoreMC.size());
    std::copy(digitStoreMC.begin(), digitStoreMC.end(), std::back_inserter(digitsAccum));
    digitLabelsAccum.mergeAtBack(digitLabelsMC);
  }
  simDigitizer.digitsMerger.process(digitsAccum, digitLabelsAccum, digitsROF);
  simClustering.preClusterizer.process(getColumnDataNonMC(simDigitizer.digitsMerger), simDigitizer.digitsMerger.getROFRecords());
  simClustering.correlation.clear();
  simClustering.clusterizer.process(simClustering.preClusterizer.getPreClusters(), simClustering.preClusterizer.getROFRecords());
  simClustering.preClusterLabeler.process(simClustering.preClusterizer.getPreClusters(), simDigitizer.digitsMerger.getMCContainer(), simClustering.preClusterizer.getROFRecords(), simDigitizer.digitsMerger.getROFRecords());
  // Check that all pre-clusters have a label
  BOOST_TEST(simClustering.preClusterizer.getPreClusters().size() == simClustering.preClusterLabeler.getContainer().getIndexedSize());
  // Check that the pre-clusters contain the hits from which they were generated
  for (size_t ievent = 0; ievent < hitsCollection.size(); ++ievent) {
    for (auto& hit : hitsCollection[ievent]) {
      auto pt = simBase.geoTrans.globalToLocal(hit.GetDetectorID(), hit.middlePoint());
      // Check only the NBP, since in the BP we can have 1 pre-cluster per column
      auto sortedPC = getRelatedPreClusters(hit, 1, simClustering.preClusterizer.getPreClusters(), simClustering.preClusterLabeler.getContainer(), simClustering.preClusterizer.getROFRecords()[ievent]);
      // Check that there is only 1 pre-cluster in the NBP
      // CAVEAT: this is valid as far as we do not have masked strips
      BOOST_TEST(sortedPC.size() == 1);
      std::string errorMsg;
      BOOST_TEST(isInside(pt.x(), pt.y(), sortedPC, errorMsg), errorMsg.c_str());
    }
  }

  simClustering.clusterLabeler.process(simClustering.preClusterizer.getPreClusters(), simClustering.preClusterLabeler.getContainer(), simClustering.clusterizer.getClusters(), simClustering.correlation);
  // Check that all clusters have a label
  BOOST_TEST(simClustering.clusterizer.getClusters().size() == simClustering.clusterLabeler.getContainer().getIndexedSize());

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
          bool isFired = (simClustering.preClusterizer.getPreClusters()[pair[1]].cathode == 0) ? label.isFiredBP() : label.isFiredNBP();
          // Test that the fired flag is correctly set
          BOOST_TEST(isFired);
          break;
        }
      }
      BOOST_TEST(isInLabels);
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

  std::vector<ColumnData> digitStoreMC, digitsAccum;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC, digitLabelsAccum;
  std::vector<ROFRecord> digitsROF;
  std::vector<std::vector<GenTrack>> genTrackCollection;

  // In the tracking algorithm, if we have two tracks that are compatible within uncertainties
  // we keep only one of the two. This is done to avoid duplicated tracks.
  // In this test we can have many tracks in the same event.
  // If two tracks are close, they can give two reconstructed tracks compatible among each others,
  // within their uncertainties. One of the two is therefore rejected.
  // However, the track might not be compatible with the (rejected) generated track that has no uncertainty.
  // To avoid this, compare adding a factor 2 in the sigma cut.
  float chi2Cut = simTracking.tracker.getSigmaCut() * simTracking.tracker.getSigmaCut();

  unsigned long int nGood = 0, nUntagged = 0, nTaggedNonCompatible = 0, nReconstructible = 0, nFakes = 0;

  for (size_t ievent = 0; ievent < 100; ++ievent) {
    auto genTracks = generateTracks(nTracks);
    std::vector<Hit> hits;
    for (auto& genTrack : genTracks) {
      std::copy(genTrack.hits.begin(), genTrack.hits.end(), std::back_inserter(hits));
      if (genTrack.isReconstructible()) {
        ++nReconstructible;
      }
    }
    genTrackCollection.emplace_back(genTracks);

    simDigitizer.digitizerNoClusterSize.process(hits, digitStoreMC, digitLabelsMC);
    digitsROF.emplace_back(o2::constants::lhc::LHCBunchSpacingNS * ievent, EventType::Standard, digitsAccum.size(), digitStoreMC.size());
    std::copy(digitStoreMC.begin(), digitStoreMC.end(), std::back_inserter(digitsAccum));
    digitLabelsAccum.mergeAtBack(digitLabelsMC);
  }

  simDigitizer.digitsMerger.process(digitsAccum, digitLabelsAccum, digitsROF);
  simClustering.preClusterizer.process(getColumnDataNonMC(simDigitizer.digitsMerger), simDigitizer.digitsMerger.getROFRecords());
  simClustering.correlation.clear();
  simClustering.clusterizer.process(simClustering.preClusterizer.getPreClusters(), simClustering.preClusterizer.getROFRecords());
  simClustering.preClusterLabeler.process(simClustering.preClusterizer.getPreClusters(), simDigitizer.digitsMerger.getMCContainer(), simClustering.preClusterizer.getROFRecords(), simDigitizer.digitsMerger.getROFRecords());
  simClustering.clusterLabeler.process(simClustering.preClusterizer.getPreClusters(), simClustering.preClusterLabeler.getContainer(), simClustering.clusterizer.getClusters(), simClustering.correlation);
  simTracking.tracker.process(simClustering.clusterizer.getClusters(), simClustering.clusterizer.getROFRecords());
  simTracking.trackLabeler.process(simTracking.tracker.getClusters(), simTracking.tracker.getTracks(), simClustering.clusterLabeler.getContainer());

  // For the moment we save all clusters
  BOOST_TEST(simTracking.tracker.getClusters().size() == simClustering.clusterizer.getClusters().size());
  BOOST_TEST(simTracking.tracker.getTracks().size() == simTracking.trackLabeler.getTracksLabels().size());
  BOOST_TEST(simTracking.tracker.getClusters().size() == simTracking.trackLabeler.getTrackClustersLabels().getIndexedSize());
  BOOST_TEST(simTracking.tracker.getTrackROFRecords().size() == digitsROF.size());

  // Test that all reconstructible tracks are reconstructed
  for (size_t ievent = 0; ievent < genTrackCollection.size(); ++ievent) {
    auto firstTrack = simTracking.tracker.getTrackROFRecords()[ievent].firstEntry;
    auto nTracks = simTracking.tracker.getTrackROFRecords()[ievent].nEntries;
    std::string debugInfo = "";
    for (size_t igen = 0; igen < genTrackCollection[ievent].size(); ++igen) {
      bool isReco = false;
      auto ids = getCompatibleGenTrackIds(igen, genTrackCollection[ievent]);
      for (size_t itrack = firstTrack; itrack < firstTrack + nTracks; ++itrack) {
        auto label = simTracking.trackLabeler.getTracksLabels()[itrack];
        bool checkReco = false;
        if (label.isFake()) {
          checkReco = true;
        }
        for (auto& id : ids) {
          if (label.getTrackID() == id) {
            checkReco = true;
            break;
          }
        }
        if (checkReco) {
          if (simTracking.tracker.getTracks()[itrack].isCompatible(genTrackCollection[ievent][igen].track, chi2Cut)) {
            isReco = true;
            break;
          }
        }
      }
      std::stringstream ss;
      ss << "Gen ID: " << igen << "  isReconstructible: " << genTrackCollection[ievent][igen].isReconstructible() << " != isReco " << isReco;
      if (debugInfo.empty()) {
        debugInfo = getDebugInfo(genTrackCollection[ievent], simTracking.tracker, simTracking.trackLabeler, simTracking.tracker.getTrackROFRecords()[ievent], simTracking.tracker.getClusterROFRecords()[ievent]);
      }
      ss << "\n"
         << debugInfo;
      BOOST_TEST(isReco == genTrackCollection[ievent][igen].isReconstructible(), ss.str().c_str());
    } // loop on generated tracks

    // Perform some statistics
    for (size_t itrack = firstTrack; itrack < firstTrack + nTracks; ++itrack) {
      auto label = simTracking.trackLabeler.getTracksLabels()[itrack];
      if (label.isEmpty()) {
        ++nUntagged;
        continue;
      }
      if (label.isFake()) {
        ++nFakes;
        continue;
      }
      const Track& matchedGenTrack(genTrackCollection[ievent][label.getTrackID()].track);
      if (simTracking.tracker.getTracks()[itrack].isCompatible(matchedGenTrack, chi2Cut)) {
        ++nGood;
      } else {
        ++nTaggedNonCompatible;
      }
      int nMatched = 0;
      for (int ich = 0; ich < 4; ++ich) {
        int trClusIdx = simTracking.tracker.getTracks()[itrack].getClusterMatched(ich);
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
    } // loop on reconstructed tracks
  }   // loop on event
  std::stringstream outMsg;
  outMsg << "Tracks per event: " << nTracks << "  fraction of good: " << static_cast<double>(nGood) / static_cast<double>(nReconstructible) << "  untagged: " << static_cast<double>(nUntagged) / static_cast<double>(nReconstructible) << "  tagged but not compatible: " << static_cast<double>(nTaggedNonCompatible) / static_cast<double>(nReconstructible) << "  fake: " << static_cast<double>(nFakes) / static_cast<double>(nReconstructible);

  BOOST_TEST_MESSAGE(outMsg.str().c_str());
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace mid
} // namespace o2
