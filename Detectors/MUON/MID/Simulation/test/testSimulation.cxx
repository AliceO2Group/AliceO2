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

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <iostream>
#include <sstream>
#include "MathUtils/Cartesian3D.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/GeometryTransformer.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/Digitizer.h"
#include "MIDSimulation/DigitsMerger.h"
#include "MIDSimulation/ChamberResponseParams.h"
#include "MIDSimulation/Hit.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDClustering/Clusterizer.h"

namespace o2
{
namespace mid
{

struct SIMUL {
  static Mapping mapping;
  static ChamberResponseParams params;
  static GeometryTransformer geoTrans;
  static Digitizer digitizer;
  static DigitsMerger digitsMerger;
  static PreClusterizer preClusterizer;
  static Clusterizer clusterizer;

  void setup()
  {
    clusterizer.init();
  }
};

Mapping SIMUL::mapping;
ChamberResponseParams SIMUL::params = createDefaultChamberResponseParams();
GeometryTransformer SIMUL::geoTrans = createDefaultTransformer();
Digitizer SIMUL::digitizer = createDefaultDigitizer();
DigitsMerger SIMUL::digitsMerger;
PreClusterizer SIMUL::preClusterizer;
Clusterizer SIMUL::clusterizer;

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
    Point3D<float> globalPoint = SIMUL::geoTrans.localToGlobal(deId, point);
    hits.emplace_back(hits.size(), deId, globalPoint, globalPoint);
  }
  return hits;
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

std::vector<int> getDEList()
{
  // The algorithm should work in the same way on all detection elements.
  // Let us just sample the detection elements with different shape
  // (2,6 are long, 3, 5 have a cut geometry and 4 is shorter w.r.t.
  // the others in the same chamber plane)
  // in the 4 different chamber planes (different dimensions)
  std::vector<int> deList = { 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 47, 48, 49, 50, 51, 65, 66, 67, 68, 69 };
  return deList;
}

BOOST_TEST_GLOBAL_FIXTURE(SIMUL);

BOOST_AUTO_TEST_SUITE(o2_mid_simulation)
// BOOST_FIXTURE_TEST_SUITE(sim, SIMUL)

BOOST_DATA_TEST_CASE(MID_SimulChain, boost::unit_test::data::make(getDEList()), deId)
{
  // In this test, we generate a cluster from one impact point and we reconstruct it.
  // If the impact point is in the RPC, the digitizer will return a list of strips,
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
    auto hits = generateHits(1, deId, SIMUL::mapping, SIMUL::geoTrans);
    std::stringstream ss;
    int nGenClusters = 1, nRecoClusters = 0;
    SIMUL::digitizer.process(hits, digitStoreMC, digitLabelsMC);
    SIMUL::digitsMerger.process(digitStoreMC, digitLabelsMC, digitStore, digitLabels);
    SIMUL::preClusterizer.process(digitStore);
    gsl::span<const PreCluster> preClusters(SIMUL::preClusterizer.getPreClusters().data(), SIMUL::preClusterizer.getPreClusters().size());
    SIMUL::clusterizer.process(preClusters);
    nRecoClusters = SIMUL::clusterizer.getClusters().size();
    ss << "nRecoClusters: " << nRecoClusters << "  nGenClusters: " << nGenClusters << "\n";
    for (auto& col : digitStore) {
      ss << col << "\n";
    }
    ss << "\n  Clusters:\n";
    for (auto it = SIMUL::clusterizer.getClusters().begin(); it < SIMUL::clusterizer.getClusters().begin() + nRecoClusters; ++it) {
      ss << "pos: (" << it->xCoor << ", " << it->yCoor << ")  sigma2: (" << it->sigmaX2 << ", " << it->sigmaY2 << ")\n";
    }

    int nColumns = digitStore.size();

    if (SIMUL::params.getParB(0, deId) == SIMUL::params.getParB(1, deId) && nColumns <= 2) {
      BOOST_TEST((nRecoClusters == nGenClusters), ss.str());
    } else {
      BOOST_TEST((nRecoClusters >= nGenClusters && nRecoClusters <= nColumns), ss.str());
    }
  }
}

BOOST_DATA_TEST_CASE(MID_Digitizer, boost::unit_test::data::make(getDEList()), deId)
{
  // In this test we generate 100 hits
  // We then tests that the MC labels are correctly assigned
  auto hits = generateHits(100, deId, SIMUL::mapping, SIMUL::geoTrans);
  std::vector<ColumnDataMC> digitStoreMC;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabelsMC;
  SIMUL::digitizer.process(hits, digitStoreMC, digitLabelsMC);
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

  // We then test the merging of the 100 hits
  std::vector<ColumnData> digitStore;
  o2::dataformats::MCTruthContainer<MCLabel> digitLabels;
  SIMUL::digitsMerger.process(digitStoreMC, digitLabelsMC, digitStore, digitLabels);
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

// BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()

} // namespace mid
} // namespace o2
