// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCHwClusterer.cxx
/// \brief This task tests the TPC HwClusterer
/// \author Sebastian Klewin <sebastian.klewin@cern.ch>

#define BOOST_TEST_MODULE Test TPC HwClusterer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsTPC/Helpers.h"
#include "TPCBase/Digit.h"
#include "TPCBase/Mapper.h"
#include "TPCReconstruction/HwClusterer.h"

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <vector>
#include <memory>
#include <iostream>

namespace o2
{
namespace TPC
{

/// @brief Test 1 basic class tests
BOOST_AUTO_TEST_CASE(HwClusterer_test1)
{
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_shared<std::vector<ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_shared<MCLabelContainer>();

  HwClusterer clusterer(clusterArray, labelArray, 0);
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digitVec = std::vector<Digit>();
  MCLabelContainer labelContainer;

  // create three digits above peak charge threshold, two close to each other
  // which should result in one cluster
  digitVec.emplace_back(0, 123, 13, 4, 2);
  digitVec.emplace_back(0, 12, 13, 5, 2);
  digitVec.emplace_back(0, 321, 7, 10, 10);

  labelContainer.addElement(0, 1);
  labelContainer.addElement(1, 2);
  labelContainer.addElement(2, 3);

  auto digits = std::make_unique<const std::vector<Digit>>(digitVec);
  auto mcDigitTruth = std::make_unique<const MCLabelContainer>(labelContainer);

  clusterer.Process(*digits.get(), mcDigitTruth.get(), 0);

  // check if clusters were found
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);

  // check if two lusters were found, one in row 13 with charge 123 and one in 12 with charge 321
  BOOST_CHECK_EQUAL(clusterArray->at(0).getContainer()->numberOfClusters, 2);
  BOOST_CHECK_EQUAL(clusterArray->at(0).getContainer()->clusters[0].getRow(), 13);
  BOOST_CHECK_EQUAL(clusterArray->at(0).getContainer()->clusters[0].getQMax(), 123);
  BOOST_CHECK_EQUAL(clusterArray->at(0).getContainer()->clusters[1].getRow(), 7);
  BOOST_CHECK_EQUAL(clusterArray->at(0).getContainer()->clusters[1].getQMax(), 321);

  // check if MC labels are propagated correctly
  BOOST_CHECK_EQUAL(labelArray->getLabels(0).size(), 2); // first cluster got two MC labels
  BOOST_CHECK_EQUAL(labelArray->getLabels(0)[0].getTrackID(), 1);
  BOOST_CHECK_EQUAL(labelArray->getLabels(0)[1].getTrackID(), 2);
  BOOST_CHECK_EQUAL(labelArray->getLabels(1).size(), 1); // second cluster got one MC label
  BOOST_CHECK_EQUAL(labelArray->getLabels(1)[0].getTrackID(), 3);
}

struct sortTime {
  inline bool operator()(const o2::TPC::Digit& d1, const o2::TPC::Digit& d2)
  {
    return (d1.getTimeStamp() < d2.getTimeStamp());
  }
};

/// @brief Test 2 CF test with single pad clusters
BOOST_AUTO_TEST_CASE(HwClusterer_test2)
{
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_shared<std::vector<o2::TPC::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::TPC::HwClusterer clusterer(clusterArray, labelArray, 0);
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = std::make_unique<std::vector<o2::TPC::Digit>>();

  // create a lot of single pad clusters, one in every pad, well separated in time
  // which should result in one cluster
  // Digit(int cru, float charge, int row, int pad, int time)
  o2::TPC::Mapper& mapper = o2::TPC::Mapper::instance();
  std::vector<unsigned> clusterPerRegionGenerated(10, 0);
  int globalRow = 0;
  for (int region = 0; region < 10; ++region) {
    for (int row = 0; row < mapper.getNumberOfRowsRegion(region); ++row) {
      int time = 0;
      for (int pad = 0; pad < mapper.getNumberOfPadsInRowSector(globalRow); ++pad) {
        ++clusterPerRegionGenerated[region];
        digits->emplace_back(region, 20, globalRow, pad, time);
        time += 7;
      }
      ++globalRow;
    }
  }
  std::sort(digits->begin(), digits->end(), sortTime());

  // Search clusters
  clusterer.Process(*digits.get(), nullptr, 0);

  // Check result
  BOOST_CHECK_EQUAL(clusterArray->size(), 47);

  // check if all clusters were found
  std::vector<unsigned> clusterPerRegionFound(10, 0);
  for (auto& cont : *clusterArray) {
    auto clusterContainer = cont.getContainer();
    clusterPerRegionFound[clusterContainer->CRU] += clusterContainer->numberOfClusters;
  }
  for (int region = 0; region < 10; ++region) {
    BOOST_CHECK_EQUAL(clusterPerRegionFound[region], clusterPerRegionGenerated[region]);
  }

  // check if all cluster charges (tot and max) are 20
  for (auto& cont : *clusterArray) {
    auto clusterContainer = cont.getContainer();
    for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(), 20);
      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(), 20);
    }
  }

  // check the pad and time positions of the clusters and sigmas (all 0 because single pad lcusters)
  for (auto& cont : *clusterArray) {
    auto clusterContainer = cont.getContainer();
    for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
      BOOST_CHECK_EQUAL((clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset) / 7,
                        clusterContainer->clusters[clIndex].getPad());
      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getSigmaPad2(), 0);
      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getSigmaTime2(), 0);
    }
  }
}
}
}
