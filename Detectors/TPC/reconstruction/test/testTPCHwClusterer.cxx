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
// some helper functions / struct
struct sortTime {
  inline bool operator()(const o2::TPC::Digit& d1, const o2::TPC::Digit& d2)
  {
    return (d1.getTimeStamp() < d2.getTimeStamp());
  }
};

float p_pre(std::array<int, 25>& data)
{
  int ret = 0;
  ret += -2 * (data[0]) - (data[1]) + 0 * (data[3]) + (data[3]) + 2 * (data[4]);
  ret += -2 * (data[5]) - (data[6]) + 0 * (data[8]) + (data[8]) + 2 * (data[9]);
  ret += -2 * (data[10]) - (data[11]) + 0 * (data[13]) + (data[13]) + 2 * (data[14]);
  ret += -2 * (data[15]) - (data[16]) + 0 * (data[18]) + (data[18]) + 2 * (data[19]);
  ret += -2 * (data[20]) - (data[21]) + 0 * (data[23]) + (data[23]) + 2 * (data[24]);
  return ret;
};

float sigma_p_pre(std::array<int, 25>& data)
{
  int ret = 0;
  ret += 4 * (data[0]) + (data[1]) + 0 * (data[3]) + (data[3]) + 4 * (data[4]);
  ret += 4 * (data[5]) + (data[6]) + 0 * (data[8]) + (data[8]) + 4 * (data[9]);
  ret += 4 * (data[10]) + (data[11]) + 0 * (data[13]) + (data[13]) + 4 * (data[14]);
  ret += 4 * (data[15]) + (data[16]) + 0 * (data[18]) + (data[18]) + 4 * (data[19]);
  ret += 4 * (data[20]) + (data[21]) + 0 * (data[23]) + (data[23]) + 4 * (data[24]);
  return ret;
};

float t_pre(std::array<int, 25>& data)
{
  int ret = 0;
  ret += -2 * (data[0]) - 2 * (data[1]) - 2 * (data[2]) - 2 * (data[3]) - 2 * (data[4]);
  ret += -1 * (data[5]) - 1 * (data[6]) - 1 * (data[7]) - 1 * (data[8]) - 1 * (data[9]);
  ret += 0 * (data[10]) + 0 * (data[11]) + 0 * (data[12]) + 0 * (data[13]) + 0 * (data[14]);
  ret += 1 * (data[15]) + 1 * (data[16]) + 1 * (data[17]) + 1 * (data[18]) + 1 * (data[19]);
  ret += 2 * (data[20]) + 2 * (data[21]) + 2 * (data[22]) + 2 * (data[23]) + 2 * (data[24]);
  return ret;
};

float sigma_t_pre(std::array<int, 25>& data)
{
  int ret = 0;
  ret += 4 * (data[0]) + 4 * (data[1]) + 4 * (data[2]) + 4 * (data[3]) + 4 * (data[4]);
  ret += 1 * (data[5]) + 1 * (data[6]) + 1 * (data[7]) + 1 * (data[8]) + 1 * (data[9]);
  ret += 0 * (data[10]) + 0 * (data[11]) + 0 * (data[12]) + 0 * (data[13]) + 0 * (data[14]);
  ret += 1 * (data[15]) + 1 * (data[16]) + 1 * (data[17]) + 1 * (data[18]) + 1 * (data[19]);
  ret += 4 * (data[20]) + 4 * (data[21]) + 4 * (data[22]) + 4 * (data[23]) + 4 * (data[24]);
  return ret;
};

/// @brief Test 1 basic class tests
BOOST_AUTO_TEST_CASE(HwClusterer_test1)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 1, basic class tests." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<MCLabelContainer>();

  HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
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

  clusterer.process(*digits.get(), mcDigitTruth.get());

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

  std::cout << std::endl;
  std::cout << "##" << std::endl;
  std::cout << "## Test 1 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}

/// @brief Test 2 CF test with single pad clusters
BOOST_AUTO_TEST_CASE(HwClusterer_test2)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 2, finding single pad clusters." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<o2::TPC::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::TPC::HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
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
  clusterer.process(*digits.get(), nullptr);

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
  std::cout << std::endl;
  std::cout << "##" << std::endl;
  std::cout << "## Test 2 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}

/// @brief Test 3 Calculation of cluster properties
BOOST_AUTO_TEST_CASE(HwClusterer_test3)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 3, computing cluster properties." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<o2::TPC::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::TPC::HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = std::make_unique<std::vector<o2::TPC::Digit>>();

  // Digit(int cru, float charge, int row, int pad, int time)
  // Create digits for different clusters
  std::array<std::array<int, 25>, 6> clusters = { { { 7, 10, 11, 8, 5,
                                                      10, 18, 21, 15, 8,
                                                      12, 22, 50, 20, 10,
                                                      9, 16, 20, 15, 8,
                                                      6, 9, 10, 8, 5 },
                                                    { 7, 10, 11, 8, 5,
                                                      11, 19, 22, 16, 9,
                                                      14, 24, 52, 22, 12,
                                                      12, 19, 23, 18, 11,
                                                      10, 13, 14, 12, 9 },
                                                    { 12, 15, 16, 13, 10,
                                                      16, 24, 27, 21, 14,
                                                      19, 29, 57, 27, 17,
                                                      17, 24, 28, 23, 16,
                                                      15, 18, 19, 17, 14 },
                                                    { 17, 20, 21, 18, 15,
                                                      21, 29, 32, 26, 19,
                                                      24, 34, 62, 32, 22,
                                                      22, 29, 33, 28, 21,
                                                      20, 23, 24, 22, 19 },
                                                    { 22, 25, 26, 23, 20,
                                                      26, 34, 37, 31, 24,
                                                      29, 39, 67, 37, 27,
                                                      27, 34, 38, 33, 26,
                                                      25, 28, 29, 27, 24 },
                                                    { 27, 30, 31, 28, 25,
                                                      31, 39, 42, 36, 29,
                                                      34, 44, 72, 42, 32,
                                                      32, 39, 43, 38, 31,
                                                      30, 33, 34, 32, 29 } } };

  for (int dp = 0; dp < 5; ++dp) {
    for (int dt = 0; dt < 5; ++dt) {
      for (int cl = 0; cl < clusters.size(); ++cl) {
        digits->emplace_back(0, clusters[cl][dt * 5 + dp], cl, cl + dp, cl * 10 + dt);
      }
    }
  }

  std::sort(digits->begin(), digits->end(), sortTime());

  // Search clusters
  clusterer.process(*digits.get(), nullptr);

  // Check outcome
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);

  auto clusterContainer = (*clusterArray)[0].getContainer();
  // Checking cluster properties
  for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
    float qtot = std::accumulate(clusters[clIndex].begin(), clusters[clIndex].end(), 0);
    // Check QTot
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(), std::accumulate(clusters[clIndex].begin(), clusters[clIndex].end(), 0));
    // Check QMax
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(), clusters[clIndex][12]);
    // Check row
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getRow(), clIndex);
    // Check Flags
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getFlags(), 0);
    // Check pad
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getPad(),
      p_pre(clusters[clIndex]) / qtot + clIndex + 2,
      0.0001);
    // Check time
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset,
      t_pre(clusters[clIndex]) / qtot + clIndex * 10 + 2,
      0.0001);
    // Check pad sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaPad2(),
      (sigma_p_pre(clusters[clIndex]) / qtot) - ((p_pre(clusters[clIndex]) * p_pre(clusters[clIndex])) / (qtot * qtot)),
      0.0001);
    // Check time sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaTime2(),
      (sigma_t_pre(clusters[clIndex]) / qtot) - ((t_pre(clusters[clIndex]) * t_pre(clusters[clIndex])) / (qtot * qtot)),
      0.0001);
  }

  std::cout << std::endl;
  std::cout << "##" << std::endl;
  std::cout << "## Test 3 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}
}
}
