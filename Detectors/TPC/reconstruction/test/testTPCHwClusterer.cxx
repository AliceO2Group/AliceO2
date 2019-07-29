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
namespace tpc
{
// some helper functions / struct
struct sortTime {
  inline bool operator()(const o2::tpc::Digit& d1, const o2::tpc::Digit& d2)
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

float p_pre_fp(std::array<float, 25>& data)
{
  int ret = 0;
  ret += -2 * (data[0] * 16) - (data[1] * 16) + 0 * (data[3] * 16) + (data[3] * 16) + 2 * (data[4] * 16);
  ret += -2 * (data[5] * 16) - (data[6] * 16) + 0 * (data[8] * 16) + (data[8] * 16) + 2 * (data[9] * 16);
  ret += -2 * (data[10] * 16) - (data[11] * 16) + 0 * (data[13] * 16) + (data[13] * 16) + 2 * (data[14] * 16);
  ret += -2 * (data[15] * 16) - (data[16] * 16) + 0 * (data[18] * 16) + (data[18] * 16) + 2 * (data[19] * 16);
  ret += -2 * (data[20] * 16) - (data[21] * 16) + 0 * (data[23] * 16) + (data[23] * 16) + 2 * (data[24] * 16);
  return ret;
};

float sigma_p_pre_fp(std::array<float, 25>& data)
{
  int ret = 0;
  ret += 4 * (data[0] * 16) + (data[1] * 16) + 0 * (data[3] * 16) + (data[3] * 16) + 4 * (data[4] * 16);
  ret += 4 * (data[5] * 16) + (data[6] * 16) + 0 * (data[8] * 16) + (data[8] * 16) + 4 * (data[9] * 16);
  ret += 4 * (data[10] * 16) + (data[11] * 16) + 0 * (data[13] * 16) + (data[13] * 16) + 4 * (data[14] * 16);
  ret += 4 * (data[15] * 16) + (data[16] * 16) + 0 * (data[18] * 16) + (data[18] * 16) + 4 * (data[19] * 16);
  ret += 4 * (data[20] * 16) + (data[21] * 16) + 0 * (data[23] * 16) + (data[23] * 16) + 4 * (data[24] * 16);
  return ret;
};

float t_pre_fp(std::array<float, 25>& data)
{
  int ret = 0;
  ret += -2 * (data[0] * 16) - 2 * (data[1] * 16) - 2 * (data[2] * 16) - 2 * (data[3] * 16) - 2 * (data[4] * 16);
  ret += -1 * (data[5] * 16) - 1 * (data[6] * 16) - 1 * (data[7] * 16) - 1 * (data[8] * 16) - 1 * (data[9] * 16);
  ret += 0 * (data[10] * 16) + 0 * (data[11] * 16) + 0 * (data[12] * 16) + 0 * (data[13] * 16) + 0 * (data[14] * 16);
  ret += 1 * (data[15] * 16) + 1 * (data[16] * 16) + 1 * (data[17] * 16) + 1 * (data[18] * 16) + 1 * (data[19] * 16);
  ret += 2 * (data[20] * 16) + 2 * (data[21] * 16) + 2 * (data[22] * 16) + 2 * (data[23] * 16) + 2 * (data[24] * 16);
  return ret;
};

float sigma_t_pre_fp(std::array<float, 25>& data)
{
  int ret = 0;
  ret += 4 * (data[0] * 16) + 4 * (data[1] * 16) + 4 * (data[2] * 16) + 4 * (data[3] * 16) + 4 * (data[4] * 16);
  ret += 1 * (data[5] * 16) + 1 * (data[6] * 16) + 1 * (data[7] * 16) + 1 * (data[8] * 16) + 1 * (data[9] * 16);
  ret += 0 * (data[10] * 16) + 0 * (data[11] * 16) + 0 * (data[12] * 16) + 0 * (data[13] * 16) + 0 * (data[14] * 16);
  ret += 1 * (data[15] * 16) + 1 * (data[16] * 16) + 1 * (data[17] * 16) + 1 * (data[18] * 16) + 1 * (data[19] * 16);
  ret += 4 * (data[20] * 16) + 4 * (data[21] * 16) + 4 * (data[22] * 16) + 4 * (data[23] * 16) + 4 * (data[24] * 16);
  return ret;
};

/// @brief Test 1 basic class tests
BOOST_AUTO_TEST_CASE(HwClusterer_test1)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 1, basic class tests." << std::endl;
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

  labelContainer.addElement(0, { 1, 0, 0, false });
  labelContainer.addElement(1, { 2, 0, 0, false });
  labelContainer.addElement(2, { 3, 0, 0, false });

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

  std::cout << "## Test 1 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}

/// @brief Test 2 CF test with single pad clusters
BOOST_AUTO_TEST_CASE(HwClusterer_test2)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 2, finding single pad clusters." << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<o2::tpc::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::tpc::HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = std::make_unique<std::vector<o2::tpc::Digit>>();

  // create a lot of single pad clusters, one in every pad, well separated in time
  // which should result in one cluster
  // Digit(int cru, float charge, int row, int pad, int time)
  o2::tpc::Mapper& mapper = o2::tpc::Mapper::instance();
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
  std::cout << "## Test 2 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}

/// @brief Test 3 Calculation of cluster properties
BOOST_AUTO_TEST_CASE(HwClusterer_test3)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 3, computing cluster properties." << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<o2::tpc::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::tpc::HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = std::make_unique<std::vector<o2::tpc::Digit>>();

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

  std::cout << "## Test 3 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}

/// @brief Test 4 Reject single pad clusters
BOOST_AUTO_TEST_CASE(HwClusterer_test4)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 4, rejecting single pad clusters." << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<o2::tpc::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::tpc::HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = std::make_unique<std::vector<o2::tpc::Digit>>();
  // Digit(int cru, float charge, int row, int pad, int time)
  // single pad and time cluster
  std::array<std::array<int, 25>, 4> clusters;
  clusters[0] = { { 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 67, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0 } };

  // single time cluster, but enlarged in pad direction
  clusters[1] = { { 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 12, 72, 24, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0 } };

  // single pad cluster, but enlarged in tim direction
  clusters[2] = { { 0, 0, 0, 0, 0,
                    0, 0, 33, 0, 0,
                    0, 0, 57, 0, 0,
                    0, 0, 12, 0, 0,
                    0, 0, 0, 0, 0 } };

  // wide cluster in both direction
  clusters[3] = { { 0, 0, 0, 0, 0,
                    0, 46, 71, 32, 0,
                    0, 32, 129, 16, 0,
                    0, 19, 53, 23, 0,
                    0, 0, 0, 0, 0 } };

  for (int dp = 0; dp < 5; ++dp) {
    for (int dt = 0; dt < 5; ++dt) {
      for (int cl = 0; cl < clusters.size(); ++cl) {
        digits->emplace_back(0, clusters[cl][dt * 5 + dp], cl, cl + dp, cl * 10 + dt);
      }
    }
  }

  std::sort(digits->begin(), digits->end(), sortTime());

  // Search clusters without thresholds, all 4 clusters shoud be found
  std::cout << "testing without thresholds..." << std::endl;
  clusterer.setRejectSinglePadClusters(false);
  clusterer.setRejectSingleTimeClusters(false);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  auto clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 4);
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

  // Search clusters with threshold in pad direction, only clusters 1 and 3 should be found
  std::cout << "testing with pad threshold..." << std::endl;
  clusterer.setRejectSinglePadClusters(true);
  clusterer.setRejectSingleTimeClusters(false);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 2);
  for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
    float qtot = std::accumulate(clusters[clIndex * 2 + 1].begin(), clusters[clIndex * 2 + 1].end(), 0);
    // Check QTot
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(), std::accumulate(clusters[clIndex * 2 + 1].begin(), clusters[clIndex * 2 + 1].end(), 0));
    // Check QMax
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(), clusters[clIndex * 2 + 1][12]);
    // Check row
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getRow(), clIndex * 2 + 1);
    // Check Flags
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getFlags(), 0);
    // Check pad
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getPad(),
      p_pre(clusters[clIndex * 2 + 1]) / qtot + (clIndex * 2 + 1) + 2,
      0.0001);
    // Check time
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset,
      t_pre(clusters[clIndex * 2 + 1]) / qtot + (clIndex * 2 + 1) * 10 + 2,
      0.0001);
    // Check pad sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaPad2(),
      (sigma_p_pre(clusters[clIndex * 2 + 1]) / qtot) - ((p_pre(clusters[clIndex * 2 + 1]) * p_pre(clusters[clIndex * 2 + 1])) / (qtot * qtot)),
      0.0001);
    // Check time sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaTime2(),
      (sigma_t_pre(clusters[clIndex * 2 + 1]) / qtot) - ((t_pre(clusters[clIndex * 2 + 1]) * t_pre(clusters[clIndex * 2 + 1])) / (qtot * qtot)),
      0.0001);
  }

  // Search clusters with threshold in time direction, only clusters 2 and 3 should be found
  std::cout << "testing with time threshold..." << std::endl;
  clusterer.setRejectSinglePadClusters(false);
  clusterer.setRejectSingleTimeClusters(true);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 2);
  for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
    float qtot = std::accumulate(clusters[clIndex + 2].begin(), clusters[clIndex + 2].end(), 0);
    // Check QTot
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(), std::accumulate(clusters[clIndex + 2].begin(), clusters[clIndex + 2].end(), 0));
    // Check QMax
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(), clusters[clIndex + 2][12]);
    // Check row
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getRow(), clIndex + 2);
    // Check Flags
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getFlags(), 0);
    // Check pad
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getPad(),
      p_pre(clusters[clIndex + 2]) / qtot + (clIndex + 2) + 2,
      0.0001);
    // Check time
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset,
      t_pre(clusters[clIndex + 2]) / qtot + (clIndex + 2) * 10 + 2,
      0.0001);
    // Check pad sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaPad2(),
      (sigma_p_pre(clusters[clIndex + 2]) / qtot) - ((p_pre(clusters[clIndex + 2]) * p_pre(clusters[clIndex + 2])) / (qtot * qtot)),
      0.0001);
    // Check time sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaTime2(),
      (sigma_t_pre(clusters[clIndex + 2]) / qtot) - ((t_pre(clusters[clIndex + 2]) * t_pre(clusters[clIndex + 2])) / (qtot * qtot)),
      0.0001);
  }

  // Search clusters with both thresholds, only cluster 3 should be found
  std::cout << "testing both thresholds..." << std::endl;
  clusterer.setRejectSinglePadClusters(true);
  clusterer.setRejectSingleTimeClusters(true);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 1);
  for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
    float qtot = std::accumulate(clusters[clIndex + 3].begin(), clusters[clIndex + 3].end(), 0);
    // Check QTot
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(), std::accumulate(clusters[clIndex + 3].begin(), clusters[clIndex + 3].end(), 0));
    // Check QMax
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(), clusters[clIndex + 3][12]);
    // Check row
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getRow(), clIndex + 3);
    // Check Flags
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getFlags(), 0);
    // Check pad
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getPad(),
      p_pre(clusters[clIndex + 3]) / qtot + (clIndex + 3) + 2,
      0.0001);
    // Check time
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset,
      t_pre(clusters[clIndex + 3]) / qtot + (clIndex + 3) * 10 + 2,
      0.0001);
    // Check pad sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaPad2(),
      (sigma_p_pre(clusters[clIndex + 3]) / qtot) - ((p_pre(clusters[clIndex + 3]) * p_pre(clusters[clIndex + 3])) / (qtot * qtot)),
      0.0001);
    // Check time sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaTime2(),
      (sigma_t_pre(clusters[clIndex + 3]) / qtot) - ((t_pre(clusters[clIndex + 3]) * t_pre(clusters[clIndex + 3])) / (qtot * qtot)),
      0.0001);
  }

  std::cout << "## Test 4 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}

/// @brief Test 5 Reject peaks in subsequent timebins
BOOST_AUTO_TEST_CASE(HwClusterer_test5)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 5, rejecting peaks in subsequent." << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<o2::tpc::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::tpc::HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = std::make_unique<std::vector<o2::tpc::Digit>>();
  // Digit(int cru, float charge, int row, int pad, int time)
  // two peaks, with the greater one afterwards
  std::array<std::array<int, 25>, 2> clusters;
  clusters[0] = { { 0, 0, 6, 0, 0,
                    0, 0, 23, 0, 0,
                    0, 0, 7, 0, 0,
                    0, 0, 77, 0, 0,
                    0, 0, 5, 0, 0 } };

  // two peaks, with the greater one first
  clusters[1] = { { 0, 0, 6, 0, 0,
                    0, 0, 67, 0, 0,
                    0, 0, 7, 0, 0,
                    0, 0, 13, 0, 0,
                    0, 0, 5, 0, 0 } };

  for (int dp = 0; dp < 5; ++dp) {
    for (int dt = 0; dt < 5; ++dt) {
      for (int cl = 0; cl < clusters.size(); ++cl) {
        digits->emplace_back(0, clusters[cl][dt * 5 + dp], cl, cl + dp, cl * 10 + dt);
      }
    }
  }

  std::sort(digits->begin(), digits->end(), sortTime());

  // Search clusters without rejection, all 4 clusters shoud be found
  std::cout << "testing without rejection..." << std::endl;
  clusterer.setRejectLaterTimebin(false);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  auto clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 4);
  BOOST_CHECK_EQUAL(clusterContainer->clusters[0].getQMax(), clusters[0][7]);
  BOOST_CHECK_EQUAL(clusterContainer->clusters[1].getQMax(), clusters[0][17]);
  BOOST_CHECK_EQUAL(clusterContainer->clusters[2].getQMax(), clusters[1][7]);
  BOOST_CHECK_EQUAL(clusterContainer->clusters[3].getQMax(), clusters[1][17]);

  // Search clusters with rejection, cluster with peak charge 13 should be surpressed
  std::cout << "testing with with rejection..." << std::endl;
  clusterer.setRejectLaterTimebin(true);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 3);
  BOOST_CHECK_EQUAL(clusterContainer->clusters[0].getQMax(), clusters[0][7]);
  BOOST_CHECK_EQUAL(clusterContainer->clusters[1].getQMax(), clusters[0][17]);
  BOOST_CHECK_EQUAL(clusterContainer->clusters[2].getQMax(), clusters[1][7]);

  std::cout << "## Test 5 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}

/// @brief Test 6 Split charge among nearby clusters
BOOST_AUTO_TEST_CASE(HwClusterer_test6)
{
  std::cout << "##" << std::endl;
  std::cout << "## Starting test 6, split charge among nearby clusters." << std::endl;
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_unique<std::vector<o2::tpc::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

  o2::tpc::HwClusterer clusterer(clusterArray.get(), 0, labelArray.get());
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = std::make_unique<std::vector<o2::tpc::Digit>>();
  // Digit(int cru, float charge, int row, int pad, int time)
  std::array<std::array<int, 100>, 6> clusters;
  // Just a single cluster
  clusters[0] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    4, 9, 12, 6, 1, 0, 0, 0, 0, 0,
                    8, 17, 25, 18, 5, 0, 0, 0, 0, 0,
                    13, 22, 50, 28, 14, 0, 0, 0, 0, 0,
                    9, 16, 27, 19, 7, 0, 0, 0, 0, 0,
                    2, 7, 11, 6, 3, 0, 0, 0, 0, 0 } };

  // Two clusters next to each other in pad direction, peaks well separated
  clusters[1] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    4, 9, 12, 6, 1, 4, 9, 12, 6, 1,
                    8, 17, 25, 18, 5, 8, 17, 25, 18, 5,
                    13, 22, 51, 28, 14, 13, 22, 52, 28, 15,
                    9, 16, 27, 19, 7, 9, 16, 27, 19, 7,
                    2, 7, 11, 6, 3, 2, 7, 11, 6, 3 } };

  // Two clusters next to each other in pad direction, peaks 4 pads appart
  clusters[2] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    4, 9, 12, 6, 1, 9, 12, 6, 1, 0,
                    8, 17, 25, 18, 5, 17, 25, 18, 5, 0,
                    13, 22, 53, 28, 14, 22, 54, 28, 15, 0,
                    9, 16, 27, 19, 7, 16, 27, 19, 7, 0,
                    2, 7, 11, 6, 3, 7, 11, 6, 3, 0 } };

  // Two clusters next to each other in pad direction, peaks 3 pads appartd
  clusters[3] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 4, 9, 12, 6, 9, 12, 6, 1, 0,
                    0, 8, 17, 25, 18, 17, 25, 18, 5, 0,
                    0, 13, 22, 55, 28, 22, 56, 28, 15, 0,
                    0, 9, 16, 27, 19, 16, 27, 19, 7, 0,
                    0, 2, 7, 11, 6, 7, 11, 6, 3, 0 } };

  // Two clusters next to each other in pad direction, peaks 2 pads appartd
  clusters[4] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 4, 9, 12, 6, 12, 6, 1, 0, 0,
                    0, 8, 17, 25, 18, 25, 18, 5, 0, 0,
                    0, 13, 22, 57, 28, 58, 28, 15, 0, 0,
                    0, 9, 16, 27, 19, 27, 19, 7, 0, 0,
                    0, 2, 7, 11, 6, 11, 6, 3, 0, 0 } };

  // Two clusters next to each other in diagonal direction
  clusters[5] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 4, 9, 12, 6, 1, 0,
                    0, 0, 0, 0, 8, 17, 25, 18, 5, 0,
                    0, 0, 0, 0, 13, 22, 59, 28, 15, 0,
                    4, 9, 12, 6, 9, 16, 27, 19, 7, 0,
                    8, 17, 25, 18, 2, 7, 11, 6, 3, 0,
                    13, 22, 60, 28, 14, 0, 0, 0, 0, 0,
                    9, 16, 27, 19, 7, 0, 0, 0, 0, 0,
                    2, 7, 11, 6, 3, 0, 0, 0, 0, 0 } };

  std::array<std::array<float, 25>, 11> clustersMode0;
  clustersMode0[0] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         13, 22, 50, 28, 14,
                         9, 16, 27, 19, 7,
                         2, 7, 11, 6, 3 } };

  clustersMode0[1] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         13, 22, 51, 28, 14,
                         9, 16, 27, 19, 7,
                         2, 7, 11, 6, 3 } };

  clustersMode0[2] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         13, 22, 52, 28, 15,
                         9, 16, 27, 19, 7,
                         2, 7, 11, 6, 3 } };

  clustersMode0[3] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         13, 22, 53, 28, 14,
                         9, 16, 27, 19, 7,
                         2, 7, 11, 6, 3 } };

  clustersMode0[4] = { { 1, 9, 12, 6, 1,
                         5, 17, 25, 18, 5,
                         14, 22, 54, 28, 15,
                         7, 16, 27, 19, 7,
                         3, 7, 11, 6, 3 } };

  clustersMode0[5] = { { 4, 9, 12, 6, 9,
                         8, 17, 25, 18, 17,
                         13, 22, 55, 28, 22,
                         9, 16, 27, 19, 16,
                         2, 7, 11, 6, 7 } };

  clustersMode0[6] = { { 6, 9, 12, 6, 1,
                         18, 17, 25, 18, 5,
                         28, 22, 56, 28, 15,
                         19, 16, 27, 19, 7,
                         6, 7, 11, 6, 3 } };

  clustersMode0[7] = { { 4, 9, 12, 6, 12,
                         8, 17, 25, 18, 25,
                         13, 22, 57, 28, 58,
                         9, 16, 27, 19, 27,
                         2, 7, 11, 6, 11 } };

  clustersMode0[8] = { { 12, 6, 12, 6, 1,
                         25, 18, 25, 18, 5,
                         57, 28, 58, 28, 15,
                         27, 19, 27, 19, 7,
                         11, 6, 11, 6, 3 } };

  clustersMode0[9] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         13, 22, 59, 28, 15,
                         9, 16, 27, 19, 7,
                         2, 7, 11, 6, 3 } };

  clustersMode0[10] = { { 4, 9, 12, 6, 9,
                          8, 17, 25, 18, 2,
                          13, 22, 60, 28, 14,
                          9, 16, 27, 19, 7,
                          2, 7, 11, 6, 3 } };

  std::array<std::array<float, 25>, 11> clustersMode1;
  clustersMode1[0] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         13, 22, 50, 28, 14,
                         9, 16, 27, 19, 7,
                         2, 7, 11, 6, 3 } };

  clustersMode1[1] = { { 4, 9, 12, 6, 0.5,
                         8, 17, 25, 18, 2.5,
                         13, 22, 51, 28, 14,
                         9, 16, 27, 19, 3.5,
                         2, 7, 11, 6, 3 } };

  clustersMode1[2] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         6.5, 22, 52, 28, 15,
                         9, 16, 27, 19, 7,
                         1, 7, 11, 6, 3 } };

  clustersMode1[3] = { { 4, 9, 12, 6, 0.5,
                         8, 17, 25, 18, 2.5,
                         13, 22, 53, 28, 7,
                         9, 16, 27, 19, 3.5,
                         2, 7, 11, 6, 1.5 } };

  clustersMode1[4] = { { 0.5, 9, 12, 6, 1,
                         2.5, 17, 25, 18, 5,
                         7, 22, 54, 28, 15,
                         3.5, 16, 27, 19, 7,
                         1.5, 7, 11, 6, 3 } };

  clustersMode1[5] = { { 4, 9, 12, 6, 0,
                         8, 17, 25, 18, 8.5,
                         13, 22, 55, 28, 11,
                         9, 16, 27, 19, 8,
                         2, 7, 11, 6, 0 } };

  clustersMode1[6] = { { 0, 9, 12, 6, 1,
                         0, 8.5, 25, 18, 5,
                         0, 11, 56, 28, 15,
                         0, 8, 27, 19, 7,
                         0, 7, 11, 6, 3 } };

  clustersMode1[7] = { { 4, 9, 12, 6, 0,
                         8, 17, 25, 9, 0,
                         13, 22, 57, 14, 0,
                         9, 16, 27, 9.5, 0,
                         2, 7, 11, 6, 0 } };

  clustersMode1[8] = { { 0, 6, 12, 6, 1,
                         0, 9, 25, 18, 5,
                         0, 14, 58, 28, 15,
                         0, 9.5, 27, 19, 7,
                         0, 6, 11, 6, 3 } };

  clustersMode1[9] = { { 4, 9, 12, 6, 1,
                         8, 17, 25, 18, 5,
                         13, 22, 59, 28, 15,
                         4.5, 16, 27, 19, 7,
                         1, 3.5, 11, 6, 3 } };

  clustersMode1[10] = { { 4, 9, 12, 3, 0,
                          8, 17, 25, 18, 1,
                          13, 22, 60, 28, 14,
                          9, 16, 27, 19, 7,
                          2, 7, 11, 6, 3 } };

  for (int cl = 0; cl < clusters.size(); ++cl) {
    for (int dt = 0; dt < 10; ++dt) {
      for (int dp = 0; dp < 10; ++dp) {
        // Digit(int cru, float charge, int row, int pad, int time)
        digits->emplace_back(0, clusters[cl][dt * 10 + dp], cl, dp, dt + 20 * cl);
      }
    }
  }

  std::sort(digits->begin(), digits->end(), sortTime());

  std::cout << "testing without splitting..." << std::endl;

  std::vector<ClusterHardware> clustersToCompare;
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7, p_pre_fp(clustersMode0[0]), t_pre_fp(clustersMode0[0]), sigma_p_pre_fp(clustersMode0[0]), sigma_t_pre_fp(clustersMode0[0]), (clustersMode0[0][12] * 2), (std::accumulate(clustersMode0[0].begin(), clustersMode0[0].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7 + 20, p_pre_fp(clustersMode0[1]), t_pre_fp(clustersMode0[1]), sigma_p_pre_fp(clustersMode0[1]), sigma_t_pre_fp(clustersMode0[1]), (clustersMode0[1][12] * 2), (std::accumulate(clustersMode0[1].begin(), clustersMode0[1].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 5, 7 + 20, p_pre_fp(clustersMode0[2]), t_pre_fp(clustersMode0[2]), sigma_p_pre_fp(clustersMode0[2]), sigma_t_pre_fp(clustersMode0[2]), (clustersMode0[2][12] * 2), (std::accumulate(clustersMode0[2].begin(), clustersMode0[2].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7 + 40, p_pre_fp(clustersMode0[3]), t_pre_fp(clustersMode0[3]), sigma_p_pre_fp(clustersMode0[3]), sigma_t_pre_fp(clustersMode0[3]), (clustersMode0[3][12] * 2), (std::accumulate(clustersMode0[3].begin(), clustersMode0[3].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 4, 7 + 40, p_pre_fp(clustersMode0[4]), t_pre_fp(clustersMode0[4]), sigma_p_pre_fp(clustersMode0[4]), sigma_t_pre_fp(clustersMode0[4]), (clustersMode0[4][12] * 2), (std::accumulate(clustersMode0[4].begin(), clustersMode0[4].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 1, 7 + 60, p_pre_fp(clustersMode0[5]), t_pre_fp(clustersMode0[5]), sigma_p_pre_fp(clustersMode0[5]), sigma_t_pre_fp(clustersMode0[5]), (clustersMode0[5][12] * 2), (std::accumulate(clustersMode0[5].begin(), clustersMode0[5].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 4, 7 + 60, p_pre_fp(clustersMode0[6]), t_pre_fp(clustersMode0[6]), sigma_p_pre_fp(clustersMode0[6]), sigma_t_pre_fp(clustersMode0[6]), (clustersMode0[6][12] * 2), (std::accumulate(clustersMode0[6].begin(), clustersMode0[6].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 1, 7 + 80, p_pre_fp(clustersMode0[7]), t_pre_fp(clustersMode0[7]), sigma_p_pre_fp(clustersMode0[7]), sigma_t_pre_fp(clustersMode0[7]), (clustersMode0[7][12] * 2), (std::accumulate(clustersMode0[7].begin(), clustersMode0[7].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 3, 7 + 80, p_pre_fp(clustersMode0[8]), t_pre_fp(clustersMode0[8]), sigma_p_pre_fp(clustersMode0[8]), sigma_t_pre_fp(clustersMode0[8]), (clustersMode0[8][12] * 2), (std::accumulate(clustersMode0[8].begin(), clustersMode0[8].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 4, 7 + 97, p_pre_fp(clustersMode0[9]), t_pre_fp(clustersMode0[9]), sigma_p_pre_fp(clustersMode0[9]), sigma_t_pre_fp(clustersMode0[9]), (clustersMode0[9][12] * 2), (std::accumulate(clustersMode0[9].begin(), clustersMode0[9].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7 + 100, p_pre_fp(clustersMode0[10]), t_pre_fp(clustersMode0[10]), sigma_p_pre_fp(clustersMode0[10]), sigma_t_pre_fp(clustersMode0[10]), (clustersMode0[10][12] * 2), (std::accumulate(clustersMode0[10].begin(), clustersMode0[10].end(), 0.0) * 16), 0, 0);
  clusterer.setSplittingMode(0);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  auto clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 11);
  for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
    std::cout << "Testing cluster " << clIndex << std::endl;
    // Checking row and flags are not relevant for this test
    // Check QTot
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(), clustersToCompare[clIndex].getQTot());
    // Check QMax
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(), clustersToCompare[clIndex].getQMax());
    // Check pad
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getPad(),
      clustersToCompare[clIndex].getPad(),
      0.0001);
    // Check time
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset,
      clustersToCompare[clIndex].getTimeLocal(),
      0.0001);
    // Check pad sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaPad2(),
      clustersToCompare[clIndex].getSigmaPad2(),
      0.0001);
    // Check time sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaTime2(),
      clustersToCompare[clIndex].getSigmaTime2(),
      0.0001);
  }

  std::cout << "testing splitting mode 1..." << std::endl;
  clustersToCompare.clear();
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7, p_pre_fp(clustersMode1[0]), t_pre_fp(clustersMode1[0]), sigma_p_pre_fp(clustersMode1[0]), sigma_t_pre_fp(clustersMode1[0]), (clustersMode1[0][12] * 2), (std::accumulate(clustersMode1[0].begin(), clustersMode1[0].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7 + 20, p_pre_fp(clustersMode1[1]), t_pre_fp(clustersMode1[1]), sigma_p_pre_fp(clustersMode1[1]), sigma_t_pre_fp(clustersMode1[1]), (clustersMode1[1][12] * 2), (std::accumulate(clustersMode1[1].begin(), clustersMode1[1].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 5, 7 + 20, p_pre_fp(clustersMode1[2]), t_pre_fp(clustersMode1[2]), sigma_p_pre_fp(clustersMode1[2]), sigma_t_pre_fp(clustersMode1[2]), (clustersMode1[2][12] * 2), (std::accumulate(clustersMode1[2].begin(), clustersMode1[2].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7 + 40, p_pre_fp(clustersMode1[3]), t_pre_fp(clustersMode1[3]), sigma_p_pre_fp(clustersMode1[3]), sigma_t_pre_fp(clustersMode1[3]), (clustersMode1[3][12] * 2), (std::accumulate(clustersMode1[3].begin(), clustersMode1[3].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 4, 7 + 40, p_pre_fp(clustersMode1[4]), t_pre_fp(clustersMode1[4]), sigma_p_pre_fp(clustersMode1[4]), sigma_t_pre_fp(clustersMode1[4]), (clustersMode1[4][12] * 2), (std::accumulate(clustersMode1[4].begin(), clustersMode1[4].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 1, 7 + 60, p_pre_fp(clustersMode1[5]), t_pre_fp(clustersMode1[5]), sigma_p_pre_fp(clustersMode1[5]), sigma_t_pre_fp(clustersMode1[5]), (clustersMode1[5][12] * 2), (std::accumulate(clustersMode1[5].begin(), clustersMode1[5].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 4, 7 + 60, p_pre_fp(clustersMode1[6]), t_pre_fp(clustersMode1[6]), sigma_p_pre_fp(clustersMode1[6]), sigma_t_pre_fp(clustersMode1[6]), (clustersMode1[6][12] * 2), (std::accumulate(clustersMode1[6].begin(), clustersMode1[6].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 1, 7 + 80, p_pre_fp(clustersMode1[7]), t_pre_fp(clustersMode1[7]), sigma_p_pre_fp(clustersMode1[7]), sigma_t_pre_fp(clustersMode1[7]), (clustersMode1[7][12] * 2), (std::accumulate(clustersMode1[7].begin(), clustersMode1[7].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 3, 7 + 80, p_pre_fp(clustersMode1[8]), t_pre_fp(clustersMode1[8]), sigma_p_pre_fp(clustersMode1[8]), sigma_t_pre_fp(clustersMode1[8]), (clustersMode1[8][12] * 2), (std::accumulate(clustersMode1[8].begin(), clustersMode1[8].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2 + 4, 7 + 97, p_pre_fp(clustersMode1[9]), t_pre_fp(clustersMode1[9]), sigma_p_pre_fp(clustersMode1[9]), sigma_t_pre_fp(clustersMode1[9]), (clustersMode1[9][12] * 2), (std::accumulate(clustersMode1[9].begin(), clustersMode1[9].end(), 0.0) * 16), 0, 0);
  clustersToCompare.emplace_back();
  clustersToCompare.back().setCluster(2, 7 + 100, p_pre_fp(clustersMode1[10]), t_pre_fp(clustersMode1[10]), sigma_p_pre_fp(clustersMode1[10]), sigma_t_pre_fp(clustersMode1[10]), (clustersMode1[10][12] * 2), (std::accumulate(clustersMode1[10].begin(), clustersMode1[10].end(), 0.0) * 16), 0, 0);
  clusterer.setSplittingMode(1);
  clusterer.process(*digits.get(), nullptr);
  BOOST_CHECK_EQUAL(clusterArray->size(), 1);
  clusterContainer = (*clusterArray)[0].getContainer();
  BOOST_CHECK_EQUAL(clusterContainer->numberOfClusters, 11);
  for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
    std::cout << "Testing cluster " << clIndex << std::endl;
    // Checking row and flags are not relevant for this test
    // Check QTot
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(), clustersToCompare[clIndex].getQTot());
    // Check QMax
    BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(), clustersToCompare[clIndex].getQMax());
    // Check pad
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getPad(),
      clustersToCompare[clIndex].getPad(),
      0.0001);
    // Check time
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset,
      clustersToCompare[clIndex].getTimeLocal(),
      0.0001);
    // Check pad sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaPad2(),
      clustersToCompare[clIndex].getSigmaPad2(),
      0.0001);
    // Check time sigma
    BOOST_CHECK_CLOSE(
      clusterContainer->clusters[clIndex].getSigmaTime2(),
      clustersToCompare[clIndex].getSigmaTime2(),
      0.0001);
  }

  std::cout << "## Test 6 done." << std::endl;
  std::cout << "##" << std::endl
            << std::endl;
}
}
}
