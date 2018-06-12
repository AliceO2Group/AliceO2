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
  int sector = 0;

  HwClusterer clusterer(clusterArray, labelArray, sector);
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
  int eventCount = 0;

  clusterer.Process(*digits.get(), mcDigitTruth.get(), eventCount);

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
}
}
