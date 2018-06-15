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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TPCBase/Digit.h"
#include "TPCBase/Mapper.h"
#include "TPCReconstruction/HwClusterer.h"

#include "DataFormatsTPC/Helpers.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <vector>
#include <memory>
#include <iostream>
#endif

struct sortTime {
  inline bool operator()(const o2::TPC::Digit& d1, const o2::TPC::Digit& d2)
  {
    return (d1.getTimeStamp() < d2.getTimeStamp());
  }
};

void testTPCHwClusterer()
{
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  auto clusterArray = std::make_shared<std::vector<o2::TPC::ClusterHardwareContainer8kb>>();
  auto labelArray = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  int sector = 0;

  o2::TPC::HwClusterer clusterer(clusterArray, labelArray, sector);
  // If continuous readout is false, all clusters are written directly to the output
  clusterer.setContinuousReadout(false);

  auto digits = make_unique<std::vector<o2::TPC::Digit>>();

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
    std::cout << "Created " << clusterPerRegionGenerated[region] << " clusters in region " << region << std::endl;
  }
  std::sort(digits->begin(), digits->end(), sortTime());

  // Search clusters
  clusterer.Process(*digits.get(), nullptr, 0);

  // Check outcome
  std::cout << "ClusterArray size: " << clusterArray->size() << std::endl;
  //  BOOST_CHECK_EQUAL(clusterArray->size(), 47);

  // check if all clusters were found
  std::vector<unsigned> clusterPerRegionFound(10, 0);
  for (auto& cont : *clusterArray) {
    auto clusterContainer = cont.getContainer();
    clusterPerRegionFound[clusterContainer->CRU] += clusterContainer->numberOfClusters;
  }
  for (int region = 0; region < 10; ++region) {
    if (clusterPerRegionFound[region] != clusterPerRegionGenerated[region]) {
      std::cout << "In region " << region << " were " << clusterPerRegionGenerated[region]
                << " clusters generated, but only " << clusterPerRegionFound[region] << " found." << std::endl;
    }
    //    BOOST_CHECK_EQUAL(clusterPerRegionFound[region],clusterPerRegionGenerated[region]);
  }

  // check if all cluster charges (tot and max) are 20
  for (auto& cont : *clusterArray) {
    auto clusterContainer = cont.getContainer();
    for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
      if (clusterContainer->clusters[clIndex].getQTot() != 20)
        std::cout << "cluster has wrong tot charge: " << clusterContainer->clusters[clIndex].getQTot() << std::endl;
      if (clusterContainer->clusters[clIndex].getQMax() != 20)
        std::cout << "cluster has wrong max charge: " << clusterContainer->clusters[clIndex].getQMax() << std::endl;
      //      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQTot(),20);
      //      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getQMax(),20);
    }
  }

  // check the pad and time positions of the clusters and sigmas (all 0 because single pad lcusters)
  for (auto& cont : *clusterArray) {
    auto clusterContainer = cont.getContainer();
    for (int clIndex = 0; clIndex < clusterContainer->numberOfClusters; ++clIndex) {
      if ((clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset) / 7 != clusterContainer->clusters[clIndex].getPad())
        std::cout << "something with pad/time recovering was wrong" << std::endl;
      //      BOOST_CHECK_EQUAL((clusterContainer->clusters[clIndex].getTimeLocal()+clusterContainer->timeBinOffset)/7,
      //          clusterContainer->clusters[clIndex].getPad());
      //      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getSigmaPad2(),0);
      //      BOOST_CHECK_EQUAL(clusterContainer->clusters[clIndex].getSigmaTime2(),0);
      std::cout
        << clusterContainer->CRU << " "
        << clusterContainer->clusters[clIndex].getRow() << " "
        << clusterContainer->clusters[clIndex].getPad() << " "
        << clusterContainer->clusters[clIndex].getTimeLocal() + clusterContainer->timeBinOffset << " "
        << clusterContainer->timeBinOffset << " "
        << clusterContainer->clusters[clIndex].getSigmaPad2() << " "
        << clusterContainer->clusters[clIndex].getSigmaTime2() << " "
        << std::endl;
    }
  }
}
