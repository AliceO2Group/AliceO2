// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterConverterSpec.cxx
/// @author Matthias Richter
/// @since  2018-03-15
/// @brief  Processor spec for converter of TPC clusters to HW cluster raw data

#include "ClusterConverterSpec.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/Cluster.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <FairMQLogger.h>
#include <memory> // for make_shared
#include <vector>
#include <map>
#include <iomanip>

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace TPC
{

using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

/// create a processor spec
/// convert incoming TPC clusters to HW clusters
/// Note: This processor does not touch the MC, see below
DataProcessorSpec getClusterConverterSpec(bool sendMC)
{
  auto initFunction = [](InitContext& ic) {
    // there is nothing to init at the moment
    auto verbosity = 0;

    auto processingFct = [verbosity](ProcessingContext& pc) {
      // this will return a span of TPC clusters
      auto inClusters = pc.inputs().get<std::vector<o2::TPC::Cluster>>("clusterin");
      auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("clusterin"));
      o2::header::Stack headerStack;
      if (sectorHeader) {
        o2::header::Stack actual{ *sectorHeader };
        std::swap(headerStack, actual);
      }
      int nClusters = inClusters.size();
      LOG(INFO) << "got clusters from input: " << nClusters;

      std::map<uint16_t, ClusterHardwareContainer> containerMetrics;
      unsigned clusterIndex = 0;
      for (; clusterIndex < nClusters; clusterIndex++) {
        // clusters are supposed to be sorted in CRU number, start a new entry
        // for every new CRU, a map is needed if the clusters are not ordered
        auto inputcluster = inClusters[clusterIndex];
        uint16_t CRU = inputcluster.getCRU();
        if (containerMetrics.find(CRU) == containerMetrics.end()) {
          if (verbosity > 0) {
            LOG(INFO) << "Inserting cluster set for CRU " << CRU;
          }
          containerMetrics[CRU] = ClusterHardwareContainer{ 0, 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF, 0, CRU };
        }
        containerMetrics[CRU].numberOfClusters++;
        if (inputcluster.getTimeMean() < containerMetrics[CRU].timeBinOffset) {
          containerMetrics[CRU].timeBinOffset = inputcluster.getTimeMean();
        }
      }

      ClusterHardwareContainer8kb clusterContainerMemory;
      auto maxClustersPerContainer = clusterContainerMemory.getMaxNumberOfClusters();
      unsigned nTotalPages = 0;
      // at this point there is at least one cluster per record
      for (const auto& m : containerMetrics) {
        auto nPages = (m.second.numberOfClusters - 1) / maxClustersPerContainer + 1;
        if (verbosity > 0) {
          LOG(INFO) << "CRU " << std::setw(3) << m.first << " "                                //
                    << std::setw(3) << m.second.numberOfClusters << " cluster(s)"              //
                    << " in " << nPages << " page(s) of capacity " << maxClustersPerContainer; //
        }
        nTotalPages += nPages;
      }
      if (verbosity > 0) {
        LOG(INFO) << "allocating " << nTotalPages << " output page(s), " << nTotalPages * sizeof(ClusterHardwareContainer8kb);
      }
      auto outputPages = pc.outputs().make<ClusterHardwareContainer8kb>(OutputRef{ "clusterout", 0, std::move(headerStack) }, nTotalPages);

      auto containerMetricsIterator = containerMetrics.begin();
      auto outputPageIterator = outputPages.begin();
      clusterIndex = 0;
      while (clusterIndex < nClusters) {
        LOG(DEBUG) << "processing CRU " << containerMetricsIterator->second.CRU;
        while (containerMetricsIterator->second.numberOfClusters > 0) {
          ClusterHardwareContainer* clusterContainer = outputPageIterator->getContainer();
          *clusterContainer = containerMetricsIterator->second;
          if (clusterContainer->numberOfClusters > maxClustersPerContainer) {
            clusterContainer->numberOfClusters = maxClustersPerContainer;
          }

          // write clusters of the page
          if (verbosity > 0) {
            LOG(INFO) << "writing " << std::setw(3) << clusterContainer->numberOfClusters << " cluster(s) of CRU "
                      << std::setw(3) << clusterContainer->CRU;
          }
          for (unsigned int clusterInPage = 0; clusterInPage < clusterContainer->numberOfClusters; clusterInPage++) {
            const auto& inputCluster = inClusters[clusterIndex + clusterInPage];
            auto& outputCluster = clusterContainer->clusters[clusterInPage];
            outputCluster.setCluster(inputCluster.getPadMean(),
                                     inputCluster.getTimeMean() - clusterContainer->timeBinOffset,
                                     inputCluster.getPadSigma() * inputCluster.getPadSigma(),
                                     inputCluster.getTimeSigma() * inputCluster.getTimeSigma(), inputCluster.getQmax(),
                                     inputCluster.getQ(), inputCluster.getRow(), 0);
          }
          clusterIndex += clusterContainer->numberOfClusters;
          containerMetricsIterator->second.numberOfClusters -= clusterContainer->numberOfClusters;
          ++outputPageIterator;
        }
        ++containerMetricsIterator;
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  // FIXME: treatment of the MC data
  // the definition of the MC input and output won't work since the In/OutputSpec are the
  // same. DPL does the routing only on the basis of the specs. Furthermore, we also need
  // forwarding functionality in the DPL I/O API
  // as we do not expect any further sorting of clusters during the conversion, we do not
  // need to define the MC data at all, it is just routed directly to the final consumer.
  // Whether or not to have MC data is thus a feature of the initial producer.
  auto createInputSpecs = [](bool makeMcInput) {
    std::vector<InputSpec> inputSpecs{
      InputSpec{ { "clusterin" }, gDataOriginTPC, "CLUSTERSIM", 0, Lifetime::Timeframe },
    };
    if (makeMcInput) {
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERMCLBL");
      inputSpecs.emplace_back(InputSpec{ "mclblin", gDataOriginTPC, datadesc, 0, Lifetime::Timeframe });
    }
    return std::move(inputSpecs);
  };

  auto createOutputSpecs = [](bool makeMcOutput) {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{ { "clusterout" }, gDataOriginTPC, "CLUSTERHW", 0, Lifetime::Timeframe },
    };
    if (makeMcOutput) {
      OutputLabel label{ "mclblout" };
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{ "converter",
                            { createInputSpecs(false) },
                            { createOutputSpecs(false) },
                            AlgorithmSpec(initFunction) };
}
} // end namespace TPC
} // end namespace o2
