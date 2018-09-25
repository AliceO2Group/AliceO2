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
DataProcessorSpec getClusterConverterSpec(bool sendMC, int fanNumber)
{
  std::string processorName = "tpc-cluster-converter";
  o2::header::DataHeader::SubSpecificationType fanSpec = 0;
  if (fanNumber < 0) {
    // only one instance
    fanNumber = 0;
  } else {
    // multiple instances, add number to name
    processorName += std::to_string(fanNumber);
    fanSpec = fanNumber;
  }

  auto initFunction = [sendMC, fanSpec](InitContext& ic) {
    // there is nothing to init at the moment
    auto verbosity = 0;

    auto processingFct = [verbosity, sendMC, fanSpec](ProcessingContext& pc) {
      // init the stacks for forwarding the sector header
      // FIXME check if there is functionality in the DPL to forward the stack
      // FIXME make one function
      o2::header::Stack rawHeaderStack;
      o2::header::Stack mcHeaderStack;
      o2::TPC::TPCSectorHeader const* sectorHeaderMC = nullptr;
      if (sendMC) {
        sectorHeaderMC = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("mclblin"));
        if (sectorHeaderMC) {
          o2::header::Stack actual{ *sectorHeaderMC };
          std::swap(mcHeaderStack, actual);
          if (sectorHeaderMC->sector < 0) {
            pc.outputs().snapshot(OutputRef{ "mclblout", fanSpec, std::move(mcHeaderStack) }, fanSpec);
          }
        }
      }
      auto const* sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get("clusterin"));
      if (sectorHeader) {
        o2::header::Stack actual{ *sectorHeader };
        std::swap(rawHeaderStack, actual);
        if (sectorHeader->sector < 0) {
          pc.outputs().snapshot(OutputRef{ "clusterout", fanSpec, std::move(rawHeaderStack) }, fanSpec);
          return;
        }
      }
      assert(sectorHeaderMC == nullptr || sectorHeader->sector == sectorHeaderMC->sector);

      // this will return a span of TPC clusters
      auto inClusters = pc.inputs().get<std::vector<o2::TPC::Cluster>>("clusterin");
      int nClusters = inClusters.size();
      LOG(INFO) << "got clusters from input: " << nClusters;

      // MC labels are received as one container of labels in the sequence matching clusters
      // in the input
      std::unique_ptr<const MCLabelContainer> mcin;
      MCLabelContainer mcout;
      if (sendMC) {
        mcin = std::move(pc.inputs().get<MCLabelContainer*>("mclblin"));
      }

      // clusters need to be sorted to write clusters of one CRU to raw pages
      struct ClusterMapper {
        std::vector<unsigned> clusterIds;
        ClusterHardwareContainer metrics;
      };
      std::map<uint16_t, ClusterMapper> mapper;
      for (unsigned clusterIndex = 0; clusterIndex < nClusters; clusterIndex++) {
        // clusters are supposed to be sorted in CRU number, start a new entry
        // for every new CRU, a map is needed if the clusters are not ordered
        auto inputcluster = inClusters[clusterIndex];
        uint16_t CRU = inputcluster.getCRU();
        if (mapper.find(CRU) == mapper.end()) {
          if (verbosity > 0) {
            LOG(INFO) << "Inserting cluster set for CRU " << CRU;
          }
          mapper[CRU] = ClusterMapper{ {}, ClusterHardwareContainer{ 0, 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF, 0, CRU } };
        }
        mapper[CRU].clusterIds.emplace_back(clusterIndex);
        mapper[CRU].metrics.numberOfClusters++;
        if (inputcluster.getTimeMean() < mapper[CRU].metrics.timeBinOffset) {
          mapper[CRU].metrics.timeBinOffset = inputcluster.getTimeMean();
        }
      }

      ClusterHardwareContainer8kb clusterContainerMemory;
      auto maxClustersPerContainer = clusterContainerMemory.getMaxNumberOfClusters();
      unsigned nTotalPages = 0;
      // at this point there is at least one cluster per record
      for (const auto& m : mapper) {
        auto nPages = (m.second.metrics.numberOfClusters - 1) / maxClustersPerContainer + 1;
        if (verbosity > 0) {
          LOG(INFO) << "CRU " << std::setw(3) << m.first << " "                                //
                    << std::setw(3) << m.second.metrics.numberOfClusters << " cluster(s)"      //
                    << " in " << nPages << " page(s) of capacity " << maxClustersPerContainer; //
        }
        nTotalPages += nPages;
      }
      if (verbosity > 0) {
        LOG(INFO) << "allocating " << nTotalPages << " output page(s), " << nTotalPages * sizeof(ClusterHardwareContainer8kb);
      }
      auto outputPages = pc.outputs().make<ClusterHardwareContainer8kb>(OutputRef{ "clusterout", fanSpec, std::move(rawHeaderStack) }, nTotalPages);

      auto outputPageIterator = outputPages.begin();
      unsigned mcoutIndex = 0;
      for (auto& cruClusters : mapper) {
        LOG(DEBUG) << "processing CRU " << cruClusters.first;
        auto clusterIndex = cruClusters.second.clusterIds.begin();
        auto clusterIndexEnd = cruClusters.second.clusterIds.end();
        while (clusterIndex != clusterIndexEnd && outputPageIterator != outputPages.end()) {
          assert(cruClusters.second.metrics.numberOfClusters > 0);
          ClusterHardwareContainer* clusterContainer = outputPageIterator->getContainer();
          *clusterContainer = cruClusters.second.metrics;
          if (clusterContainer->numberOfClusters > maxClustersPerContainer) {
            clusterContainer->numberOfClusters = maxClustersPerContainer;
          }

          // write clusters of the page
          if (verbosity > 0) {
            LOG(INFO) << "writing " << std::setw(3) << clusterContainer->numberOfClusters << " cluster(s) of CRU "
                      << std::setw(3) << clusterContainer->CRU;
          }
          for (unsigned int clusterInPage = 0;
               clusterInPage < clusterContainer->numberOfClusters && clusterIndex != clusterIndexEnd;
               clusterInPage++, ++clusterIndex) {
            const auto& inputCluster = inClusters[*clusterIndex];
            auto& outputCluster = clusterContainer->clusters[clusterInPage];
            outputCluster.setCluster(inputCluster.getPadMean(),
                                     inputCluster.getTimeMean() - clusterContainer->timeBinOffset,
                                     inputCluster.getPadSigma() * inputCluster.getPadSigma(),
                                     inputCluster.getTimeSigma() * inputCluster.getTimeSigma(), inputCluster.getQmax(),
                                     inputCluster.getQ(), inputCluster.getRow(), 0);
            if (mcin) {
              // write the new sequence of MC labels
              for (auto const& label : mcin->getLabels(*clusterIndex)) {
                mcout.addElement(mcoutIndex, label);
              }
              ++mcoutIndex;
            }
          }
          cruClusters.second.metrics.numberOfClusters -= clusterContainer->numberOfClusters;
          ++outputPageIterator;
        }
        assert(cruClusters.second.metrics.numberOfClusters == 0);
      }
      if (mcin) {
        if (verbosity > 0) {
          LOG(INFO) << "writing MC labels for " << mcoutIndex << " cluster(s), index size " << mcout.getIndexedSize();
        }
        pc.outputs().snapshot(OutputRef{ "mclblout", fanSpec, std::move(mcHeaderStack) }, mcout);
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
  auto createInputSpecs = [fanSpec](bool makeMcInput) {
    std::vector<InputSpec> inputSpecs{
      InputSpec{ { "clusterin" }, gDataOriginTPC, "CLUSTERSIM", fanSpec, Lifetime::Timeframe },
    };
    if (makeMcInput) {
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERMCLBL");
      inputSpecs.emplace_back(InputSpec{ "mclblin", gDataOriginTPC, datadesc, fanSpec, Lifetime::Timeframe });
    }
    return std::move(inputSpecs);
  };

  auto createOutputSpecs = [fanSpec](bool makeMcOutput) {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{ { "clusterout" }, gDataOriginTPC, "CLUSTERHW", fanSpec, Lifetime::Timeframe },
    };
    if (makeMcOutput) {
      OutputLabel label{ "mclblout" };
      // FIXME: define common data type specifiers
      constexpr o2::header::DataDescription datadesc("CLUSTERHWMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, fanSpec, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{ processorName,
                            { createInputSpecs(sendMC) },
                            { createOutputSpecs(sendMC) },
                            AlgorithmSpec(initFunction) };
}
} // end namespace TPC
} // end namespace o2
