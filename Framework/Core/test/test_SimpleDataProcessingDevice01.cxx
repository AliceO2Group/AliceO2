// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/MetricsService.h"
#include "FairMQLogger.h"

using namespace o2::framework;

struct FakeCluster {
  float x;
  float y;
  float z;
  float q;
};

struct Summary {
  int inputCount;
  int clustersCount;
};

using DataHeader = o2::Header::DataHeader;

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;

// This is how you can define your processing in a declarative way
void defineDataProcessing(std::vector<DataProcessorSpec> &specs) {
  DataProcessorSpec timeframeReader{
    "reader",
    Inputs{},
    Outputs{
      {"TPC", "CLUSTERS", OutputSpec::Timeframe},
      {"ITS", "CLUSTERS", OutputSpec::Timeframe}
    },
    [](const std::vector<DataRef> inputs,
       ServiceRegistry& services,
       DataAllocator& allocator) {
       sleep(1);
       // Creates a new message of size 1000 which 
       // has "TPC" as data origin and "CLUSTERS" as data description.
       auto tpcClusters = allocator.newCollectionChunk<FakeCluster>(OutputSpec{"TPC", "CLUSTERS", 0}, 1000);
       int i = 0;

       for (auto &cluster : tpcClusters) {
         assert(i < 1000);
         cluster.x = i;
         cluster.y = i;
         cluster.z = i;
         cluster.q = i;
         i++;
       }

       auto itsClusters = allocator.newCollectionChunk<FakeCluster>(OutputSpec{"ITS", "CLUSTERS", 0}, 1000);
       i = 0;
       for (auto &cluster : itsClusters) {
         assert(i < 1000);
         cluster.x = i;
         cluster.y = i;
         cluster.z = i;
         cluster.q = i;
         i++;
       }
//       LOG(INFO) << "Invoked" << std::endl;
    }
  };

  DataProcessorSpec tpcClusterSummary {
    "tpc-cluster-summary",
    Inputs{
       {"TPC", "CLUSTERS", InputSpec::Timeframe}
    },
    Outputs{
       {"TPC", "SUMMARY", OutputSpec::Timeframe}
    },
    [](const std::vector<DataRef> inputs,
       ServiceRegistry& services,
       DataAllocator& allocator)
    {
      auto tpcSummary = allocator.newCollectionChunk<Summary>(OutputSpec{"TPC", "SUMMARY", 0}, 1);
      tpcSummary.at(0).inputCount = inputs.size();
    },
    {
      ConfigParamSpec{"some-cut", ConfigParamSpec::Float, 1.0}
    },
    {
      "CPUTimer"
    }
  };

  DataProcessorSpec itsClusterSummary {
    "its-cluster-summary",
    Inputs{
      {"ITS", "CLUSTERS", InputSpec::Timeframe}
    },
    Outputs{
      {"ITS", "SUMMARY", OutputSpec::Timeframe},
    },
    [](const std::vector<DataRef> inputs,
       ServiceRegistry& services,
       DataAllocator& allocator) {
      auto itsSummary = allocator.newCollectionChunk<Summary>(OutputSpec{"ITS", "SUMMARY", 0}, 1);
      itsSummary.at(0).inputCount = inputs.size();
    },
    {
      ConfigParamSpec{"some-cut", ConfigParamSpec::Float, 1.0}
    },
    {
      "CPUTimer"
    }
  };

  DataProcessorSpec merger{
    "merger",
    Inputs{
      {"TPC", "CLUSTERS", InputSpec::Timeframe},
      {"TPC", "SUMMARY", InputSpec::Timeframe},
      {"ITS", "SUMMARY", InputSpec::Timeframe}
    },
    Outputs{},
    [](const std::vector<DataRef> inputs,
       ServiceRegistry& services,
       DataAllocator& allocator) {
      // We verify we got inputs in the correct order
      auto h0 = reinterpret_cast<const DataHeader*>(inputs[0].header);
      auto h1 = reinterpret_cast<const DataHeader*>(inputs[1].header);
      auto h2 = reinterpret_cast<const DataHeader*>(inputs[2].header);
      if (h0->dataOrigin != o2::Header::DataOrigin("TPC")) {
        throw std::runtime_error("Unexpected data origin");
      }

      if (h1->dataOrigin != o2::Header::DataOrigin("TPC")) {
        throw std::runtime_error("Unexpected data origin");
      }

      if (h1->dataOrigin != o2::Header::DataOrigin("ITS")) {
        throw std::runtime_error("Unexpected data origin");
      }

      LOG(DEBUG) << "Consumer Invoked";
      LOG(DEBUG) << "Number of inputs" << inputs.size();
      auto &metrics = services.get<MetricsService>();
      metrics.post("merger/invoked", 1);
      metrics.post("merger/inputs", (int) inputs.size());
    },

  };
  specs.push_back(timeframeReader);
  specs.push_back(tpcClusterSummary);
  specs.push_back(itsClusterSummary);
  specs.push_back(merger);
}
