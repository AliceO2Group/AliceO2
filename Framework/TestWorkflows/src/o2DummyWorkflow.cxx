// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/runDataProcessing.h"
#include <Monitoring/Monitoring.h>
#include "Framework/Logger.h"

#include <chrono>

using Monitoring = o2::monitoring::Monitoring;
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

using DataHeader = o2::header::DataHeader;
using DataOrigin = o2::header::DataOrigin;

// This is how you can define your processing in a declarative way
std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&) {
  DataProcessorSpec timeframeReader{
    "reader",
    Inputs{},
    { OutputSpec{ { "tpc" }, "TPC", "CLUSTERS" },
      OutputSpec{ { "its" }, "ITS", "CLUSTERS" } },
    AlgorithmSpec{
      [](ProcessingContext& ctx) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        // Creates a new message of size 1000 which
        // has "TPC" as data origin and "CLUSTERS" as data description.
        auto tpcClusters = ctx.outputs().make<FakeCluster>(OutputRef{ "tpc" }, 1000);
        int i = 0;

        for (auto& cluster : tpcClusters) {
          // The assert is here simply because at some point we were allocating the
          // wrong number of items.
          assert(i < 1000);
          cluster.x = i;
          cluster.y = i;
          cluster.z = i;
          cluster.q = i;
          i++;
        }

        auto itsClusters = ctx.outputs().make<FakeCluster>(OutputRef{ "its" }, 1000);
        i = 0;
        for (auto& cluster : itsClusters) {
          assert(i < 1000);
          cluster.x = i;
          cluster.y = i;
          cluster.z = i;
          cluster.q = i;
          i++;
        }
        //       LOG(INFO) << "Invoked" << std::endl;
      } }
  };

  DataProcessorSpec tpcClusterSummary{
    "tpc-cluster-summary",
    { InputSpec{ "clusters", "TPC", "CLUSTERS"} },
    { OutputSpec{ {"summary"}, "TPC", "SUMMARY"} },
    AlgorithmSpec{ [](ProcessingContext& ctx) {
      auto tpcSummary = ctx.outputs().make<Summary>(OutputRef{"summary"}, 1);
      tpcSummary.at(0).inputCount = ctx.inputs().size();
    } },
    { ConfigParamSpec{ "some-cut", VariantType::Float, 1.0f, { "some cut" } } },
    { "CPUTimer" }
  };

  DataProcessorSpec itsClusterSummary{
    "its-cluster-summary",
    { InputSpec{ "clusters", "ITS", "CLUSTERS" } },
    {
      OutputSpec{ {"summary"}, "ITS", "SUMMARY" },
    },
    AlgorithmSpec{ [](ProcessingContext& ctx) {
      auto itsSummary = ctx.outputs().make<Summary>(OutputRef{"summary"}, 1);
      itsSummary.at(0).inputCount = ctx.inputs().size();
    } },
    { ConfigParamSpec{ "some-cut", VariantType::Float, 1.0f, { "some cut" } } },
    { "CPUTimer" }
  };

  DataProcessorSpec merger{
    "merger",
    {
      InputSpec{"clusters", "TPC", "CLUSTERS"},
      InputSpec{"summary", "TPC", "SUMMARY"},
      InputSpec{"other_summary", "ITS", "SUMMARY"}
    },
    Outputs{},
    AlgorithmSpec{
      [](ProcessingContext &ctx) {
        // We verify we got inputs in the correct order
        auto h0 = o2::header::get<DataHeader*>(ctx.inputs().get("clusters").header);
        auto h1 = o2::header::get<DataHeader*>(ctx.inputs().get("summary").header);
        auto h2 = o2::header::get<DataHeader*>(ctx.inputs().get("other_summary").header);
        // This should always be the case, since the 
        // test for an actual DataHeader should happen in the device itself.
        assert(h0 && h1 && h2);
        if (h0->dataOrigin != o2::header::DataOrigin("TPC")) {
          throw std::runtime_error("Unexpected data origin" + std::string(h0->dataOrigin.str));
        }

        if (h1->dataOrigin != o2::header::DataOrigin("TPC")) {
          throw std::runtime_error("Unexpected data origin" + std::string(h1->dataOrigin.str));
        }

        if (h2->dataOrigin != o2::header::DataOrigin("ITS")) {
          throw std::runtime_error("Unexpected data origin" + std::string(h2->dataOrigin.str));
        }

        auto& metrics = ctx.services().get<Monitoring>();
        metrics.send({ 1, "merger/invoked" });
        metrics.send({ (int)ctx.inputs().size(), "merger/inputs" });
      },
    }
  };
  return {
    timeframeReader,
    tpcClusterSummary,
    itsClusterSummary,
    merger
  };
}
