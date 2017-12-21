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
#include "Framework/AlgorithmSpec.h"
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

using DataHeader = o2::header::DataHeader;

// This is how you can define your processing in a declarative way
void defineDataProcessing(std::vector<DataProcessorSpec> &specs) {
  DataProcessorSpec simple{
    "simple",
    Inputs{},
    {
      OutputSpec{"TPC", "CLUSTERS", OutputSpec::Timeframe},
      OutputSpec{"ITS", "CLUSTERS", OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      [](ProcessingContext &ctx) {
        sleep(1);
        // Creates a new message of size 1000 which
        // has "TPC" as data origin and "CLUSTERS" as data description.
        auto tpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS", 0}, 1000);
        int i = 0;

        for (auto &cluster : tpcClusters) {
          assert(i < 1000);
          cluster.x = i;
          cluster.y = i;
          cluster.z = i;
          cluster.q = i;
          i++;
        }

        auto itsClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"ITS", "CLUSTERS", 0}, 1000);
        i = 0;
        for (auto &cluster : itsClusters) {
          assert(i < 1000);
          cluster.x = i;
          cluster.y = i;
          cluster.z = i;
          cluster.q = i;
          i++;
        }
      }
    }
  };

  specs.push_back(simple);
}
