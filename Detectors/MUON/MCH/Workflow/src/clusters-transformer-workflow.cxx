// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <fstream>
#include <chrono>
#include <vector>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "MCHBase/ClusterBlock.h"
#include "Framework/runDataProcessing.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MathUtils/Cartesian.h"

using namespace std;
using namespace o2::framework;
using namespace o2::mch;

// convert all clusters from local to global reference frames
void local2global(geo::TransformationCreator transformation,
                  gsl::span<const ClusterStruct> localClusters,
                  std::vector<ClusterStruct, o2::pmr::polymorphic_allocator<ClusterStruct>>& globalClusters)
{
  int i{0};
  globalClusters.insert(globalClusters.end(), localClusters.begin(), localClusters.end());
  for (auto& c : localClusters) {
    auto deId = c.getDEId();
    o2::math_utils::Point3D<float> local{c.x, c.y, c.z};
    auto t = transformation(deId);
    auto global = t(local);
    auto& gcluster = globalClusters[i];
    gcluster.x = global.x();
    gcluster.y = global.y();
    gcluster.z = global.z();
    i++;
  }
}

class ClusterTransformerTask
{
 public:
  void init(InitContext& ic)
  {
    auto geoFile = ic.options().get<std::string>("geometry");
    std::ifstream in(geoFile);
    if (!in.is_open()) {
      throw std::invalid_argument("cannot open geometry file" + geoFile);
    }
    transformation = o2::mch::geo::transformationFromJSON(in);
  }

  // read the clusters (assumed to be in local reference frame) and
  // tranform them into master reference frame.
  void run(ProcessingContext& pc)
  {
    // get the input clusters
    auto localClusters = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");

    // create the output message
    auto& globalClusters = pc.outputs().make<std::vector<ClusterStruct>>(OutputRef{"globalclusters"});

    local2global(transformation, localClusters, globalClusters);
  }

 public:
  o2::mch::geo::TransformationCreator transformation;
};

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  std::string inputConfig = fmt::format("rofs:MCH/CLUSTERROFS;clusters:MCH/CLUSTERS");

  return WorkflowSpec{
    DataProcessorSpec{
      "mch-clusters-transformer",
      Inputs{o2::framework::select(inputConfig.c_str())},
      Outputs{OutputSpec{{"globalclusters"}, "MCH", "GLOBALCLUSTERS", 0, Lifetime::Timeframe}},
      AlgorithmSpec{adaptFromTask<ClusterTransformerTask>()},
      Options{
        {"geometry", VariantType::String, "geometry-o2.json", {"input geometry (json format)"}}}}};
}
