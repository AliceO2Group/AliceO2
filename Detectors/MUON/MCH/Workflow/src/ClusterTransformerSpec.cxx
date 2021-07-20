// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ClusterTransformerSpec.h"

#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MathUtils/Cartesian.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace std;
using namespace o2::framework;
namespace fs = std::filesystem;

namespace o2::mch
{

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
    std::string ext = fs::path(geoFile).extension();
    std::transform(ext.begin(), ext.begin(), ext.end(), [](unsigned char c) { return std::tolower(c); });

    if (ext == ".json") {
      std::ifstream in(geoFile);
      if (!in.is_open()) {
        throw std::invalid_argument("cannot open geometry file" + geoFile);
      }
      transformation = o2::mch::geo::transformationFromJSON(in);
    } else if (ext == ".root") {
      o2::base::GeometryManager::loadGeometry(geoFile);
      transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
    } else {
      throw std::invalid_argument("Geometry can only be in JSON or Root format");
    }
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

DataProcessorSpec getClusterTransformerSpec()
{
  std::string inputConfig = fmt::format("rofs:MCH/CLUSTERROFS;clusters:MCH/CLUSTERS");
  return DataProcessorSpec{
    "mch-clusters-transformer",
    Inputs{o2::framework::select(inputConfig.c_str())},
    Outputs{OutputSpec{{"globalclusters"}, "MCH", "GLOBALCLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterTransformerTask>()},
    Options{
      {"geometry", VariantType::String, o2::base::NameConf::getGeomFileName(), {"input geometry file (JSON or Root format)"}}}};
}

} // namespace o2::mch
