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

#include "MCHGeometryTransformer/ClusterTransformerSpec.h"

#include "DetectorsBase/GeometryManager.h"
#include "CommonUtils/NameConf.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsMCH/Cluster.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MathUtils/Cartesian.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>
#include "DetectorsBase/GRPGeomHelper.h"

using namespace std;
using namespace o2::framework;
namespace fs = std::filesystem;

namespace o2::mch
{

// convert all clusters from local to global reference frames
void local2global(geo::TransformationCreator transformation,
                  gsl::span<const Cluster> localClusters,
                  std::vector<Cluster, o2::pmr::polymorphic_allocator<Cluster>>& globalClusters)
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
  ClusterTransformerTask(std::shared_ptr<base::GRPGeomRequest> req) : mCcdbRequest(req)
  {
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (mCcdbRequest) {
      base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
      transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
    }
  }

  void readGeometryFromFile(std::string geoFile)
  {
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
      throw std::invalid_argument("Geometry from file can only be in JSON or Root format");
    }
  }

  void init(InitContext& ic)
  {
    if (mCcdbRequest) {
      base::GRPGeomHelper::instance().setRequest(mCcdbRequest);
    } else {
      auto geoFile = ic.options().get<std::string>("geometry");
      readGeometryFromFile(geoFile);
    }
  }

  // read the clusters (assumed to be in local reference frame) and
  // tranform them into master reference frame.
  void run(ProcessingContext& pc)
  {
    if (mCcdbRequest) {
      base::GRPGeomHelper::instance().checkUpdates(pc);
    }

    // get the input clusters
    auto localClusters = pc.inputs().get<gsl::span<Cluster>>("clusters");

    // create the output message
    auto& globalClusters = pc.outputs().make<std::vector<Cluster>>(OutputRef{"globalclusters"});

    local2global(transformation, localClusters, globalClusters);
  }

 public:
  o2::mch::geo::TransformationCreator transformation;
  std::shared_ptr<base::GRPGeomRequest> mCcdbRequest;
};

DataProcessorSpec getClusterTransformerSpec(const char* specName, bool disableCcdb)
{
  std::string inputConfig = fmt::format("rofs:MCH/CLUSTERROFS;clusters:MCH/CLUSTERS");
  auto inputs = o2::framework::select(inputConfig.c_str());

  auto ccdbRequest = disableCcdb ? nullptr : std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                                                        false,                             // GRPECS=true
                                                                                        false,                             // GRPLHCIF
                                                                                        false,                             // GRPMagField
                                                                                        false,                             // askMatLUT
                                                                                        o2::base::GRPGeomRequest::Aligned, // geometry
                                                                                        inputs);
  return DataProcessorSpec{
    specName,
    inputs,
    Outputs{OutputSpec{{"globalclusters"}, "MCH", "GLOBALCLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterTransformerTask>(ccdbRequest)},
    Options{
      {"geometry", VariantType::String, o2::base::NameConf::getGeomFileName(), {"input geometry file (JSON or Root format)"}}}};
}

} // namespace o2::mch
