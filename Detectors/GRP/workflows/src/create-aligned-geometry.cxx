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

#include <ctime>
#include <chrono>
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/NameConf.h"
#include <TGeoManager.h>
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CallbacksPolicy.h"
#include "Headers/STFHeader.h"
#include "DetectorsRaw/HBFUtils.h"

using DetID = o2::detectors::DetID;
using namespace o2::framework;

/*
  Simple workflow to create aligned geometry file from the ideal geometry and alignment objects on the CCDB
  One can profit from the DPL CCDB fetcher --condition-remap functionality to impose difference CCDB
  servers for different alignment objects.
  Also, one can use --configKeyValues "HBFUtils.startTime=... functionality to impose the timestamp for the objects to fetch.
  E.g. to create the geometry using current timestamp and production server, use just
  o2-create-aligned-geometry-workflow

  To create the geometry using specific timestamp and ITS alignment from the test CCDB server and MFT alignment
  from the local snapshot file in  the locCCDB directory
  o2-create-aligned-geometry-workflow --condition-remap http://ccdb-test.cern.ch:8080=ITS/Calib/Align;file://locCCDB=MFT/Calib/Align --configKeyValues "HBFUtils.startTime=1546300800000"
 */

namespace o2::base
{

class SeederTask : public Task
{
 public:
  void run(ProcessingContext& pc) final
  {
    const auto& hbfu = o2::raw::HBFUtils::Instance();
    auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    if (hbfu.startTime != 0) {
      tinfo.creation = hbfu.startTime;
    } else {
      tinfo.creation = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
    }
    if (hbfu.orbitFirstSampled != 0) {
      tinfo.firstTForbit = hbfu.orbitFirstSampled;
    } else {
      tinfo.firstTForbit = 0;
    }
    auto& stfDist = pc.outputs().make<o2::header::STFHeader>(Output{"FLP", "DISTSUBTIMEFRAME", 0});
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
};

class AlignerTask : public Task
{
 public:
  AlignerTask(DetID::mask_t dets, std::shared_ptr<o2::base::GRPGeomRequest> gr) : mDetsMask(dets), mGGCCDBRequest(gr) {}
  ~AlignerTask() override = default;

  void init(InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    for (auto id = DetID::First; id <= DetID::Last; id++) {
      if (mDetsMask[id]) {
        auto alg = o2::base::GRPGeomHelper::instance().getAlignment(id);
        if (!alg->empty()) {
          LOGP(info, "Applying alignment for detector {}", DetID::getName(id));
          o2::base::GeometryManager::applyAlignment(*alg);
        } else {
          LOGP(info, "Alignment for detector {} is empty", DetID::getName(id));
        }
      }
    }
    gGeoManager->SetName(std::string(o2::base::NameConf::CCDBOBJECT).c_str());
    auto fnm = o2::base::NameConf::getAlignedGeomFileName();
    gGeoManager->Export(fnm.c_str());
    LOG(info) << "Stored to local file " << fnm;
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 private:
  DetID::mask_t mDetsMask{};
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
};

} // namespace o2::base

DataProcessorSpec getAlignerSpec(DetID::mask_t dets)
{
  std::vector<InputSpec> inputs{{"STFDist", "FLP", "DISTSUBTIMEFRAME", 0}};                         // just to have some input
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                                // orbitResetTime
                                                              false,                                // GRPECS
                                                              false,                                // GRPLHCIF
                                                              false,                                // GRPMagField
                                                              false,                                // askMatLUT
                                                              o2::base::GRPGeomRequest::Alignments, // geometry
                                                              inputs);

  return DataProcessorSpec{
    "geometry-aligned-producer",
    inputs,
    {},
    AlgorithmSpec{adaptFromTask<o2::base::AlignerTask>(dets, ggRequest)},
    Options{}};
}

DataProcessorSpec getSeederSpec()
{
  return DataProcessorSpec{
    "seeder",
    Inputs{},
    Outputs{{"FLP", "DISTSUBTIMEFRAME", 0}},
    AlgorithmSpec{adaptFromTask<o2::base::SeederTask>()},
    Options{}};
}

// ------------------------------------------------------------------

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"onlyDet", VariantType::String, std::string{DetID::ALL}, {"comma-separated list of detectors to account"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  DetID::mask_t dets = DetID::FullMask & DetID::getMask(configcontext.options().get<std::string>("onlyDet"));
  o2::framework::WorkflowSpec specs{getAlignerSpec(dets), getSeederSpec()};

  return std::move(specs);
}
