// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "O2GPUDPLDisplay.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "GPUO2InterfaceDisplay.h"
#include "GPUO2InterfaceConfiguration.h"

using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::gpu;

static std::unique_ptr<GPUO2InterfaceDisplay> display;
std::unique_ptr<GPUO2InterfaceConfiguration> config;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"enable-mc", o2::framework::VariantType::Bool, false, {"enable visualization of MC data"}},
    {"display-clusters", VariantType::String, "TPC", {"comma-separated list of clusters to display"}},
    {"display-tracks", VariantType::String, "TPC", {"comma-separated list of tracks to display"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

static o2::globaltracking::DataRequest dataRequest;

void O2GPUDPLDisplaySpec::init(InitContext& ic)
{
  const auto grp = o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName());
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP(o2::base::NameConf::getGRPFileName());
  config.reset(new GPUO2InterfaceConfiguration);
  config->configGRP.solenoidBz = 5.00668f * grp->getL3Current() / 30000.;
  config->configGRP.continuousMaxTimeBin = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC) ? -1 : 0; // Number of timebins in timeframe if continuous, 0 otherwise
  config->ReadConfigurableParam();
  display.reset(new GPUO2InterfaceDisplay(config.get()));
}

void O2GPUDPLDisplaySpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, dataRequest);
  static bool first = false;
  if (first == false) {
    display->startDisplay();
  }
  display->show();
}

void O2GPUDPLDisplaySpec::endOfStream(EndOfStreamContext& ec)
{
  display->endDisplay();
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;

  bool useMC = cfgc.options().get<bool>("enable-mc");
  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-tracks"));
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-clusters"));
  dataRequest.requestTracks(srcTrk, useMC);
  dataRequest.requestClusters(srcCl, useMC);

  specs.emplace_back(DataProcessorSpec{
    "o2-gpu-display",
    dataRequest.inputs,
    {},
    AlgorithmSpec{adaptFromTask<O2GPUDPLDisplaySpec>(useMC, srcTrk, srcCl)}});

  return std::move(specs);
}
