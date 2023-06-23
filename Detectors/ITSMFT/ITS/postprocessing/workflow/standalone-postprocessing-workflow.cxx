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

#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

// Include studies hereafter
#include "ITSStudies/ImpactParameter.h"
#include "ITSStudies/AvgClusSize.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"track-sources", VariantType::String, std::string{"ITS,ITS-TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC,ITS-TPC-TRD"}, {"comma-separated list of track sources to use"}},
    {"cluster-sources", VariantType::String, std::string{"ITS"}, {"comma-separated list of cluster sources to use"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    // {"niceparam-mc", o2::framework::VariantType::Bool, true, {"disable MC"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options, "o2_tfidinfo.root");
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  LOGP(info, "USEMC beginning {}", useMC);
  // LOGP(info, "USEMC NICEPARAM {}", configcontext.options().get<bool>("niceparam-mc"));
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  GID::mask_t itsSource = GID::getSourceMask(GID::ITS); // ITS tracks and clusters
  // specs.emplace_back(o2::its::study::getAvgClusSizeStudy(false, useMC)); // WHY TF DOESN'T THIS ERROR?  IT MATCHES NO CONSTRUCTOR FOR THIS FUNCTION WTF // its cuz 1 = default useMC value but ALSO ITS IS 1 ITS A COINKYDINK!!
  specs.emplace_back(o2::its::study::getAvgClusSizeStudy(itsSource, itsSource, useMC)); // technically this ignores the user input but whatever LMAO

  LOGP(info, "USEMC before {}", useMC);
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, itsSource, itsSource, itsSource, useMC, itsSource);
  // o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, itsSource, itsSource, itsSource, useMC);
  // o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC); // P-vertex is always needed
  o2::globaltracking::InputHelper::addInputSpecsSVertex(configcontext, specs);        // S-vertex is always needed
  LOGP(info, "USEMC after {}", useMC);
  // useMC = true;
  LOGP(info, "INPUTTEST {}", itsSource.to_ulong());


  // // Old method of extracting data: what is the difference?
  // GID::mask_t allowedSourcesTrc = GID::getSourcesMask("ITS,ITS-TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC,ITS-TPC-TRD");
  // GID::mask_t allowedSourcesClus = GID::getSourcesMask("ITS");

  // // Update the (declared) parameters if changed from the command line
  // o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // auto useMC = !configcontext.options().get<bool>("disable-mc");
  // GID::mask_t srcTrc = allowedSourcesTrc & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  // srcTrc |= GID::getSourcesMask("ITS"); // guarantee that we will at least use ITS tracks
  // GID::mask_t srcCls = allowedSourcesClus & GID::getSourcesMask(configcontext.options().get<std::string>("cluster-sources"));
  
  // o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcCls, srcTrc, srcTrc, useMC, srcCls); // useMC is actually default TRUE as disable-mc is default FALSE
  //                                                                                                      // SO bc the last param is GID:ALL, we are looking for MC data for everything
  //                                                                                                      // when we obviously don't have it.  so that's why shit breaks.
  // o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC);
  // o2::globaltracking::InputHelper::addInputSpecsSVertex(configcontext, specs);
  // // and yet it doesn't work if I either set addInputSpecs to use !useMC or set the two following arguments manually to 
  // // (GID::getSourcesMask(GID::NONE)... somehow it's still looking for MC data??
  // // Attempted putting useMC but srcTrc, srcClus, or GID::getSourceMask(GID::ITS) but none work (still dropping incomplete)
  // // I think the only possible solution is that getSourceMask and getSourcesMask is somehow working differently here

  // // Declare specs related to studies hereafter
  // // specs.emplace_back(o2::its::study::getImpactParameterStudy(srcTrc, srcCls, useMC));
  // specs.emplace_back(o2::its::study::getAvgClusSizeStudy(srcTrc, srcCls, useMC));
  // // ENd of old method
  
  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}