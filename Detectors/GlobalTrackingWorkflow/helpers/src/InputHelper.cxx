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

/// @file  InputHelper.cxx

#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "MFTWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/ClusterReaderSpec.h"
#include "TPCReaderWorkflow/TriggerReaderSpec.h"
#include "TPCWorkflow/ClusterSharingMapSpec.h"
#include "HMPIDWorkflow/ClustersReaderSpec.h"
#include "HMPIDWorkflow/HMPMatchedReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/GlobalFwdTrackReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/MatchedMFTMCHReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/MatchedMCHMIDReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/PrimaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/SecondaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/StrangenessTrackingReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackCosmicsReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/IRFrameReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "FT0Workflow/RecPointReaderSpec.h"
#include "FV0Workflow/RecPointReaderSpec.h"
#include "FDDWorkflow/RecPointReaderSpec.h"
#include "ZDCWorkflow/RecEventReaderSpec.h"
#include "TRDWorkflowIO/TRDTrackletReaderSpec.h"
#include "TRDWorkflowIO/TRDTrackReaderSpec.h"
#include "CTPWorkflowIO/DigitReaderSpec.h"
#include "MCHIO/TrackReaderSpec.h"
#include "MCHIO/ClusterReaderSpec.h"
#include "MIDWorkflow/TrackReaderSpec.h"
#include "PHOSWorkflow/CellReaderSpec.h"
#include "CPVWorkflow/ClusterReaderSpec.h"
#include "EMCALWorkflow/PublisherSpec.h"
// #include "StrangenessTrackingWorkflow/StrangenessTrackingReaderSpec.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using namespace o2::dataformats;
using GID = o2::dataformats::GlobalTrackID;

int InputHelper::addInputSpecs(const ConfigContext& configcontext, WorkflowSpec& specs,
                               GID::mask_t maskClusters, GID::mask_t maskMatches, GID::mask_t maskTracks,
                               bool useMC, GID::mask_t maskClustersMC, GID::mask_t maskTracksMC,
                               bool subSpecStrict)
{
  if (configcontext.options().get<bool>("disable-root-input")) {
    return 0;
  }
  if (useMC && configcontext.options().get<bool>("disable-mc")) {
    useMC = false;
  }
  if (!useMC) {
    maskClustersMC = GID::getSourcesMask(GID::NONE);
    maskTracksMC = GID::getSourcesMask(GID::NONE);
  } else {
    // some detectors do not support MC labels
    if (maskClusters[GID::MCH] && maskClustersMC[GID::MCH]) {
      LOG(warn) << "MCH global clusters do not support MC lables, disabling";
      maskClustersMC &= ~GID::getSourceMask(GID::MCH);
    }
  }

  if (maskTracks[GID::ITS]) {
    specs.emplace_back(o2::its::getITSTrackReaderSpec(maskTracksMC[GID::ITS]));
  }
  if (maskClusters[GID::ITS]) {
    specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(maskClustersMC[GID::ITS], true));
  }
  if (maskTracks[GID::MFT]) {
    specs.emplace_back(o2::mft::getMFTTrackReaderSpec(maskTracksMC[GID::MFT]));
  }
  if (maskClusters[GID::MFT]) {
    specs.emplace_back(o2::itsmft::getMFTClusterReaderSpec(maskClustersMC[GID::MFT], true));
  }
  if (maskTracks[GID::MCH] || maskMatches[GID::MCHMID]) {
    specs.emplace_back(o2::mch::getTrackReaderSpec(maskTracksMC[GID::MCH] || maskTracksMC[GID::MCHMID]));
  }
  if (maskTracks[GID::MID]) {
    specs.emplace_back(o2::mid::getTrackReaderSpec(maskTracksMC[GID::MID]));
  }
  if (maskTracks[GID::TPC]) {
    specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(maskTracksMC[GID::TPC]));
  }
  if (maskClusters[GID::TPC]) {
    specs.emplace_back(o2::tpc::getClusterReaderSpec(maskClustersMC[GID::TPC]));
    if (!getenv("DPL_DISABLE_TPC_TRIGGER_READER") || atoi(getenv("DPL_DISABLE_TPC_TRIGGER_READER")) != 1) {
      specs.emplace_back(o2::tpc::getTPCTriggerReaderSpec());
    }
  }
  if (maskTracks[GID::TPC] && maskClusters[GID::TPC]) {
    specs.emplace_back(o2::tpc::getClusterSharingMapSpec());
  }
  if (maskMatches[GID::ITSTPC] || maskMatches[GID::ITSTPCTOF] || maskTracks[GID::ITSTPC] || maskTracks[GID::ITSTPCTOF]) {
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(maskTracksMC[GID::ITSTPC] || maskTracksMC[GID::ITSTPCTOF]));
  }
  if (maskMatches[GID::ITSTPCTOF] || maskTracks[GID::ITSTPCTOF]) {
    specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(maskTracksMC[GID::ITSTPCTOF], 1, /*maskTracks[GID::ITSTPCTOF]*/ false)); // ITSTPCTOF does not provide tracks, only matchInfo
  }
  if (maskMatches[GID::MFTMCH] || maskTracks[GID::MFTMCH]) {
    specs.emplace_back(o2::globaltracking::getGlobalFwdTrackReaderSpec(maskTracksMC[GID::MFTMCH])); // MFTMCH matches does not provide tracks, only matchInfo
  }
  if (maskMatches[GID::MCHMID] || maskTracks[GID::MCHMID]) {
    specs.emplace_back(o2::globaltracking::getMCHMIDMatchedReaderSpec(maskTracksMC[GID::MCHMID])); // MCHMID matches does not provide tracks, only matchInfo
  }
  if (maskMatches[GID::ITSTPCTRDTOF] || maskTracks[GID::ITSTPCTRDTOF]) {
    specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(maskTracksMC[GID::ITSTPCTRDTOF], 3, /*maskTracks[GID::ITSTPCTOF]*/ false)); // ITSTPCTOF does not provide tracks, only matchInfo
  }
  if (maskMatches[GID::TPCTRDTOF] || maskTracks[GID::TPCTRDTOF]) {
    specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(maskTracksMC[GID::TPCTRDTOF], 2, /*maskTracks[GID::ITSTPCTOF]*/ false)); // ITSTPCTOF does not provide tracks, only matchInfo
  }
  if (maskClusters[GID::TOF] ||
      maskTracks[GID::ITSTPCTOF] || maskTracks[GID::ITSTPCTRDTOF] || maskTracks[GID::TPCTRDTOF] ||
      maskMatches[GID::ITSTPCTOF] || maskMatches[GID::ITSTPCTRDTOF] || maskMatches[GID::TPCTRDTOF]) {
    specs.emplace_back(o2::tof::getClusterReaderSpec(maskClustersMC[GID::TOF]));
  }
  if (maskClusters[GID::HMP]) {
    specs.emplace_back(o2::hmpid::getClusterReaderSpec());
  }
  if (maskMatches[GID::TPCTOF] || maskTracks[GID::TPCTOF]) {
    specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(maskTracksMC[GID::TPCTOF], 0, maskTracks[GID::TPCTOF], subSpecStrict));
  }
  if (maskMatches[GID::HMP]) {
    specs.emplace_back(o2::hmpid::getHMPMatchedReaderSpec(maskTracksMC[GID::HMP]));
  }
  if (maskTracks[GID::FT0] || maskClusters[GID::FT0]) {
    specs.emplace_back(o2::ft0::getRecPointReaderSpec(maskTracksMC[GID::FT0] || maskClustersMC[GID::FT0]));
  }
  if (maskTracks[GID::FV0] || maskClusters[GID::FV0]) {
    specs.emplace_back(o2::fv0::getRecPointReaderSpec(maskTracksMC[GID::FV0] || maskClustersMC[GID::FV0]));
  }
  if (maskTracks[GID::FDD] || maskClusters[GID::FDD]) {
    specs.emplace_back(o2::fdd::getFDDRecPointReaderSpec(maskTracksMC[GID::FDD] || maskClustersMC[GID::FDD]));
  }
  if (maskTracks[GID::ZDC] || maskClusters[GID::ZDC]) {
    specs.emplace_back(o2::zdc::getRecEventReaderSpec(maskTracksMC[GID::ZDC] || maskClustersMC[GID::ZDC]));
  }

  if (maskClusters[GID::TRD]) {
    specs.emplace_back(o2::trd::getTRDTrackletReaderSpec(maskClustersMC[GID::TRD], true));
  }
  if (maskTracks[GID::ITSTPCTRD] || maskTracks[GID::ITSTPCTRDTOF]) {
    specs.emplace_back(o2::trd::getTRDGlobalTrackReaderSpec(maskTracksMC[GID::ITSTPCTRD]));
  }
  if (maskTracks[GID::TPCTRD] || maskTracks[GID::TPCTRDTOF]) {
    specs.emplace_back(o2::trd::getTRDTPCTrackReaderSpec(maskTracksMC[GID::TPCTRD], subSpecStrict));
  }
  if (maskTracks[GID::CTP] || maskClusters[GID::CTP]) {
    specs.emplace_back(o2::ctp::getDigitsReaderSpec(maskTracksMC[GID::CTP] || maskClustersMC[GID::CTP]));
  }

  if (maskTracks[GID::PHS] || maskClusters[GID::PHS]) {
    specs.emplace_back(o2::phos::getPHOSCellReaderSpec(maskTracksMC[GID::PHS] || maskClustersMC[GID::PHS]));
  }

  if (maskTracks[GID::CPV] || maskClusters[GID::CPV]) {
    specs.emplace_back(o2::cpv::getCPVClusterReaderSpec(maskTracksMC[GID::CPV] || maskClustersMC[GID::CPV]));
  }

  if (maskTracks[GID::EMC] || maskClusters[GID::EMC]) {
    specs.emplace_back(o2::emcal::getCellReaderSpec(maskTracksMC[GID::EMC] || maskClustersMC[GID::EMC]));
  }

  if (maskClusters[GID::MCH]) {
    specs.emplace_back(o2::mch::getClusterReaderSpec(maskClustersMC[GID::MCH], "mch-cluster-reader", true, false));
  }

  return 0;
}

// attach primary vertex reader
int InputHelper::addInputSpecsPVertex(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs, bool mc)
{
  if (configcontext.options().get<bool>("disable-root-input")) {
    return 0;
  }
  specs.emplace_back(o2::vertexing::getPrimaryVertexReaderSpec(mc));
  return 0;
}

// attach secondary vertex reader
int InputHelper::addInputSpecsSVertex(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs)
{
  if (configcontext.options().get<bool>("disable-root-input")) {
    return 0;
  }
  specs.emplace_back(o2::vertexing::getSecondaryVertexReaderSpec());
  return 0;
}

// attach strangeness tracking reader
int InputHelper::addInputSpecsStrangeTrack(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs, bool mc)
{
  if (configcontext.options().get<bool>("disable-root-input")) {
    return 0;
  }
  specs.emplace_back(o2::strangeness_tracking::getStrangenessTrackingReaderSpec(mc));
  return 0;
}

// attach cosmic tracks reader
int InputHelper::addInputSpecsCosmics(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs, bool mc)
{
  if (configcontext.options().get<bool>("disable-root-input")) {
    return 0;
  }
  specs.emplace_back(o2::globaltracking::getTrackCosmicsReaderSpec(mc));
  return 0;
}

// attach vector of ITS reconstructed IRFrames
int InputHelper::addInputSpecsIRFramesITS(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs)
{
  if (configcontext.options().get<bool>("disable-root-input")) {
    return 0;
  }
  specs.emplace_back(o2::globaltracking::getIRFrameReaderSpec("ITS", 0, "its-irframe-reader", "o2_its_irframe.root"));
  return 0;
}
