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

/// \file AO2DConverter.cxx
/// \author Piotr Nowakowski

#include "EveWorkflow/AO2DConverter.h"
#include "EveWorkflow/EveWorkflowHelper.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "TRDBase/GeometryFlat.h"
#include "TOFBase/Geo.h"
#include "TPCFastTransform.h"
#include "TRDBase/Geometry.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/ClusterBlock.h"

using namespace o2::event_visualisation;
using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::globaltracking;
using namespace o2::tpc;
using namespace o2::trd;

#include "Framework/runDataProcessing.h"

void AO2DConverter::init(o2::framework::InitContext& ic)
{
  LOG(INFO) << "------------------------    AO2DConverter::init version " << mWorkflowVersion << "    ------------------------------------";
  const auto grp = o2::parameters::GRPObject::loadFrom();
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  mConfig.reset(new EveConfiguration);
  mConfig->configGRP.solenoidBz = 5.00668f * grp->getL3Current() / 30000.;
  mConfig->configGRP.continuousMaxTimeBin = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC) ? -1 : 0; // Number of timebins in timeframe if continuous, 0 otherwise
  mConfig->ReadConfigurableParam();

  auto gm = o2::trd::Geometry::instance();
  gm->createPadPlaneArray();
  gm->createClusterMatrixArray();
  mTrdGeo.reset(new o2::trd::GeometryFlat(*gm));
  mConfig->configCalib.trdGeometry = mTrdGeo.get();

  std::string dictFileITS = o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance().dictFilePath;
  dictFileITS = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, dictFileITS);
  if (o2::utils::Str::pathExists(dictFileITS)) {
    mITSDict.readFromFile(dictFileITS);
    LOG(INFO) << "Running with provided ITS clusters dictionary: " << dictFileITS;
  } else {
    LOG(INFO) << "Dictionary " << dictFileITS << " is absent, ITS expects cluster patterns for all clusters";
  }
  mConfig->configCalib.itsPatternDict = &mITSDict;

  std::string dictFileMFT = o2::itsmft::ClustererParam<o2::detectors::DetID::MFT>::Instance().dictFilePath;
  dictFileMFT = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::MFT, dictFileMFT);
  if (o2::utils::Str::pathExists(dictFileMFT)) {
    mMFTDict.readFromFile(dictFileMFT);
    LOG(INFO) << "Running with provided MFT clusters dictionary: " << dictFileMFT;
  } else {
    LOG(INFO) << "Dictionary " << dictFileMFT << " is absent, MFT expects cluster patterns for all clusters";
  }
  mConfig->configCalib.mftPatternDict = &mMFTDict;

  o2::tof::Geo::Init();

  o2::its::GeometryTGeo::Instance()->fillMatrixCache(
    o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot,
                             o2::math_utils::TransformType::T2G,
                             o2::math_utils::TransformType::L2G,
                             o2::math_utils::TransformType::T2L));
}

void AO2DConverter::process(EveWorkflowHelper::AODFullTracks const& tracks)
{
  std::unordered_map<std::size_t, std::vector<EveWorkflowHelper::AODFullTrack>> colTracks;

  for (auto& track : tracks) {
    // operator[] automatically adds a new entry to the map if not already present
    colTracks[track.collisionId()].push_back(track);
  }

  for (auto const& p : colTracks) {
    EveWorkflowHelper helper;
    for(auto const &track: p.second) {
      helper.drawAOD(track);
    }
    helper.save(jsonPath, colTracks.size(), {}, {}, mWorkflowVersion);
  }
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  LOG(INFO) << "------------------------    defineDataProcessing " << AO2DConverter::mWorkflowVersion << "    ------------------------------------";

  return WorkflowSpec{
    adaptAnalysisTask<AO2DConverter>(cfgc, TaskName{"o2-aodconverter"})
  };
}
