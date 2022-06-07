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
#include "DetectorsBase/Propagator.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "CommonUtils/NameConf.h"
#include "TRDBase/GeometryFlat.h"
#include "TRDBase/Geometry.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/ROFRecord.h"

using namespace o2::event_visualisation;
using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::globaltracking;
using namespace o2::tpc;
using namespace o2::trd;

#include "Framework/runDataProcessing.h"

void AO2DConverter::init(o2::framework::InitContext& ic)
{
  LOG(info) << "------------------------    AO2DConverter::init version " << mWorkflowVersion << "    ------------------------------------";

  mData.init();
  mHelper = std::make_shared<EveWorkflowHelper>();
}

void AO2DConverter::process(o2::aod::Collisions const& collisions, EveWorkflowHelper::AODBarrelTracks const& barrelTracks, EveWorkflowHelper::AODForwardTracks const& fwdTracks, EveWorkflowHelper::AODMFTTracks const& mftTracks)
{
  for (auto const& c : collisions) {
    auto const barrelTracksForCol = barrelTracks.sliceBy(aod::track::collisionId, c.globalIndex());
    auto const mftTracksForCol = mftTracks.sliceBy(aod::fwdtrack::collisionId, c.globalIndex());

    for (auto const& track : barrelTracksForCol) {
      mHelper->drawAODBarrel(track, c.collisionTime());
    }

    for (auto const& track : mftTracksForCol) {
      mHelper->drawAODMFT(track, c.collisionTime());
    }

    mHelper->save(jsonPath, collisions.size(), GlobalTrackID::MASK_ALL, GlobalTrackID::MASK_NONE, mRunNumber, mCreationTime);
    mHelper->clear();
  }
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<AO2DConverter>(cfgc, TaskName{"o2-aodconverter"}),
  };
}
