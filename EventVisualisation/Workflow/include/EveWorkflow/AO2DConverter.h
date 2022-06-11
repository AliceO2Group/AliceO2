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

///
/// \file   AO2DConverter.h
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_AO2DCONVERTER_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_AO2DCONVERTER_H

#include "EveWorkflow/EveWorkflowHelper.h"
#include "EveWorkflow/DetectorData.h"
#include "Framework/AnalysisTask.h"
#include <memory>

using GID = o2::dataformats::GlobalTrackID;

namespace o2::trd
{
class GeometryFlat;
}

namespace o2::itsmft
{
class TopologyDictionary;
}

namespace o2::event_visualisation
{
class TPCFastTransform;

struct AO2DConverter {
  o2::framework::Configurable<std::string> jsonPath{"jsons-folder", "./json", "name of the folder to store json files"};

  static constexpr float mWorkflowVersion = 1.00;
  o2::header::DataHeader::RunNumberType mRunNumber = 0;
  o2::header::DataHeader::TFCounterType mTfCounter = 0;
  o2::header::DataHeader::TForbitType mTfOrbit = 0;
  o2::framework::DataProcessingHeader::CreationTime mCreationTime;

  void init(o2::framework::InitContext& ic);
  void process(o2::aod::Collision const& collision, EveWorkflowHelper::AODBarrelTracks const& barrelTracks, EveWorkflowHelper::AODForwardTracks const& fwdTracks, EveWorkflowHelper::AODMFTTracks const& mftTracks);

  DetectorData mData;
  std::shared_ptr<EveWorkflowHelper> mHelper;
};

} // namespace o2::event_visualisation

#endif
