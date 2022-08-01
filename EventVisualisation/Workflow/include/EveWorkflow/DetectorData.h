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
/// \file   DetectorData.h
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_DETECTORDATA_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_DETECTORDATA_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "EveWorkflow/EveConfiguration.h"
#include "EveWorkflow/EveWorkflowHelper.h"
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

class DetectorData
{
 public:
  void init();
  void setITSDict(const o2::itsmft::TopologyDictionary* d);
  void setMFTDict(const o2::itsmft::TopologyDictionary* d);

  const o2::itsmft::TopologyDictionary* mITSDict = nullptr;
  const o2::itsmft::TopologyDictionary* mMFTDict = nullptr;
  EveConfiguration mConfig{};
  std::unique_ptr<o2::trd::GeometryFlat> mTrdGeo;
};

} // namespace o2::event_visualisation

#endif
