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
/// \file   EveWorkflowHelper.h
/// \author julian.myrcha@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "EveWorkflow/EveConfiguration.h"

namespace o2::event_visualisation
{
class EveWorkflowHelper
{
  using GID = o2::dataformats::GlobalTrackID;

 public:
  struct tmpDataContainer {
    std::vector<o2::BaseCluster<float>> ITSClustersArray;
    std::vector<int> tpcLinkITS, tpcLinkTRD, tpcLinkTOF;
    std::vector<const o2::track::TrackParCov*> globalTracks;
    std::vector<float> globalTrackTimes;
  };
  static std::shared_ptr<const tmpDataContainer> compute(const o2::globaltracking::RecoContainer& recoCont, const CalibObjectsConst* calib = nullptr, GID::mask_t maskCl = GID::MASK_ALL, GID::mask_t maskTrk = GID::MASK_ALL, GID::mask_t maskMatch = GID::MASK_ALL);
};
} // namespace o2::event_visualisation

#endif //ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H
