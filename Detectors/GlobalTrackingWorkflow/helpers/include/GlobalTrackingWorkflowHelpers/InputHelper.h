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

#ifndef O2_GLOBALTRACKING_INPUTHELPER_H
#define O2_GLOBALTRACKING_INPUTHELPER_H

#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace globaltracking
{

class InputHelper
{
 public:
  using GID = o2::dataformats::GlobalTrackID;
  // If "--disable-mc" is passed as option, useMC is overwritten to false.
  // If useMC is false, maskClustersMC and maskTracksMC are overwritten to NONE.
  // The masks define what data to load in a quite generic way, masks with MC suffix are for the corresponding MC labels.
  // For matched tracks, maskMatches refers only to the matching information, while the corresponding maskTracks can still be set to load also the refit matched tracks
  // If subSpecStrict==true, then those inputs which are supposed to be produced in the "strict" mode of the extended matching workflow will by pushed with special
  // subspec corresponding to this mode (see MatchingType.h)
  static int addInputSpecs(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs,
                           GID::mask_t maskClusters, GID::mask_t maskMatches, GID::mask_t maskTracks,
                           bool useMC = true, GID::mask_t maskClustersMC = GID::getSourcesMask(GID::ALL), GID::mask_t maskTracksMC = GID::getSourcesMask(GID::ALL),
                           bool subSpecStrict = false);
  static int addInputSpecsPVertex(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs, bool mc);
  static int addInputSpecsSVertex(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs);
  static int addInputSpecsStrangeTrack(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs, bool mc);
  static int addInputSpecsCosmics(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs, bool mc);
  static int addInputSpecsIRFramesITS(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& specs);
};

} // namespace globaltracking
} // namespace o2

#endif
