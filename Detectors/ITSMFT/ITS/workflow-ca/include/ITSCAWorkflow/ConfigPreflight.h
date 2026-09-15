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
/// \file ConfigPreflight.h
/// \brief Driver-level configuration and vertex-constraint preflight for the
///        ITS common-CA tracker workflow.
///
/// Resolve driver options before constructing any DPL device.

#ifndef ALICEO2_ITS_CA_WORKFLOW_CONFIGPREFLIGHT_H_
#define ALICEO2_ITS_CA_WORKFLOW_CONFIGPREFLIGHT_H_

#include <string>

#include "ITSMFTTracking/Configuration.h"

namespace o2::framework
{
class ConfigContext;
}

namespace o2::its::ca
{

/// Rejects a raw --configKeyValues string carrying an ITSCATrackerParam.*
/// override before applying the accepted string to ConfigurableParam.
void applyConfigKeyValuesOrFatal(const std::string& configKeyValues);

/// Fatals unless mode is Sync or Async, naming the rejected mode explicitly,
/// before device construction.
void requireSupportedTrackingModeOrFatal(o2::itsmft::TrackingMode::Type mode);

enum class VertexSource { Diamond,
                          Truth };
struct WorkflowOptions {
  bool useMC = true;
  bool useFullGeometry = false;
  bool writeRootOutput = true;
  o2::itsmft::TrackingMode::Type mode = o2::itsmft::TrackingMode::Sync;
  int nThreads = 1;
  VertexSource vertexSource = VertexSource::Diamond;
  std::string truthContext = "collisioncontext.root";
};

// An empty explicit source requires exactly one legacy alias. No physics
// constraint is enabled by default, and MC output labels are independent.
VertexSource resolveVertexSource(const std::string& explicitSource, bool useDiamond, bool useTruth);
WorkflowOptions readWorkflowOptions(const o2::framework::ConfigContext&);

} // namespace o2::its::ca

#endif // ALICEO2_ITS_CA_WORKFLOW_CONFIGPREFLIGHT_H_
