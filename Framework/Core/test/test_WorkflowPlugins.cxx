// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Loads the workflow plugins of test_WorkflowPluginA and test_WorkflowPluginB,
// given with --workflow-plugin, in place of its own.
#include "Framework/runDataProcessing.h"
#include "Framework/RuntimeError.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  throw runtime_error("The embedded workflow must not be used when --workflow-plugin is given.");
}
