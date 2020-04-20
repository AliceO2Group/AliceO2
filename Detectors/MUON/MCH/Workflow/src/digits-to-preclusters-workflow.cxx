// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file digits-to-preclusters-workflow.cxx
/// \brief This is an executable that runs the preclusterization via DPL.
///
/// This is an executable that takes digits from the Data Processing Layer, runs the preclusterization and sends the preclusters via the Data Processing Layer.
///
/// \author Philippe Pillot, Subatech
/// \author Andrea Ferrero, CEA

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/runDataProcessing.h"
#include "MCHWorkflow/PreClusterFinderSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  DataProcessorSpec producer = o2::mch::getPreClusterFinderSpec();
  specs.push_back(producer);

  return specs;
}
