// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    raw-to-digits-workflow.cxx
/// \author  Andrea Ferrero
///
/// \brief This is an executable that runs the decoding via DPL.
///
/// This is an executable that takes a raw buffer from the Data Processing Layer, runs the decoding and sends the digits via the Data Processing Layer.
///

#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/runDataProcessing.h"
#include "MCHWorkflow/DataDecoderSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  DataProcessorSpec producer = o2::mch::raw::getDecodingSpec();
  specs.push_back(producer);

  return specs;
}
