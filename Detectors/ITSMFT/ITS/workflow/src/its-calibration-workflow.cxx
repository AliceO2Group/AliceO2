// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITSWorkflow/CalibrationWorkflow.h"
#include "ITSWorkflow/CalibratorSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Configuration.h"

using namespace o2::framework;

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
    LOG(info) << "Initializing O2 ITS Calibration:-))))))";

    return std::move(o2::its::calibration_workflow::getWorkflow());
}
