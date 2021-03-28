// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/Task.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/Digit.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"

using namespace o2::framework;

namespace o2::ft0
{

class FT0ViewObjectGenerator final : public o2::framework::Task
{
  //work in progress
  //are there any ccdb output parsers?
  //my plan is to use ccdb api with "list" method, than parse timestamps with range given by the user
  //and convert retrieved object into TGraph

};

}
