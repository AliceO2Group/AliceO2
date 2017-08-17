// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_WORKFLOW_SPEC_H
#define FRAMEWORK_WORKFLOW_SPEC_H

#include "Framework/DataProcessorSpec.h"

#include <vector>

namespace o2 {
namespace framework {
using WorkflowSpec = std::vector<DataProcessorSpec>;
}
}

#endif // FRAMEWORK_WORKFLOW_SPEC_H
