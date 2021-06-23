// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ZDC_RECOWORKFLOW_H
#define O2_ZDC_RECOWORKFLOW_H

/// @file   RecoWorkflow.h

#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace zdc
{
framework::WorkflowSpec getRecoWorkflow(const bool useMC, const bool disableRootInp, const bool disableRootOut, const int verbosity, const bool enableDebugOut);
} // namespace zdc
} // namespace o2
#endif
