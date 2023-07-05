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

#ifndef O2_FT0_RECOWORKFLOW_H
#define O2_FT0_RECOWORKFLOW_H

/// @file   RecoWorkflow.h

#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace ft0
{
framework::WorkflowSpec getRecoWorkflow(bool useMC, std::string ccdbpath, bool useTimeOffsetCalib, bool disableRootInp, bool disableRootOut);
} // namespace ft0
} // namespace o2
#endif
