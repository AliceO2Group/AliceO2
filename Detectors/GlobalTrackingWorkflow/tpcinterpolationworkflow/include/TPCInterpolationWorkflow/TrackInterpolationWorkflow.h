// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_INTERPOLATION_WORKFLOW_H
#define O2_TPC_INTERPOLATION_WORKFLOW_H

/// @file   TrackInterpolationWorkflow.h

#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace tpc
{

framework::WorkflowSpec getTPCInterpolationWorkflow(bool useMC);

} // namespace tpc
} // namespace o2

#endif
