// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FIT_FT0WORKFLOW_H
#define O2_FIT_FT0WORKFLOW_H

/// @file   FT0Workflow.h

#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace ft0
{
framework::WorkflowSpec getFT0Workflow(bool isExtendedMode, bool useProcess,
                                       bool dumpProcessor, bool dumpReader,
                                       bool disableRootOut);
} // namespace ft0
} // namespace o2
#endif
