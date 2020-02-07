// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_RAWFILE_READER_WORKFLOW_H
#define O2_RAWFILE_READER_WORKFLOW_H

/// @file   RawFileReaderWorkflow.h

#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace raw
{

framework::WorkflowSpec getRawFileReaderWorkflow(std::string inifile, bool tfAsMessage = false, int loop = 0);

} // namespace raw
} // namespace o2
#endif
