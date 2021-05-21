// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_O2CONTROLHELPERS_H
#define FRAMEWORK_O2CONTROLHELPERS_H

#include "Framework/DeviceSpec.h"
#include "Framework/DeviceExecution.h"
#include "Framework/CommandInfo.h"
#include <vector>
#include <iosfwd>

namespace o2
{
namespace framework
{

/// \brief Dumps the AliECS compatible workflow and task templates for a DPL workflow.
///
/// Dumps the AliECS compatible workflow (WFT) and task templates (TT) for a DPL workflow.
/// The current procedure to obtain working templates:
/// - Build the project(s)
/// - Enter the environment and go to ControlWorkflows local repository.
/// - Run the DPL workflow(s) with the argument `--o2-control <workflow-name>`.
///   The WFT will be created in the "workflows" directory and, analogously, TTs will be put in "tasks".
///   It can be included by a mother workflow at the deployment time.
/// - Replace arguments with templates if needed.
/// - Commit, push, test, merge to master.
void dumpDeviceSpec2O2Control(std::string workflowName,
                              std::vector<DeviceSpec> const& specs,
                              std::vector<DeviceExecution> const& executions,
                              CommandInfo const& commandInfo);

/// \brief Dumps only the workflow file
void dumpWorkflow(std::ostream& dumpOut,
                  const std::vector<DeviceSpec>& specs,
                  const std::vector<DeviceExecution>& executions,
                  const CommandInfo& commandInfo,
                  std::string workflowName,
                  std::string indLevel);

/// \brief Dumps only one task
void dumpTask(std::ostream& dumpOut,
              const DeviceSpec& spec,
              const DeviceExecution& execution,
              std::string taskName,
              std::string indLevel);

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_O2CONTROLHELPERS_H
