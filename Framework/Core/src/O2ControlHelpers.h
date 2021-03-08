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
/// - Copy the WFT contents into existing "mother" workflow, e.g. readout-dataflow.yaml.
///   Later, AliECS will allow to include subworkflows in WFTs.
/// - Create the standard DPL dump (`--dump > <workflow-name>`).
///   Make sure it is copied into the path specified with dpl_config var in WTF.
/// - Commit, push, test, merge to master.
/// With the future developments we aim to minimise the amount of effort to create WFTs and TTs,
/// then finally reach JIT workflow translation.
/// To avoid creating and using the standard DPL dump, one can paste the DPL command
/// in the place of "cat {{ dpl_config }}"

void dumpDeviceSpec2O2Control(std::string workflowName,
                              std::vector<DeviceSpec> const& specs,
                              std::vector<DeviceExecution> const& executions);

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_O2CONTROLHELPERS_H
