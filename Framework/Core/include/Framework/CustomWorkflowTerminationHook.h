// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef CUSTOMWORKFLOWTERMINATIONHOOK_H
#define CUSTOMWORKFLOWTERMINATIONHOOK_H

namespace o2
{
namespace framework
{

/// A callback definition for a hook to be invoked when processes terminate
///
/// The parameter is the nullptr if the process is the main driver, for all
/// child processes, the id string is passed. This allows to customize the
/// hook depending on the process.
/// Note that the callback hook is invoked for every process, i.e. main driver
/// and all childs.
///
/// \par Usage:
/// The customize the hook, add a function with the following signature before
/// including heder file runDataProcessing.h:
///
///     void customize(o2::framework::OnWorkflowTerminationHook& hook)
///     {
///       hook = [](const char* idstring){
///         if (idstring == nullptr) {
///           std::cout << "hook" << std::endl;
///         } else {
///           std::cout << "child process " << idstring << " terminating" << std::endl;
///         }
///       };
///     }
///     #include "Framework/runDataProcessing.h"
using OnWorkflowTerminationHook = std::function<void(const char*)>;

} // namespace framework
} // namespace o2

#endif
