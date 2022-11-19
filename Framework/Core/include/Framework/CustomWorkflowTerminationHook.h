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
#ifndef CUSTOMWORKFLOWTERMINATIONHOOK_H
#define CUSTOMWORKFLOWTERMINATIONHOOK_H

#include <functional>

namespace o2::framework
{

/// A callback definition for a hook to be invoked when processes terminate
///
/// The parameter is the nullptr if the process is the main driver, for all
/// child processes, the id string is passed. This allows to customize the
/// hook depending on the process.
/// Note that the callback hook is invoked for every process, i.e. main driver
/// and all children.
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

} // namespace o2::framework

#endif
