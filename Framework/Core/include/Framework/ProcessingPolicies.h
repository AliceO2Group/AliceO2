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
#ifndef O2_FRAMEWORK_PROCESSING_POLICIES_H_
#define O2_FRAMEWORK_PROCESSING_POLICIES_H_

namespace o2::framework
{

/// These are the possible actions we can do
/// when a workflow is deemed complete (e.g. when we are done
/// reading from file) or when a data processor has
/// an error.
enum struct TerminationPolicy { QUIT,
                                WAIT,
                                RESTART };

/// When to enable the early forwarding optimization:
enum struct EarlyForwardPolicy { NEVER, //< never
                                 NORAW, //< disabled for raw, aod, and optional inputs
                                 ALWAYS //< disabled for aod
};

struct ProcessingPolicies {
  enum TerminationPolicy termination;
  enum TerminationPolicy error;
  enum EarlyForwardPolicy earlyForward;
};

/// The mode in which the driver is running. Should be MASTER when running locally,
/// EMBEDDED when running under a control system.
enum struct DriverMode {
  /// Default mode. The master is responsible for spawning and controlling the devices.
  STANDALONE,
  /// The master is also running under a control system, so it does not control the devices, however, it does
  /// receive metrics and information from them and displays it via the RemoteGUI.
  EMBEDDED
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_PROCESSING_POLICIES_H_
