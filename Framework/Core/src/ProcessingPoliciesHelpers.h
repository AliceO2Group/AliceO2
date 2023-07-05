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
#ifndef O2_FRAMEWORK_PROCESSINGPOLICIESHELPERS_H_
#define O2_FRAMEWORK_PROCESSINGPOLICIESHELPERS_H_

#include "Framework/ProcessingPolicies.h"
#include "Framework/LogParsingHelpers.h"
#include <ostream>
#include <istream>

namespace o2::framework
{

std::istream& operator>>(std::istream& in, enum TerminationPolicy& policy);
std::ostream& operator<<(std::ostream& out, const enum TerminationPolicy& policy);
std::istream& operator>>(std::istream& in, enum EarlyForwardPolicy& policy);
std::ostream& operator<<(std::ostream& out, const enum EarlyForwardPolicy& policy);
std::istream& operator>>(std::istream& in, enum LogParsingHelpers::LogLevel& level);
std::ostream& operator<<(std::ostream& out, const enum LogParsingHelpers::LogLevel& level);
std::istream& operator>>(std::istream& in, enum DriverMode& level);
std::ostream& operator<<(std::ostream& out, const enum DriverMode& level);
} // namespace o2::framework

#endif // O2_FRAMEWORK_PROCESSINGPOLICIESHELPERS_H_
