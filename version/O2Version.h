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

#pragma once

#include <string>

namespace o2
{
/// get full version information (official O2 release and git commit)
std::string fullVersion();

/// get O2 git commit used to build this
std::string gitRevision();

/// get information about build platform (for example OS and alidist release when used)
std::string getBuildInfo();
} // namespace o2
