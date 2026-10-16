// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfig_print.inc
/// \author David Rohr

// Generates qConfigPrint for all config structs in the namespace it is included in, after the instances of the top-level configs (e.g. configStandalone).
// Requires <functional>, <iostream>, <string>, <vector>, "qconfig_helpers.h" to be included before (outside of the namespace).

#ifndef QCONFIG_PRINT_INC
#define QCONFIG_PRINT_INC

std::vector<std::function<void()>> qprint_global;
#define QCONFIG_PRINT
#include "qconfig_internal.h"
#undef QCONFIG_PRINT

#endif // QCONFIG_PRINT_INC
