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

#ifndef O2_TRD_TRAPCONFIGEVENTSPEC_H
#define O2_TRD_TRAPCONFIGEVENTSPEC_H

/// \file  TrapConfigEventSpec.h
/// \brief Steers TrapConfigEvent comparison and parsing.
/// \author Sean Murray

// input TRD config events, current config from ccdb
// output new ccdb config if required

#include <vector>
#include <array>
#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "TRDReconstruction/TrapConfigEventParser.h"

namespace o2
{
namespace trd
{

/// create a processor spec
framework::DataProcessorSpec getTrapConfigEventSpec();

} // end namespace trd
} // end namespace o2

#endif // O2_TRD_TRAPCONFIGSPEC
