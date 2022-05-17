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

/// \file TimeClusterFinderSpec.h
/// \brief Definition of a data processor to run the time clusterizer
///
/// \author Andrea Ferrero, CEA

#ifndef O2_MCH_TIMECLUSTERFINDERSPEC_H_
#define O2_MCH_TIMECLUSTERFINDERSPEC_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mch
{

o2::framework::DataProcessorSpec
  getTimeClusterFinderSpec(const char* specName = "mch-time-cluster-finder",
                           std::string_view inputDigitDataDescription = "F-DIGITS",
                           std::string_view inputDigitRofDataDescription = "F-DIGITROFS",
                           std::string_view outputDigitRofDataDescription = "TC-F-DIGITROFS",
                           std::string_view inputIRFrameDataDescription = "ITS/IRFRAMES");

} // end namespace mch
} // end namespace o2

#endif // O2_MCH_TIMECLUSTERFINDERSPEC_H_
