// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawToDigitsSpec.h
/// @author Jens Wiechula
/// @since  2021-02-26
/// @brief  Processor spec for running TPC Krypton cluster finder

#ifndef TPC_KryptonClustererSpec_H_
#define TPC_KryptonClustererSpec_H_

#include "Framework/DataProcessorSpec.h"
#include <string_view>

namespace o2
{
namespace tpc
{

/// create a processor spec
/// read simulated TPC clusters from file and publish
o2::framework::DataProcessorSpec getKryptonClustererSpec(const std::string inputSpec, int ilane, std::vector<int> const& sectors);

} // end namespace tpc
} // end namespace o2

#endif // TPC_RAWTODIGITSSPEC_H_
