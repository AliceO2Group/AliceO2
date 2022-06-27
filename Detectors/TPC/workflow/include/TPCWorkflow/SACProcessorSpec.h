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

/// @file SACProcessorSpec.h
/// @brief TPC Sampled Analogue Current processing
/// @author Jens Wiechula

#ifndef TPC_SACProcessorSpec_H_
#define TPC_SACProcessorSpec_H_

#include "Framework/DataProcessorSpec.h"

namespace o2::tpc
{

/// decode SAC raw data
o2::framework::DataProcessorSpec getSACProcessorSpec();

} // end namespace o2::tpc

#endif // TPC_SACProcessorSpec_H_
