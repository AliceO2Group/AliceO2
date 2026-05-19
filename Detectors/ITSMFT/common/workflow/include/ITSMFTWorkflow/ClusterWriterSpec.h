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

/// @file   ClusterWriterSpec.h

#ifndef O2_ITSMFT_CLUSTERWRITER
#define O2_ITSMFT_CLUSTERWRITER

#include "Framework/DataProcessorSpec.h"

namespace o2::itsmft
{

template <int N>
framework::DataProcessorSpec getClusterWriterSpec(bool useMC, bool doStag, bool clusterROFOnly = false);
framework::DataProcessorSpec getITSClusterWriterSpec(bool useMC, bool doStag, bool clusterROFOnly = false);
framework::DataProcessorSpec getMFTClusterWriterSpec(bool useMC, bool doStag, bool clusterROFOnly = false);

} // namespace o2::itsmft

#endif /* O2_ITS_CLUSTERWRITER */
