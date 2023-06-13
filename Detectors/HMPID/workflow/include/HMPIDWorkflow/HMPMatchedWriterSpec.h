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

/// @file   HMPMatchedWriterSpec.h

#ifndef HMPWORKFLOW_HMPMATCHEDWRITER_H_
#define HMPWORKFLOW_HMPMATCHEDWRITER_H_

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace hmpid
{

/// create a processor spec
/// write HMP matching info in a root file
o2::framework::DataProcessorSpec getHMPMatchedWriterSpec(bool useMC, const char* outdef = "o2match_hmpid.root"); // int mode = 0, bool strict = false);

} // namespace hmpid
} // namespace o2

#endif /* HMPWORKFLOW_HMPMATCHEDWRITER_H_ */
