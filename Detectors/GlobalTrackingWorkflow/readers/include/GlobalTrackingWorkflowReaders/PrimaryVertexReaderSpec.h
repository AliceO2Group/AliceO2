// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See http://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   PrimaryVertexReaderSpec.h

#ifndef O2_PRIMARY_VERTEXREADER
#define O2_PRIMARY_VERTEXREADER

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace vertexing
{
/// create a processor spec
/// read primary vertex data from a root file
o2::framework::DataProcessorSpec getPrimaryVertexReaderSpec(bool useMC);

} // namespace vertexing
} // namespace o2

#endif
