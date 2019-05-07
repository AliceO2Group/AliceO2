// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCClusterDecompressor.h
/// \author David Rohr

#ifndef TPCCLUSTERDECOMPRESSOR_H
#define TPCCLUSTERDECOMPRESSOR_H

#include "GPUTPCCompression.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
using CompressedClusters = o2::TPC::CompressedClusters;
struct ClusterNativeAccessExt;

class TPCClusterDecompressor
{
 public:
 protected:
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
