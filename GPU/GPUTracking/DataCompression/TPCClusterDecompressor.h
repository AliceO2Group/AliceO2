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
#include <vector>

namespace o2
{
namespace tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
} // namespace tpc
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
using CompressedClusters = o2::tpc::CompressedClusters;
struct GPUParam;

class TPCClusterDecompressor
{
 public:
  static constexpr unsigned int NSLICES = GPUCA_NSLICES;
  int decompress(const CompressedClusters* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::vector<o2::tpc::ClusterNative>& clusterBuffer, const GPUParam& param);

 protected:
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
