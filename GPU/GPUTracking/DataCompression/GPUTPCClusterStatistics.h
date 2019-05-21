// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterStatistics.h
/// \author David Rohr

#ifndef GPUTPCCLUSTERSTATISTICS_H
#define GPUTPCCLUSTERSTATISTICS_H

#include "GPUTPCCompression.h"
#include "TPCClusterDecompressor.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct ClusterNativeAccessExt;

class GPUTPCClusterStatistics
{
 public:
#ifndef HAVE_O2HEADERS
  void RunStatistics(const ClusterNativeAccessExt* clustersNative, const o2::TPC::CompressedClusters* clustersCompressed, const GPUParam& param){};
#else
  static constexpr unsigned int NSLICES = GPUCA_NSLICES;
  void RunStatistics(const ClusterNativeAccessExt* clustersNative, const o2::TPC::CompressedClusters* clustersCompressed, const GPUParam& param);

 protected:
  TPCClusterDecompressor mDecoder;
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
