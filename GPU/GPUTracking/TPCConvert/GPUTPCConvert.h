// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCConvert.h
/// \author David Rohr

#ifndef GPUTPCCONVERT_H
#define GPUTPCCONVERT_H

#include "GPUDef.h"
#include "GPUProcessor.h"

namespace o2
{
namespace TPC
{
struct ClusterNativeAccessFullTPC;
struct ClusterNative;
} // namespace TPC
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterData;
struct ClusterNativeAccessExt;
class TPCFastTransform;

class GPUTPCConvert : public GPUProcessor
{
  friend class GPUTPCConvertKernel;
  friend class GPUChainTracking;

 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData();

  void* SetPointersInput(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersMemory(void* mem);

  void set(ClusterNativeAccessExt* clusterNative, const TPCFastTransform* transform);
#endif

  constexpr static unsigned int NSLICES = GPUCA_NSLICES;

  struct Memory {
    GPUTPCClusterData* clusters[NSLICES];
  };

 protected:
  ClusterNativeAccessExt* mClustersNative = nullptr;
  ClusterNativeAccessExt* mClustersNativeBuffer;

  const TPCFastTransform* mTransform = nullptr;
  Memory* mMemory = nullptr;
  o2::TPC::ClusterNative* mInputClusters;
  GPUTPCClusterData* mClusters = nullptr;
  unsigned int mNClustersTotal = 0;

  short mMemoryResInput = -1;
  short mMemoryResOutput = -1;
  short mMemoryResMemory = -1;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
