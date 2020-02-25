// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionConvert.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONCONVERT_H
#define GPURECONSTRUCTIONCONVERT_H

#include <memory>
#include "GPUDef.h"

namespace o2
{
namespace tpc
{
struct ClusterNative;
struct ClusterNativeAccess;
} // namespace tpc
} // namespace o2

class AliHLTTPCRawCluster;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUParam;
class GPUTPCClusterData;
class TPCFastTransform;
struct GPUTrackingInOutDigits;
struct GPUTrackingInOutZS;
namespace deprecated
{
struct PackedDigit;
}

class GPUReconstructionConvert
{
 public:
  constexpr static unsigned int NSLICES = GPUCA_NSLICES;
  static void ConvertNativeToClusterData(o2::tpc::ClusterNativeAccess* native, std::unique_ptr<GPUTPCClusterData[]>* clusters, unsigned int* nClusters, const TPCFastTransform* transform, int continuousMaxTimeBin = 0);
  static void ConvertRun2RawToNative(o2::tpc::ClusterNativeAccess& native, std::unique_ptr<o2::tpc::ClusterNative[]>& nativeBuffer, const AliHLTTPCRawCluster** rawClusters, unsigned int* nRawClusters);
  static void RunZSEncoder(const GPUTrackingInOutDigits* in, GPUTrackingInOutZS*& out, const GPUParam& param, bool zs12bit);
  static void RunZSFilter(std::unique_ptr<deprecated::PackedDigit[]>* buffers, const deprecated::PackedDigit* const* ptrs, size_t* nsb, const size_t* ns, const GPUParam& param, bool zs12bit);
  static int GetMaxTimeBin(const o2::tpc::ClusterNativeAccess& native);
  static int GetMaxTimeBin(const GPUTrackingInOutDigits& digits);
  static int GetMaxTimeBin(const GPUTrackingInOutZS& zspages);

 private:
  static void ZSstreamOut(unsigned short* bufIn, unsigned int& lenIn, unsigned char* bufOut, unsigned int& lenOut, unsigned int nBits);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
