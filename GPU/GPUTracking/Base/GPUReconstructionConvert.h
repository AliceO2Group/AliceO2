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
struct InteractionRecord;
namespace tpc
{
struct ClusterNative;
struct ClusterNativeAccess;
class Digit;
} // namespace tpc
namespace raw
{
class RawFileWriter;
} // namespace raw
} // namespace o2

struct AliHLTTPCRawCluster;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUParam;
struct GPUTPCClusterData;
class TPCFastTransform;
struct GPUTrackingInOutDigits;
struct GPUTrackingInOutZS;

class GPUReconstructionConvert
{
 public:
  constexpr static unsigned int NSLICES = GPUCA_NSLICES;
  static void ConvertNativeToClusterData(o2::tpc::ClusterNativeAccess* native, std::unique_ptr<GPUTPCClusterData[]>* clusters, unsigned int* nClusters, const TPCFastTransform* transform, int continuousMaxTimeBin = 0);
  static void ConvertRun2RawToNative(o2::tpc::ClusterNativeAccess& native, std::unique_ptr<o2::tpc::ClusterNative[]>& nativeBuffer, const AliHLTTPCRawCluster** rawClusters, unsigned int* nRawClusters);
  template <class T, class S>
  static void RunZSEncoder(const S& in, std::unique_ptr<unsigned long long int[]>* outBuffer, unsigned int* outSizes, o2::raw::RawFileWriter* raw, const o2::InteractionRecord* ir, const GPUParam& param, bool zs12bit, bool verify, float threshold = 0.f, bool padding = true);
  static void RunZSEncoderCreateMeta(const unsigned long long int* buffer, const unsigned int* sizes, void** ptrs, GPUTrackingInOutZS* out);
  static void RunZSFilter(std::unique_ptr<o2::tpc::Digit[]>* buffers, const o2::tpc::Digit* const* ptrs, size_t* nsb, const size_t* ns, const GPUParam& param, bool zs12bit, float threshold);
  static int GetMaxTimeBin(const o2::tpc::ClusterNativeAccess& native);
  static int GetMaxTimeBin(const GPUTrackingInOutDigits& digits);
  static int GetMaxTimeBin(const GPUTrackingInOutZS& zspages);

 private:
  static void ZSstreamOut(unsigned short* bufIn, unsigned int& lenIn, unsigned char* bufOut, unsigned int& lenOut, unsigned int nBits);
  static void ZSfillEmpty(void* ptr, int shift, unsigned int feeId, int orbit);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
