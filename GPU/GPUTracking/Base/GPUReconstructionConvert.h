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

/// \file GPUReconstructionConvert.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONCONVERT_H
#define GPURECONSTRUCTIONCONVERT_H

#include <memory>
#include <functional>
#include <vector>
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
  template <class S>
  static void RunZSEncoder(const S& in, std::unique_ptr<unsigned long long int[]>* outBuffer, unsigned int* outSizes, o2::raw::RawFileWriter* raw, const o2::InteractionRecord* ir, const GPUParam& param, int version, bool verify, float threshold = 0.f, bool padding = false, std::function<void(std::vector<o2::tpc::Digit>&)> digitsFilter = nullptr);
  static void RunZSEncoderCreateMeta(const unsigned long long int* buffer, const unsigned int* sizes, void** ptrs, GPUTrackingInOutZS* out);
  static void RunZSFilter(std::unique_ptr<o2::tpc::Digit[]>* buffers, const o2::tpc::Digit* const* ptrs, size_t* nsb, const size_t* ns, const GPUParam& param, bool zs12bit, float threshold);
  static int GetMaxTimeBin(const o2::tpc::ClusterNativeAccess& native);
  static int GetMaxTimeBin(const GPUTrackingInOutDigits& digits);
  static int GetMaxTimeBin(const GPUTrackingInOutZS& zspages);
  static std::function<void(std::vector<o2::tpc::Digit>&, const void*, unsigned int, unsigned int)> GetDecoder(int version, const GPUParam& param);
};

class GPUReconstructionZSDecoder
{
 public:
  void DecodePage(std::vector<o2::tpc::Digit>& outputBuffer, const void* page, unsigned int tfFirstOrbit, const GPUParam& param, unsigned int triggerBC = 0);

 private:
  std::vector<std::function<void(std::vector<o2::tpc::Digit>&, const void*, unsigned int, unsigned int)>> mDecoders;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
