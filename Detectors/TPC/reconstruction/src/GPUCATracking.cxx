// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCATracking.cxx
/// \author David Rohr

#include "TPCReconstruction/GPUCATracking.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsTPC/Digit.h"

#include "GPUO2Interface.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCBase/Sector.h"

#include <atomic>
#include <optional>
#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2;
using namespace o2::dataformats;

GPUCATracking::GPUCATracking() = default;
GPUCATracking::~GPUCATracking() { deinitialize(); }

int GPUCATracking::initialize(const GPUO2InterfaceConfiguration& config)
{
  mTrackingCAO2Interface.reset(new GPUO2Interface);
  int retVal = mTrackingCAO2Interface->Initialize(config);
  if (retVal) {
    mTrackingCAO2Interface.reset();
  }
  return (retVal);
}

void GPUCATracking::deinitialize()
{
  mTrackingCAO2Interface.reset();
}

int GPUCATracking::runTracking(GPUO2InterfaceIOPtrs* data, GPUInterfaceOutputs* outputs)
{
  if ((int)(data->tpcZS != nullptr) + (int)(data->o2Digits != nullptr && (data->tpcZS == nullptr || data->o2DigitsMC == nullptr)) + (int)(data->clusters != nullptr) + (int)(data->compressedClusters != nullptr) != 1) {
    throw std::runtime_error("Invalid input for gpu tracking");
  }

  std::vector<o2::tpc::Digit> gpuDigits[Sector::MAXSECTOR];
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> gpuDigitsMC[Sector::MAXSECTOR];
  ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer gpuDigitsMCConst[Sector::MAXSECTOR];

  GPUTrackingInOutDigits gpuDigitsMap;
  GPUTPCDigitsMCInput gpuDigitsMapMC;
  GPUTrackingInOutPointers ptrs;

  ptrs.tpcCompressedClusters = data->compressedClusters;
  ptrs.tpcZS = data->tpcZS;
  if (data->o2Digits) {
    const float zsThreshold = mTrackingCAO2Interface->getConfig().configReconstruction.tpcZSthreshold;
    const int maxContTimeBin = mTrackingCAO2Interface->getConfig().configEvent.continuousMaxTimeBin;
    for (int i = 0; i < Sector::MAXSECTOR; i++) {
      const auto& d = (*(data->o2Digits))[i];
      if (zsThreshold > 0 && data->tpcZS == nullptr) {
        gpuDigits[i].reserve(d.size());
      }
      for (int j = 0; j < d.size(); j++) {
        if (maxContTimeBin && d[j].getTimeStamp() >= maxContTimeBin) {
          throw std::runtime_error("Digit time bin exceeds time frame length");
        }
        if (zsThreshold > 0 && data->tpcZS == nullptr) {
          if (d[j].getChargeFloat() >= zsThreshold) {
            if (data->o2DigitsMC) {
              for (const auto& element : (*data->o2DigitsMC)[i]->getLabels(j)) {
                gpuDigitsMC[i].addElement(gpuDigits[i].size(), element);
              }
            }
            gpuDigits[i].emplace_back(d[j]);
          }
        }
      }
      if (zsThreshold > 0 && data->tpcZS == nullptr) {
        gpuDigitsMap.tpcDigits[i] = gpuDigits[i].data();
        gpuDigitsMap.nTPCDigits[i] = gpuDigits[i].size();
        if (data->o2DigitsMC) {
          gpuDigitsMC[i].flatten_to(gpuDigitsMCConst[i].first);
          gpuDigitsMCConst[i].second = gpuDigitsMCConst[i].first;
          gpuDigitsMapMC.v[i] = &gpuDigitsMCConst[i].second;
        }
      } else {
        gpuDigitsMap.tpcDigits[i] = (*(data->o2Digits))[i].data();
        gpuDigitsMap.nTPCDigits[i] = (*(data->o2Digits))[i].size();
        if (data->o2DigitsMC) {
          gpuDigitsMapMC.v[i] = (*data->o2DigitsMC)[i];
        }
      }
    }
    if (data->o2DigitsMC) {
      gpuDigitsMap.tpcDigitsMC = &gpuDigitsMapMC;
    }
    ptrs.tpcPackedDigits = &gpuDigitsMap;
  }
  ptrs.clustersNative = data->clusters;
  int retVal = mTrackingCAO2Interface->RunTracking(&ptrs, outputs);
  if (data->o2Digits || data->tpcZS || data->compressedClusters) {
    data->clusters = ptrs.clustersNative;
  }
  data->compressedClusters = ptrs.tpcCompressedClusters;
  data->outputTracks = {ptrs.outputTracksTPCO2, ptrs.nOutputTracksTPCO2};
  data->outputClusRefs = {ptrs.outputClusRefsTPCO2, ptrs.nOutputClusRefsTPCO2};
  data->outputTracksMCTruth = {ptrs.outputTracksTPCO2MC, ptrs.outputTracksTPCO2MC ? ptrs.nOutputTracksTPCO2 : 0};

  if (retVal || mTrackingCAO2Interface->getConfig().configInterface.dumpEvents >= 2) {
    return retVal;
  }

  mTrackingCAO2Interface->Clear(false);

  return (retVal);
}

void GPUCATracking::GetClusterErrors2(int row, float z, float sinPhi, float DzDs, short clusterState, float& ErrY2, float& ErrZ2) const
{
  if (mTrackingCAO2Interface == nullptr) {
    return;
  }
  mTrackingCAO2Interface->GetClusterErrors2(row, z, sinPhi, DzDs, clusterState, ErrY2, ErrZ2);
}

int GPUCATracking::registerMemoryForGPU(const void* ptr, size_t size)
{
  return mTrackingCAO2Interface->registerMemoryForGPU(ptr, size);
}

int GPUCATracking::unregisterMemoryForGPU(const void* ptr)
{
  return mTrackingCAO2Interface->unregisterMemoryForGPU(ptr);
}
