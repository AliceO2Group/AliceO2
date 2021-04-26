// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCTrackingDigitsPreCheck.cxx
/// \author David Rohr

#include "TPCReconstruction/TPCTrackingDigitsPreCheck.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/ClusterNative.h"

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

TPCTrackingDigitsPreCheck::precheckModifiedData::precheckModifiedData() = default;
TPCTrackingDigitsPreCheck::precheckModifiedData::~precheckModifiedData() = default;
TPCTrackingDigitsPreCheck::precheckModifiedData::precheckModifiedData(std::unique_ptr<precheckModifiedDataInternal>&& v) : data(std::move(v)) {}

struct TPCTrackingDigitsPreCheck::precheckModifiedDataInternal {
  std::vector<o2::tpc::Digit> gpuDigits[Sector::MAXSECTOR];
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> gpuDigitsMC[Sector::MAXSECTOR];
  ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer gpuDigitsMCConst[Sector::MAXSECTOR];
  GPUTrackingInOutDigits gpuDigitsMap;
  GPUTPCDigitsMCInput gpuDigitsMapMC;
};

TPCTrackingDigitsPreCheck::precheckModifiedData TPCTrackingDigitsPreCheck::runPrecheck(o2::gpu::GPUO2InterfaceIOPtrs* data, o2::gpu::GPUTrackingInOutPointers* ptrs, o2::gpu::GPUO2InterfaceConfiguration* config)
{
  if (data->o2Digits) {
    std::unique_ptr<precheckModifiedDataInternal> retVal = std::make_unique<precheckModifiedDataInternal>();
    const float zsThreshold = config->configReconstruction.tpcZSthreshold;
    const int maxContTimeBin = config->configGRP.continuousMaxTimeBin;
    for (int i = 0; i < Sector::MAXSECTOR; i++) {
      const auto& d = (*data->o2Digits)[i];
      if (zsThreshold > 0 && data->tpcZS == nullptr) {
        retVal->gpuDigits[i].reserve(d.size());
      }
      for (int j = 0; j < d.size(); j++) {
        if (maxContTimeBin && d[j].getTimeStamp() >= maxContTimeBin) {
          throw std::runtime_error("Digit time bin exceeds time frame length");
        }
        if (zsThreshold > 0 && data->tpcZS == nullptr) {
          if (d[j].getChargeFloat() >= zsThreshold) {
            if (data->o2DigitsMC) {
              for (const auto& element : (*data->o2DigitsMC)[i]->getLabels(j)) {
                retVal->gpuDigitsMC[i].addElement(retVal->gpuDigits[i].size(), element);
              }
            }
            retVal->gpuDigits[i].emplace_back(d[j]);
          }
        }
      }
      if (zsThreshold > 0 && data->tpcZS == nullptr) {
        retVal->gpuDigitsMap.tpcDigits[i] = retVal->gpuDigits[i].data();
        retVal->gpuDigitsMap.nTPCDigits[i] = retVal->gpuDigits[i].size();
        if (data->o2DigitsMC) {
          retVal->gpuDigitsMC[i].flatten_to(retVal->gpuDigitsMCConst[i].first);
          retVal->gpuDigitsMCConst[i].second = retVal->gpuDigitsMCConst[i].first;
          retVal->gpuDigitsMapMC.v[i] = &retVal->gpuDigitsMCConst[i].second;
        }
      } else {
        retVal->gpuDigitsMap.tpcDigits[i] = (*(data->o2Digits))[i].data();
        retVal->gpuDigitsMap.nTPCDigits[i] = (*(data->o2Digits))[i].size();
        if (data->o2DigitsMC) {
          retVal->gpuDigitsMapMC.v[i] = (*data->o2DigitsMC)[i];
        }
      }
    }
    if (data->o2DigitsMC) {
      retVal->gpuDigitsMap.tpcDigitsMC = &retVal->gpuDigitsMapMC;
    }
    ptrs->tpcPackedDigits = &retVal->gpuDigitsMap;
    return precheckModifiedData(std::move(retVal));
  }
  return precheckModifiedData();
}
