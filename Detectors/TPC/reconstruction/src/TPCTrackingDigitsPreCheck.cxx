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
#include "Framework/Logger.h"

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
  GPUTrackingInOutDigits tpcDigitsMap;
  GPUTPCDigitsMCInput tpcDigitsMapMC;
};

TPCTrackingDigitsPreCheck::precheckModifiedData TPCTrackingDigitsPreCheck::runPrecheck(o2::gpu::GPUTrackingInOutPointers* ptrs, o2::gpu::GPUO2InterfaceConfiguration* config)
{
  if (ptrs->tpcPackedDigits) {
    std::unique_ptr<precheckModifiedDataInternal> retVal = std::make_unique<precheckModifiedDataInternal>();
    retVal->tpcDigitsMap = *ptrs->tpcPackedDigits;
    const float zsThreshold = config->configReconstruction.tpc.zsThreshold;
    const int maxContTimeBin = config->configGRP.continuousMaxTimeBin;
    bool updateDigits = zsThreshold > 0 && ptrs->tpcZS == nullptr;
    const auto& d = ptrs->tpcPackedDigits;
    for (int i = 0; i < Sector::MAXSECTOR; i++) {
      if (updateDigits) {
        retVal->gpuDigits[i].reserve(d->nTPCDigits[i]);
      }
      int lastTime = 0;
      for (int j = 0; j < d->nTPCDigits[i]; j++) {
        int timeBin = d->tpcDigits[i][j].getTimeStamp();
        if (maxContTimeBin && timeBin >= maxContTimeBin) {
          static bool filterOutOfTF = getenv("TPC_WORKFLOW_FILTER_DIGITS_OUTSIDE_OF_TF") && atoi(getenv("TPC_WORKFLOW_FILTER_DIGITS_OUTSIDE_OF_TF"));
          if (filterOutOfTF) {
            continue;
          }
          throw std::runtime_error("Digit time bin exceeds time frame length");
        }
        if (timeBin < lastTime) {
          LOG(fatal) << "Incorrect digit ordering: time[" << i << "][" << j << "] = " << timeBin << " < lastTime = " << lastTime;
        }
        lastTime = timeBin;
        if (updateDigits) {
          if (d->tpcDigits[i][j].getChargeFloat() >= zsThreshold) {
            if (d->tpcDigitsMC) {
              for (const auto& element : d->tpcDigitsMC->v[i]->getLabels(j)) {
                retVal->gpuDigitsMC[i].addElement(retVal->gpuDigits[i].size(), element);
              }
            }
            retVal->gpuDigits[i].emplace_back(d->tpcDigits[i][j]);
          }
        }
      }
      if (updateDigits) {
        retVal->tpcDigitsMap.tpcDigits[i] = retVal->gpuDigits[i].data();
        retVal->tpcDigitsMap.nTPCDigits[i] = retVal->gpuDigits[i].size();
        if (ptrs->tpcPackedDigits->tpcDigitsMC) {
          retVal->gpuDigitsMC[i].flatten_to(retVal->gpuDigitsMCConst[i].first);
          retVal->gpuDigitsMCConst[i].second = retVal->gpuDigitsMCConst[i].first;
          retVal->tpcDigitsMapMC.v[i] = &retVal->gpuDigitsMCConst[i].second;
        }
      }
    }
    if (updateDigits) {
      if (ptrs->tpcPackedDigits->tpcDigitsMC) {
        retVal->tpcDigitsMap.tpcDigitsMC = &retVal->tpcDigitsMapMC;
      }
      ptrs->tpcPackedDigits = &retVal->tpcDigitsMap;
    }
    return precheckModifiedData(std::move(retVal));
  }
  return precheckModifiedData();
}
