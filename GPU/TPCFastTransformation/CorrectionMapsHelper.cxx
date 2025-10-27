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

#include "CorrectionMapsHelper.h"
#include "Framework/ConfigParamRegistry.h"
using namespace o2::gpu;
using namespace o2::tpc;

void CorrectionMapsHelper::setCorrMap(const TPCFastTransformPOD* m)
{
  // non-owning: just store the pointer, clear any previously owned buffer
#if !defined(GPUCA_GPUCODE_DEVICE)
  mCorrMapBuffer.clear();
#endif
  mCorrMap = m;
}

void CorrectionMapsHelper::setCorrMap(std::vector<char>&& buffer)
{
  mCorrMapBuffer = std::move(buffer);
  mCorrMap = &TPCFastTransformPOD::get(mCorrMapBuffer.data());
}

CorrectionMapsLoaderGloOpts CorrectionMapsHelper::parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts)
{
  CorrectionMapsLoaderGloOpts tpcopt;
  auto lumiTypeVal = opts.get<int>("lumi-type");
  if (lumiTypeVal < -1 || lumiTypeVal > 2) {
    LOGP(fatal, "Invalid lumi-type value: {}", lumiTypeVal);
  }
  tpcopt.lumiType = static_cast<LumiScaleType>(lumiTypeVal);

  auto lumiModeVal = opts.get<int>("corrmap-lumi-mode");
  if (lumiModeVal < -1 || lumiModeVal > 2) {
    LOGP(fatal, "Invalid corrmap-lumi-mode value: {}", lumiModeVal);
  }
  tpcopt.lumiMode = static_cast<LumiScaleMode>(lumiModeVal);

  tpcopt.enableMShapeCorrection = opts.get<bool>("enable-M-shape-correction");
  tpcopt.requestCTPLumi = !opts.get<bool>("disable-ctp-lumi-request");
  tpcopt.checkCTPIDCconsistency = !opts.get<bool>("disable-lumi-type-consistency-check");
  if (!tpcopt.requestCTPLumi && tpcopt.lumiType == LumiScaleType::CTPLumi) {
    LOGP(fatal, "Scaling with CTP Lumi is requested but this input is disabled");
  }
  return tpcopt;
}
