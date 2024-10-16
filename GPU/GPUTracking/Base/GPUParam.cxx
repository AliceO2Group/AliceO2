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

/// \file GPUParam.cxx
/// \author David Rohr, Sergey Gorbunov

#include "GPUParam.h"
#include "GPUParamRTC.h"
#include "GPUDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonConstants.h"
#include "GPUTPCGMPolynomialFieldManager.h"
#include "GPUDataTypes.h"
#include "GPUConstantMem.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifdef GPUCA_ALIROOT_LIB
#include "AliTPCClusterParam.h"
#include "AliTPCcalibDB.h"
#include <iostream>
#endif
#include <cstring>
#include <tuple>
#ifdef GPUCA_HAVE_O2HEADERS
#include "DetectorsBase/Propagator.h"
#endif

#include "utils/qconfigrtc.h"

void GPUParam::SetDefaults(float solenoidBz)
{
  memset((void*)this, 0, sizeof(*this));
  new (&tpcGeometry) GPUTPCGeometry;
  new (&rec) GPUSettingsRec;
  occupancyMap = nullptr;
  occupancyTotal = 0;

#ifdef GPUCA_TPC_GEOMETRY_O2
  const float kErrorsY[4] = {0.06, 0.24, 0.12, 0.1};
  const float kErrorsZ[4] = {0.06, 0.24, 0.15, 0.1};

  UpdateRun3ClusterErrors(kErrorsY, kErrorsZ);
#else
  // clang-format off
  const float kParamS0Par[2][3][6] =
  {
    { { 6.45913474727e-04, 2.51547407970e-05, 1.57551113516e-02, 1.99872811635e-08, -5.86769729853e-03, 9.16301505640e-05 },
    { 9.71546804067e-04, 1.70938055817e-05, 2.17084009200e-02, 3.90275758377e-08, -1.68631039560e-03, 8.40498323669e-05 },
    { 7.27469159756e-05, 2.63869314949e-05, 3.29690799117e-02, -2.19274429725e-08, 1.77378822118e-02, 3.26595727529e-05 }
    }, {
    { 1.46874145139e-03, 6.36232061879e-06, 1.28665426746e-02, 1.19409449439e-07, 1.15883778781e-02, 1.32179644424e-04 },
    { 1.15970033221e-03, 1.30452335725e-05, 1.87015570700e-02, 5.39766737973e-08, 1.64790824056e-02, 1.44115634612e-04 },
    { 6.27940462437e-04, 1.78520094778e-05, 2.83537860960e-02, 1.16867742150e-08, 5.02607785165e-02, 1.88510020962e-04 } }
  };
  const float kParamErrorsSeeding0[2][3][4] =
  {
    { { 4.17516864836e-02, 1.87623649254e-04, 5.63788712025e-02, 5.38373768330e-01, },
    { 8.29434990883e-02, 2.03291710932e-04, 6.81538805366e-02, 9.70965325832e-01, },
    { 8.67543518543e-02, 2.10733342101e-04, 1.38366967440e-01, 2.55089461803e-01, }
    }, {
    { 5.96254616976e-02, 8.62886518007e-05, 3.61776389182e-02, 4.79704320431e-01, },
    { 6.12571723759e-02, 7.23929333617e-05, 3.93057651818e-02, 9.29222583771e-01, },
    { 6.58465921879e-02, 1.03639606095e-04, 6.07583411038e-02, 9.90289509296e-01, } }
  };
  // clang-format on

  for (int32_t i = 0; i < 2; i++) {
    for (int32_t j = 0; j < 3; j++) {
      for (int32_t k = 0; k < 6; k++) {
        ParamS0Par[i][j][k] = kParamS0Par[i][j][k];
      }
    }
  }

  for (int32_t i = 0; i < 2; i++) {
    for (int32_t j = 0; j < 3; j++) {
      for (int32_t k = 0; k < 4; k++) {
        ParamErrorsSeeding0[i][j][k] = kParamErrorsSeeding0[i][j][k];
      }
    }
  }
#endif

  par.dAlpha = 0.349066f;
  UpdateBzOnly(solenoidBz);
  par.dodEdx = 0;

  constexpr float plusZmin = 0.0529937;
  constexpr float plusZmax = 249.778;
  constexpr float minusZmin = -249.645;
  constexpr float minusZmax = -0.0799937;
  for (int32_t i = 0; i < GPUCA_NSLICES; i++) {
    const bool zPlus = (i < GPUCA_NSLICES / 2);
    SliceParam[i].ZMin = zPlus ? plusZmin : minusZmin;
    SliceParam[i].ZMax = zPlus ? plusZmax : minusZmax;
    int32_t tmp = i;
    if (tmp >= GPUCA_NSLICES / 2) {
      tmp -= GPUCA_NSLICES / 2;
    }
    if (tmp >= GPUCA_NSLICES / 4) {
      tmp -= GPUCA_NSLICES / 2;
    }
    SliceParam[i].Alpha = 0.174533f + par.dAlpha * tmp;
    SliceParam[i].CosAlpha = CAMath::Cos(SliceParam[i].Alpha);
    SliceParam[i].SinAlpha = CAMath::Sin(SliceParam[i].Alpha);
    SliceParam[i].AngleMin = SliceParam[i].Alpha - par.dAlpha / 2.f;
    SliceParam[i].AngleMax = SliceParam[i].Alpha + par.dAlpha / 2.f;
  }

  par.assumeConstantBz = false;
  par.toyMCEventsFlag = false;
  par.continuousTracking = false;
  continuousMaxTimeBin = 0;
  par.debugLevel = 0;
  par.earlyTpcTransform = false;
}

void GPUParam::UpdateSettings(const GPUSettingsGRP* g, const GPUSettingsProcessing* p, const GPURecoStepConfiguration* w)
{
  if (g) {
    UpdateBzOnly(g->solenoidBzNominalGPU);
    par.assumeConstantBz = g->constBz;
    par.toyMCEventsFlag = g->homemadeEvents;
    par.continuousTracking = g->continuousMaxTimeBin != 0;
    continuousMaxTimeBin = g->continuousMaxTimeBin == -1 ? GPUSettings::TPC_MAX_TF_TIME_BIN : g->continuousMaxTimeBin;
  }
  par.earlyTpcTransform = rec.tpc.forceEarlyTransform == -1 ? (!par.continuousTracking) : rec.tpc.forceEarlyTransform;
  qptB5Scaler = CAMath::Abs(bzkG) > 0.1f ? CAMath::Abs(bzkG) / 5.006680f : 1.f; // Repeat here, since passing in g is optional
  if (p) {
    par.debugLevel = p->debugLevel;
    UpdateRun3ClusterErrors(p->param.tpcErrorParamY, p->param.tpcErrorParamZ);
  }
  if (w) {
    par.dodEdx = dodEdxDownscaled = w->steps.isSet(GPUDataTypes::RecoStep::TPCdEdx);
    if (par.dodEdx && p && p->tpcDownscaledEdx != 0) {
      dodEdxDownscaled = (rand() % 100) < p->tpcDownscaledEdx;
    }
  }
}

void GPUParam::UpdateBzOnly(float newSolenoidBz)
{
  bzkG = newSolenoidBz;
  bzCLight = bzkG * GPUCA_NAMESPACE::gpu::gpu_common_constants::kCLight;
  polynomialField.Reset();
  if (par.assumeConstantBz) {
    GPUTPCGMPolynomialFieldManager::GetPolynomialField(GPUTPCGMPolynomialFieldManager::kUniform, bzkG, polynomialField);
  } else {
    GPUTPCGMPolynomialFieldManager::GetPolynomialField(bzkG, polynomialField);
  }
  qptB5Scaler = CAMath::Abs(bzkG) > 0.1f ? CAMath::Abs(bzkG) / 5.006680f : 1.f;
}

void GPUParam::SetDefaults(const GPUSettingsGRP* g, const GPUSettingsRec* r, const GPUSettingsProcessing* p, const GPURecoStepConfiguration* w)
{
  SetDefaults(g->solenoidBzNominalGPU);
  if (r) {
    rec = *r;
    if (rec.fitPropagateBzOnly == -1) {
      rec.fitPropagateBzOnly = rec.tpc.nWays - 1;
    }
  }
  UpdateSettings(g, p, w);
}

void GPUParam::UpdateRun3ClusterErrors(const float* yErrorParam, const float* zErrorParam)
{
#ifdef GPUCA_TPC_GEOMETRY_O2
  for (int32_t yz = 0; yz < 2; yz++) {
    const float* param = yz ? zErrorParam : yErrorParam;
    for (int32_t rowType = 0; rowType < 4; rowType++) {
      constexpr int32_t regionMap[4] = {0, 4, 6, 8};
      ParamErrors[yz][rowType][0] = param[0] * param[0];
      ParamErrors[yz][rowType][1] = param[1] * param[1] * tpcGeometry.PadHeightByRegion(regionMap[rowType]);
      ParamErrors[yz][rowType][2] = param[2] * param[2] / tpcGeometry.TPCLength() / tpcGeometry.PadHeightByRegion(regionMap[rowType]);
      ParamErrors[yz][rowType][3] = param[3] * param[3] * rec.tpc.clusterErrorOccupancyScaler * rec.tpc.clusterErrorOccupancyScaler;
    }
  }
#endif
}

#ifndef GPUCA_ALIROOT_LIB
void GPUParam::LoadClusterErrors(bool Print)
{
}
#else

#include <iomanip>
#include <iostream>
void GPUParam::LoadClusterErrors(bool Print)
{
  // update of calculated values
  const AliTPCClusterParam* clparam = AliTPCcalibDB::Instance()->GetClusterParam();
  if (!clparam) {
    std::cout << "Error: GPUParam::LoadClusterErrors():: No AliTPCClusterParam instance found !!!! " << std::endl;
    return;
  }

  for (int32_t i = 0; i < 2; i++) {
    for (int32_t j = 0; j < 3; j++) {
      for (int32_t k = 0; k < 6; k++) {
        ParamS0Par[i][j][k] = clparam->GetParamS0Par(i, j, k);
      }
    }
  }

  for (int32_t i = 0; i < 2; i++) {
    for (int32_t j = 0; j < 3; j++) {
      for (int32_t k = 0; k < 4; k++) {
        ParamErrorsSeeding0[i][j][k] = clparam->GetParamRMS0(i, j, k);
      }
    }
  }

  if (Print) {
    typedef std::numeric_limits<float> flt;
    std::cout << std::scientific;
#if __cplusplus >= 201103L
    std::cout << std::setprecision(flt::max_digits10 + 2);
#endif
    std::cout << "ParamS0Par[2][3][7]=" << std::endl;
    std::cout << " { " << std::endl;
    for (int32_t i = 0; i < 2; i++) {
      std::cout << "   { " << std::endl;
      for (int32_t j = 0; j < 3; j++) {
        std::cout << " { ";
        for (int32_t k = 0; k < 6; k++) {
          std::cout << ParamS0Par[i][j][k] << ", ";
        }
        std::cout << " }, " << std::endl;
      }
      std::cout << "   }, " << std::endl;
    }
    std::cout << " }; " << std::endl;

    std::cout << "ParamErrorsSeeding0[2][3][4]=" << std::endl;
    std::cout << " { " << std::endl;
    for (int32_t i = 0; i < 2; i++) {
      std::cout << "   { " << std::endl;
      for (int32_t j = 0; j < 3; j++) {
        std::cout << " { ";
        for (int32_t k = 0; k < 4; k++) {
          std::cout << ParamErrorsSeeding0[i][j][k] << ", ";
        }
        std::cout << " }, " << std::endl;
      }
      std::cout << "   }, " << std::endl;
    }
    std::cout << " }; " << std::endl;

    const THnBase* waveMap = clparam->GetWaveCorrectionMap();
    const THnBase* resYMap = clparam->GetResolutionYMap();
    std::cout << "waveMap = " << (void*)waveMap << std::endl;
    std::cout << "resYMap = " << (void*)resYMap << std::endl;
  }
}
#endif

void GPUParamRTC::setFrom(const GPUParam& param)
{
  memcpy((void*)this, (void*)&param, sizeof(param));
}

std::string GPUParamRTC::generateRTCCode(const GPUParam& param, bool useConstexpr)
{
  return "#ifndef GPUCA_GPUCODE_DEVICE\n"
         "#include <string>\n"
         "#include <vector>\n"
         "#include <cstdint>\n"
         "#include <cstddef>\n"
         "#endif\n"
         "namespace o2::gpu { class GPUDisplayFrontendInterface; }\n" +
         qConfigPrintRtc(std::make_tuple(&param.rec.tpc, &param.rec.trd, &param.rec, &param.par), useConstexpr);
}

static_assert(sizeof(GPUCA_NAMESPACE::gpu::GPUParam) == sizeof(GPUCA_NAMESPACE::gpu::GPUParamRTC), "RTC param size mismatch");
