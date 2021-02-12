// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#ifdef HAVE_O2HEADERS
#include "DetectorsBase/Propagator.h"
#endif

#include "utils/qconfigrtc.h"

void GPUParam::SetDefaults(float solenoidBz)
{
  memset((void*)this, 0, sizeof(*this));
  new (&tpcGeometry) GPUTPCGeometry;
  new (&rec) GPUSettingsRec;

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
  const float kParamRMS0[2][3][4] =
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

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 6; k++) {
        ParamS0Par[i][j][k] = kParamS0Par[i][j][k];
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        ParamRMS0[i][j][k] = kParamRMS0[i][j][k];
      }
    }
  }

  par.DAlpha = 0.349066f;
  par.BzkG = solenoidBz;
  constexpr double kCLight = 0.000299792458f;
  par.ConstBz = solenoidBz * kCLight;
  par.dodEdx = 0;

  constexpr float plusZmin = 0.0529937;
  constexpr float plusZmax = 249.778;
  constexpr float minusZmin = -249.645;
  constexpr float minusZmax = -0.0799937;
  for (int i = 0; i < GPUCA_NSLICES; i++) {
    const bool zPlus = (i < GPUCA_NSLICES / 2);
    SliceParam[i].ZMin = zPlus ? plusZmin : minusZmin;
    SliceParam[i].ZMax = zPlus ? plusZmax : minusZmax;
    int tmp = i;
    if (tmp >= GPUCA_NSLICES / 2) {
      tmp -= GPUCA_NSLICES / 2;
    }
    if (tmp >= GPUCA_NSLICES / 4) {
      tmp -= GPUCA_NSLICES / 2;
    }
    SliceParam[i].Alpha = 0.174533 + par.DAlpha * tmp;
    SliceParam[i].CosAlpha = CAMath::Cos(SliceParam[i].Alpha);
    SliceParam[i].SinAlpha = CAMath::Sin(SliceParam[i].Alpha);
    SliceParam[i].AngleMin = SliceParam[i].Alpha - par.DAlpha / 2.f;
    SliceParam[i].AngleMax = SliceParam[i].Alpha + par.DAlpha / 2.f;
  }

  par.AssumeConstantBz = false;
  par.ToyMCEventsFlag = false;
  par.ContinuousTracking = false;
  par.continuousMaxTimeBin = 0;
  par.debugLevel = 0;
  par.resetTimers = false;
  par.earlyTpcTransform = false;

  polynomialField.Reset(); // set very wrong initial value in order to see if the field was not properly initialised
  GPUTPCGMPolynomialFieldManager::GetPolynomialField(par.BzkG, polynomialField);
}

void GPUParam::UpdateEventSettings(const GPUSettingsEvent* e, const GPUSettingsProcessing* p)
{
  if (e) {
    par.AssumeConstantBz = e->constBz;
    par.ToyMCEventsFlag = e->homemadeEvents;
    par.ContinuousTracking = e->continuousMaxTimeBin != 0;
    par.continuousMaxTimeBin = e->continuousMaxTimeBin == -1 ? GPUSettings::TPC_MAX_TF_TIME_BIN : e->continuousMaxTimeBin;
    polynomialField.Reset();
    if (par.AssumeConstantBz) {
      GPUTPCGMPolynomialFieldManager::GetPolynomialField(GPUTPCGMPolynomialFieldManager::kUniform, par.BzkG, polynomialField);
    } else {
      GPUTPCGMPolynomialFieldManager::GetPolynomialField(par.BzkG, polynomialField);
    }
  }
  if (p) {
    par.debugLevel = p->debugLevel;
    par.resetTimers = p->resetTimers;
  }
  par.earlyTpcTransform = rec.ForceEarlyTPCTransform == -1 ? (!par.ContinuousTracking) : rec.ForceEarlyTPCTransform;
}

void GPUParam::SetDefaults(const GPUSettingsEvent* e, const GPUSettingsRec* r, const GPUSettingsProcessing* p, const GPURecoStepConfiguration* w)
{
  SetDefaults(e->solenoidBz);
  if (w) {
    par.dodEdx = w->steps.isSet(GPUDataTypes::RecoStep::TPCdEdx);
  }
  if (r) {
    rec = *r;
    if (rec.fitPropagateBzOnly == -1) {
      rec.fitPropagateBzOnly = rec.NWays - 1;
    }
  }
  UpdateEventSettings(e, p);
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

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 6; k++) {
        ParamS0Par[i][j][k] = clparam->GetParamS0Par(i, j, k);
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        ParamRMS0[i][j][k] = clparam->GetParamRMS0(i, j, k);
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
    for (int i = 0; i < 2; i++) {
      std::cout << "   { " << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << " { ";
        for (int k = 0; k < 6; k++) {
          std::cout << ParamS0Par[i][j][k] << ", ";
        }
        std::cout << " }, " << std::endl;
      }
      std::cout << "   }, " << std::endl;
    }
    std::cout << " }; " << std::endl;

    std::cout << "ParamRMS0[2][3][4]=" << std::endl;
    std::cout << " { " << std::endl;
    for (int i = 0; i < 2; i++) {
      std::cout << "   { " << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << " { ";
        for (int k = 0; k < 4; k++) {
          std::cout << ParamRMS0[i][j][k] << ", ";
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
  memcpy((char*)this + sizeof(gpu_rtc::GPUSettingsRec) + sizeof(gpu_rtc::GPUSettingsParam), (char*)&param + sizeof(GPUSettingsRec) + sizeof(GPUSettingsParam), sizeof(param) - sizeof(GPUSettingsRec) - sizeof(GPUSettingsParam));
  qConfigConvertRtc(this->rec, param.rec);
  qConfigConvertRtc(this->par, param.par);
}

std::string GPUParamRTC::generateRTCCode(const GPUParam& param, bool useConstexpr)
{
  return "namespace o2::gpu { class GPUDisplayBackend; }\n" + qConfigPrintRtc(std::make_tuple(&param.rec, &param.par), useConstexpr);
}

static_assert(alignof(GPUCA_NAMESPACE::gpu::GPUParam) == alignof(GPUCA_NAMESPACE::gpu::GPUSettingsRec));
static_assert(alignof(GPUCA_NAMESPACE::gpu::GPUParam) == alignof(GPUCA_NAMESPACE::gpu::GPUSettingsParam));
static_assert(sizeof(GPUCA_NAMESPACE::gpu::GPUParam) - sizeof(GPUCA_NAMESPACE::gpu::GPUParamRTC) == sizeof(GPUCA_NAMESPACE::gpu::GPUSettingsRec) + sizeof(GPUCA_NAMESPACE::gpu::GPUSettingsParam) - sizeof(GPUCA_NAMESPACE::gpu::gpu_rtc::GPUSettingsRec) - sizeof(GPUCA_NAMESPACE::gpu::gpu_rtc::GPUSettingsParam));
static_assert(sizeof(GPUParam) % alignof(GPUConstantMem) == 0 && sizeof(GPUParamRTC) % alignof(GPUConstantMem) == 0, "Size of both GPUParam and of GPUParamRTC must be a multiple of the alignmeent of GPUConstantMem");

o2::base::Propagator* GPUParam::GetDefaultO2Propagator(bool useGPUField) const
{
  o2::base::Propagator* prop = nullptr;
#ifdef HAVE_O2HEADERS
  if (useGPUField == false) {
    throw std::runtime_error("o2 propagator withouzt gpu field unsupported");
  }
  prop = o2::base::Propagator::Instance(useGPUField);
  if (useGPUField) {
    prop->setGPUField(&polynomialField);
    prop->setBz(polynomialField.GetNominalBz());
  }
#else
  throw std::runtime_error("o2 propagator unsupported");
#endif
  return prop;
}
