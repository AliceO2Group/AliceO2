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

/// \file GPUO2Interface.cxx
/// \author David Rohr

#include "GPUO2InterfaceUtils.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCPadGainCalib.h"
#include "CalibdEdxContainer.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "GPUParam.h"
#include "GPUReconstructionConvert.h"
#include "DataFormatsTPC/Digit.h"
#include "DetectorsRaw/RDHUtils.h"
#include "TPCBase/CRU.h"
#include "TPCBase/RDHUtils.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include <gsl/span>

using namespace o2::gpu;

using DigitArray = std::array<gsl::span<const o2::tpc::Digit>, o2::tpc::Sector::MAXSECTOR>;

std::unique_ptr<TPCPadGainCalib> GPUO2InterfaceUtils::getPadGainCalibDefault()
{
  return std::make_unique<TPCPadGainCalib>();
}

std::unique_ptr<TPCPadGainCalib> GPUO2InterfaceUtils::getPadGainCalib(const o2::tpc::CalDet<float>& in)
{
  return std::make_unique<TPCPadGainCalib>(in);
}

std::unique_ptr<o2::tpc::CalibdEdxContainer> GPUO2InterfaceUtils::getCalibdEdxContainerDefault()
{
  return std::make_unique<o2::tpc::CalibdEdxContainer>();
}

template <>
void GPUO2InterfaceUtils::RunZSEncoder<DigitArray>(const DigitArray& in, std::unique_ptr<unsigned long long int[]>* outBuffer, unsigned int* outSizes, o2::raw::RawFileWriter* raw, const o2::InteractionRecord* ir, int version, bool verify, float threshold, bool padding, std::function<void(std::vector<o2::tpc::Digit>&)> digitsFilter)
{
  GPUParam param;
  param.SetDefaults(5.00668);
  o2::gpu::GPUReconstructionConvert::RunZSEncoder(in, outBuffer, outSizes, raw, ir, param, version, verify, threshold, padding, digitsFilter);
}
template <>
void GPUO2InterfaceUtils::RunZSEncoder<DigitArray>(const DigitArray& in, std::unique_ptr<unsigned long long int[]>* outBuffer, unsigned int* outSizes, o2::raw::RawFileWriter* raw, const o2::InteractionRecord* ir, GPUO2InterfaceConfiguration& config, int version, bool verify, bool padding, std::function<void(std::vector<o2::tpc::Digit>&)> digitsFilter)
{
  GPUParam param;
  param.SetDefaults(&config.configGRP, &config.configReconstruction, &config.configProcessing, nullptr);
  o2::gpu::GPUReconstructionConvert::RunZSEncoder(in, outBuffer, outSizes, raw, ir, param, version, verify, config.configReconstruction.tpc.zsThreshold, padding, digitsFilter);
}

void GPUO2InterfaceUtils::GPUReconstructionZSDecoder::DecodePage(std::vector<o2::tpc::Digit>& outputBuffer, const void* page, unsigned int tfFirstOrbit, const GPUParam* param, unsigned int triggerBC)
{
  const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)page;
  if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
    return;
  }
  o2::tpc::TPCZSHDR* const hdr = (o2::tpc::TPCZSHDR*)(o2::tpc::rdh_utils::getLink(o2::raw::RDHUtils::getFEEID(*rdh)) == o2::tpc::rdh_utils::DLBZSLinkID ? ((const char*)page + o2::raw::RDHUtils::getMemorySize(*rdh) - sizeof(o2::tpc::TPCZSHDRV2)) : ((const char*)page + sizeof(o2::header::RAWDataHeader)));

  if (mDecoders.size() < hdr->version + 1) {
    mDecoders.resize(hdr->version + 1);
  }
  if (mDecoders[hdr->version] == nullptr) {
    mDecoders[hdr->version] = GPUReconstructionConvert::GetDecoder(hdr->version, param);
  }
  mDecoders[hdr->version](outputBuffer, page, tfFirstOrbit, triggerBC);
}

std::unique_ptr<GPUParam> GPUO2InterfaceUtils::getFullParam(float solenoidBz, unsigned int nHbfPerTf, std::unique_ptr<GPUO2InterfaceConfiguration>* pConfiguration, std::unique_ptr<GPUSettingsO2>* pO2Settings, bool* autoMaxTimeBin)
{
  std::unique_ptr<GPUParam> retVal = std::make_unique<GPUParam>();
  std::unique_ptr<GPUO2InterfaceConfiguration> tmpConfig;
  if (!pConfiguration) {
    tmpConfig = std::make_unique<GPUO2InterfaceConfiguration>();
    pConfiguration = &tmpConfig;
    (*pConfiguration)->configGRP.continuousMaxTimeBin = -1;
  } else if (!*pConfiguration) {
    *pConfiguration = std::make_unique<GPUO2InterfaceConfiguration>();
    (*pConfiguration)->configGRP.continuousMaxTimeBin = -1;
  }
  (*pConfiguration)->configGRP.solenoidBzNominalGPU = solenoidBz;
  if (pO2Settings && *pO2Settings) {
    **pO2Settings = (*pConfiguration)->ReadConfigurableParam();
  } else if (pO2Settings) {
    *pO2Settings = std::make_unique<GPUSettingsO2>((*pConfiguration)->ReadConfigurableParam());
  } else {
    (*pConfiguration)->ReadConfigurableParam();
  }
  if (nHbfPerTf == 0) {
    nHbfPerTf = 256;
  }
  if (autoMaxTimeBin) {
    *autoMaxTimeBin = (*pConfiguration)->configGRP.continuousMaxTimeBin == -1;
  }
  if ((*pConfiguration)->configGRP.continuousMaxTimeBin == -1) {
    (*pConfiguration)->configGRP.continuousMaxTimeBin = (nHbfPerTf * o2::constants::lhc::LHCMaxBunches + 2 * o2::tpc::constants::LHCBCPERTIMEBIN - 2) / o2::tpc::constants::LHCBCPERTIMEBIN;
  }
  retVal->SetDefaults(&(*pConfiguration)->configGRP, &(*pConfiguration)->configReconstruction, &(*pConfiguration)->configProcessing, nullptr);
  return retVal;
}

std::shared_ptr<GPUParam> GPUO2InterfaceUtils::getFullParamShared(float solenoidBz, unsigned int nHbfPerTf, std::unique_ptr<GPUO2InterfaceConfiguration>* pConfiguration, std::unique_ptr<GPUSettingsO2>* pO2Settings, bool* autoMaxTimeBin)
{
  return std::move(getFullParam(solenoidBz, nHbfPerTf, pConfiguration, pO2Settings, autoMaxTimeBin));
}
