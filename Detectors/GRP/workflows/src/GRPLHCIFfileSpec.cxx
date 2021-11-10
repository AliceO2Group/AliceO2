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

/// @file GRPLHCIFfileSpec.cxx

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GRPWorkflows/GRPLHCIFfileSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"

#include <chrono>
#include <cstdint>

using namespace o2::framework;
using TFType = uint64_t;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;
using LHCIFData = o2::parameters::LHCIFData;

namespace o2
{
namespace grp
{

void GRPLHCIFfileProcessor::init(o2::framework::InitContext& ic)
{
  mVerbose = ic.options().get<bool>("use-verbose-mode");
  LOG(INFO) << " ************************* Verbose?" << mVerbose;
}

//__________________________________________________________________

void GRPLHCIFfileProcessor::run(o2::framework::ProcessingContext& pc)
{
  auto configBuff = pc.inputs().get<gsl::span<char>>("inputConfig");
  auto configFileName = pc.inputs().get<std::string>("inputConfigFileName");
  auto timer = std::chrono::duration_cast<std::chrono::milliseconds>(HighResClock::now().time_since_epoch()).count();
  LOG(INFO) << "got input file " << configFileName << " of size " << configBuff.size();
  mReader.loadLHCIFfile(configBuff);
  std::vector<std::pair<long, std::vector<int32_t>>> beamEnergy;
  std::vector<std::pair<long, std::vector<std::string>>> injScheme;
  std::vector<std::pair<long, std::vector<std::string>>> fillNb;
  std::vector<std::pair<long, std::vector<int32_t>>> atomicNbB1;
  std::vector<std::pair<long, std::vector<int32_t>>> atomicNbB2;

  int nEleBeamEn, nEleInjSch, nEleFillNb, nEleAtNbB1, nEleAtNbB2 = 0;
  int nMeasBeamEn, nMeasInjSch, nMeasFillNb, nMeasAtNbB1, nMeasAtNbB2 = 0;
  std::string type{""};

  LHCIFData lhcifdata;

  mReader.readValue<int32_t>("BEAM_ENERGY", type, nEleBeamEn, nMeasBeamEn, beamEnergy);
  if (nMeasBeamEn == 0) {
    LOG(FATAL) << "Beam energy not present";
  }
  if (nEleBeamEn != 1 || nMeasBeamEn != 1) {
    LOG(ERROR) << "More than one value/measurement found for Beam Energy, keeping the last one";
  }
  LOG(INFO) << "beam energy size = " << beamEnergy.size();
  lhcifdata.setBeamEnergy(beamEnergy.back().first, beamEnergy.back().second.back());

  mReader.readValue<std::string>("INJECTION_SCHEME", type, nEleInjSch, nMeasInjSch, injScheme);
  if (nMeasInjSch == 0) {
    LOG(FATAL) << "Injection scheme not present";
  }
  if (nEleInjSch != 1 || nMeasInjSch != 1) {
    LOG(ERROR) << "More than one value/measurement found for Injection Scheme, keeping the last one";
  }
  lhcifdata.setInjectionScheme(injScheme.back().first, injScheme.back().second.back());

  mReader.readValue<std::string>("FILL_NUMBER", type, nEleFillNb, nMeasFillNb, fillNb);
  if (nMeasFillNb == 0) {
    LOG(FATAL) << "Fill Number not present";
  }
  if (nEleFillNb != 1 || nMeasFillNb != 1) {
    LOG(ERROR) << "More than one value/measurement found for Fill Number, keeping the last one";
  }
  lhcifdata.setFillNumber(fillNb.back().first, atoi(fillNb.back().second.back().c_str()));

  mReader.readValue<int32_t>("ATOMIC_NUMBER_B1", type, nEleAtNbB1, nMeasAtNbB1, atomicNbB1);
  if (nMeasAtNbB1 == 0) {
    LOG(FATAL) << "Atomic Number Beam 1 not present";
  }
  if (nEleAtNbB1 != 1 || nMeasAtNbB1 != 1) {
    LOG(ERROR) << "More than one value/measurement found for Atomic Number Beam 1, keeping the last one";
  }
  lhcifdata.setAtomicNumberB1(atomicNbB1.back().first, atomicNbB1.back().second.back());

  mReader.readValue<int32_t>("ATOMIC_NUMBER_B2", type, nEleAtNbB2, nMeasAtNbB2, atomicNbB2);
  if (nMeasAtNbB2 == 0) {
    LOG(FATAL) << "Atomic Number Beam 2 not present";
  }
  if (nEleAtNbB2 != 1 || nMeasAtNbB2 != 1) {
    LOG(ERROR) << "More than one value/measurement found for Atomic Number Beam 2, keeping the last one";
  }
  lhcifdata.setAtomicNumberB2(atomicNbB2.back().first, atomicNbB2.back().second.back());

  if (mVerbose) {
    LOG(INFO) << " **** Beam Energy ****";
    for (auto& el : beamEnergy) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(INFO) << " **** Injection Scheme ****";
    for (auto& el : injScheme) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(INFO) << " **** Fill Number ****";
    for (auto& el : fillNb) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(INFO) << " **** Atomic Number Beam 1 ****";
    for (auto& el : atomicNbB1) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(INFO) << " **** Atomic Number B2 ****";
    for (auto& el : atomicNbB2) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
  }

  sendOutput(pc.outputs(), timer, lhcifdata);
}

//__________________________________________________________________

void GRPLHCIFfileProcessor::endOfStream(o2::framework::EndOfStreamContext& ec)
{
}

//__________________________________________________________________

void GRPLHCIFfileProcessor::sendOutput(DataAllocator& output, long tf, const LHCIFData& lhcifdata)
{
  // sending output to CCDB

  constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;

  using clbUtils = o2::calibration::Utils;
  auto clName = o2::utils::MemFileHelper::getClassName(lhcifdata);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> md;
  md.emplace("created by", "dpl");
  o2::ccdb::CcdbObjectInfo info("GRP/Data/LHCIFData", clName, flName, md, tf, INFINITE_TF);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&lhcifdata, &info);
  LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_LHCIFData", 0}, *image.get()); // vector<char>
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_LHCIFData", 0}, info);         // root-serialized
}

} // namespace grp

namespace framework
{
DataProcessorSpec getGRPLHCIFfileSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_LHCIFData"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_LHCIFData"}, Lifetime::Sporadic);
  return DataProcessorSpec{
    "grp-lhc-if-file-processor",
    Inputs{{"inputConfig", "GRP", "DCS_CONFIG_FILE", Lifetime::Timeframe},
           {"inputConfigFileName", "GRP", "DCS_CONFIG_NAME", Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::grp::GRPLHCIFfileProcessor>()},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2
