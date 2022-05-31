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
#include "CommonTypes/Units.h"

#include <chrono>
#include <cstdint>

using namespace o2::framework;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;
using GRPLHCIFData = o2::parameters::GRPLHCIFData;

namespace o2
{
namespace grp
{

void GRPLHCIFfileProcessor::init(o2::framework::InitContext& ic)
{
  mVerbose = ic.options().get<bool>("use-verbose-mode");
  LOG(info) << " ************************* Verbose?" << mVerbose;
}

//__________________________________________________________________

void GRPLHCIFfileProcessor::run(o2::framework::ProcessingContext& pc)
{
  auto configBuff = pc.inputs().get<gsl::span<char>>("inputConfig");
  auto configFileName = pc.inputs().get<std::string>("inputConfigFileName");
  auto timer = std::chrono::duration_cast<std::chrono::milliseconds>(HighResClock::now().time_since_epoch()).count();
  LOG(info) << "got input file " << configFileName << " of size " << configBuff.size();
  mReader.loadLHCIFfile(configBuff);
  std::vector<std::pair<long, std::vector<int32_t>>> beamEnergy;
  std::vector<std::pair<long, std::vector<std::string>>> injScheme;
  std::vector<std::pair<long, std::vector<std::string>>> fillNb;
  std::vector<std::pair<long, std::vector<int32_t>>> atomicNbB1;
  std::vector<std::pair<long, std::vector<int32_t>>> atomicNbB2;
  std::vector<std::pair<long, std::vector<o2::units::AngleRad_t>>> crossAngle;
  std::vector<std::pair<long, std::vector<int32_t>>> bunchConfigB1;
  std::vector<std::pair<long, std::vector<int32_t>>> bunchConfigB2;

  int nEle = 0;
  int nMeas = 0;
  std::string type{""};

  GRPLHCIFData lhcifdata;

  // Beam Energy
  mReader.readValue<int32_t>("BEAM_ENERGY", type, nEle, nMeas, beamEnergy);
  if (nMeas == 0) {
    LOG(warn) << "Beam energy not present/empty"; // RS: in absence of the beam it is empty, don't produce an error
  }
  if (nEle > 1 || nMeas > 1) {
    LOGP(warn, "More than one value/measurement {}/{} found for Beam Energy, keeping the last one", nEle, nMeas);
  }
  if (!beamEnergy.empty()) {
    lhcifdata.setBeamEnergyPerZWithTime(beamEnergy.back().first, beamEnergy.back().second.back());
  }
  // Injection scheme
  mReader.readValue<std::string>("INJECTION_SCHEME", type, nEle, nMeas, injScheme);
  if (nMeas == 0) {
    LOG(warn) << "Injection scheme not present/empty"; // RS: same comment
  }
  if (nEle > 1 || nMeas > 1) {
    LOGP(warn, "More than one value/measurement {}/{} found for Injection Scheme, keeping the last one", nEle, nMeas);
  }
  if (!injScheme.empty()) {
    lhcifdata.setInjectionSchemeWithTime(injScheme.back().first, injScheme.back().second.back());
  }

  // fill number
  mReader.readValue<std::string>("FILL_NUMBER", type, nEle, nMeas, fillNb);
  if (nMeas == 0) {
    LOG(warn) << "Fill Number not present/empty";
  }
  if (nEle > 1 || nMeas > 1) {
    LOG(warn) << "More than one value/measurement found for Fill Number, keeping the last one";
  }
  if (!fillNb.empty()) {
    lhcifdata.setFillNumberWithTime(fillNb.back().first, atoi(fillNb.back().second.back().c_str()));
  }

  // Atomic Number (Z) for B1
  mReader.readValue<int32_t>("ATOMIC_NUMBER_B1", type, nEle, nMeas, atomicNbB1);
  if (nMeas == 0) {
    LOG(warn) << "Atomic Number Beam 1 (clockwise) not present/empty"; // RS same comment
  }
  if (nEle > 1 || nMeas > 1) {
    LOGP(warn, "More than one value/measurement {}/{} found for Atomic Number Beam 1 (clockwise), keeping the last one", nEle, nMeas);
  }
  if (!atomicNbB1.empty()) {
    lhcifdata.setAtomicNumberB1WithTime(atomicNbB1.back().first, atomicNbB1.back().second.back());
  }

  // Atomic Number (Z) for B2
  mReader.readValue<int32_t>("ATOMIC_NUMBER_B2", type, nEle, nMeas, atomicNbB2);
  if (nMeas == 0) {
    LOG(warn) << "Atomic Number Beam 2 (anticlockwise) not present/empty";
  }
  if (nEle > 1 || nMeas > 1) {
    LOGP(warn, "More than one value/measurement {}/{} found for Atomic Number Beam 2 (anticlockwise), keeping the last one", nEle, nMeas);
  }
  if (!atomicNbB2.empty()) {
    lhcifdata.setAtomicNumberB2WithTime(atomicNbB2.back().first, atomicNbB2.back().second.back());
  }

  // Crossing Angle
  mReader.readValue<float>("IP2_XING_V_MURAD", type, nEle, nMeas, crossAngle);
  if (nMeas == 0) {
    LOG(warn) << "Crossing Angle not present/empty";
  }
  if (nEle > 1 || nMeas > 1) {
    LOGP(warn, "More than one value/measurement {}/{} found for Crossing Angle, keeping the last one", nEle, nMeas);
  }
  if (!crossAngle.empty()) {
    lhcifdata.setCrossingAngleWithTime(crossAngle.back().first, crossAngle.back().second.back());
  }

  // Bunch Config for B1
  mReader.readValue<int>("CIRCULATING_BUNCH_CONFIG_BEAM1", type, nEle, nMeas, bunchConfigB1);
  if (nMeas == 0) {
    LOG(warn) << "Bunch Config Beam 1 not present/empty";
  }
  if (nMeas > 1) {
    LOGP(warn, "More than one measurement {} found for Bunch Config Beam 1, keeping the last one", nMeas);
  }

  // Bunch Config for B2
  mReader.readValue<int>("CIRCULATING_BUNCH_CONFIG_BEAM2", type, nEle, nMeas, bunchConfigB2);
  if (nMeas == 0) {
    LOG(warn) << "Bunch Config Beam 2 not present/empty";
  }
  if (nMeas > 1) {
    LOGP(warn, "More than one measurement {} found for Bunch Config Beam 2, keeping the last one", nMeas);
  }

  // Building Bunch Filling
  if (!bunchConfigB1.empty() && !bunchConfigB2.empty()) {
    o2::BunchFilling bunchFilling;
    bunchFilling.buckets2BeamPattern(bunchConfigB1.back().second, 0);
    bunchFilling.buckets2BeamPattern(bunchConfigB2.back().second, 1);
    bunchFilling.setInteractingBCsFromBeams();
    lhcifdata.setBunchFillingWithTime((bunchConfigB1.back().first + bunchConfigB2.back().first) / 2, bunchFilling);
  }

  if (mVerbose) {
    LOG(info) << " **** Beam Energy ****";
    for (auto& el : beamEnergy) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(info) << " **** Injection Scheme ****";
    for (auto& el : injScheme) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(info) << " **** Fill Number ****";
    for (auto& el : fillNb) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(info) << " **** Atomic Number Beam 1 (clockwise) ****";
    for (auto& el : atomicNbB1) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(info) << " **** Atomic Number B2 (anticlockwise) ****";
    for (auto& el : atomicNbB2) {
      for (auto elVect : el.second) {
        std::cout << el.first << " --> " << elVect << std::endl;
      }
    }
    LOG(info) << " **** Crossing Angle ****";
    for (auto& el : crossAngle) {
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

void GRPLHCIFfileProcessor::sendOutput(DataAllocator& output, long start, const GRPLHCIFData& lhcifdata)
{
  // sending output to CCDB

  using clbUtils = o2::calibration::Utils;
  auto clName = o2::utils::MemFileHelper::getClassName(lhcifdata);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> md;
  md.emplace("created_by", "dpl");
  if (lhcifdata.getFillNumberTime()) {
    md.emplace("fillNumber", fmt::format("{}", lhcifdata.getFillNumber()));
  }
  o2::ccdb::CcdbObjectInfo info("GLO/Config/GRPLHCIF", clName, flName, md, start, start + o2::ccdb::CcdbObjectInfo::MONTH);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&lhcifdata, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
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
