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

/// @file   DCSConfigSpec.h
/// @author Jens Wiechula
/// @brief  DCS configuration processing

#include <memory>
#include <cassert>
#include <chrono>
#include <bitset>
#include <fstream>
#include <string_view>
#include <filesystem>

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsCalibration/Utils.h"

#include "TPCBase/CDBInterface.h"
#include "TPCBase/Utils.h"
#include "TPCBase/CRUCalibHelpers.h"

#include "TPCdcs/DCSConfigSpec.h"

namespace fs = std::filesystem;
using namespace o2::framework;
constexpr auto CDBPayload = o2::calibration::Utils::gDataOriginCDBPayload;
constexpr auto CDBWrapper = o2::calibration::Utils::gDataOriginCDBWrapper;

namespace o2::tpc
{

const std::unordered_map<CDBType, o2::header::DataDescription> CDBDescMap{
  {CDBType::ConfigFEEPad, o2::header::DataDescription{"TPC_FEEPad"}},
};

class DCSConfigDevice : public o2::framework::Task
{
 public:
  DCSConfigDevice() = default;

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  template <typename T>
  void sendObject(DataAllocator& output, T& obj, const CDBType calibType, const long start, const long end);

  void updateCCDB(DataAllocator& output);

  void writeRootFile();

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
  }

  void stop() final
  {
    LOGP(info, "stop");
  }

 private:
  /// Names of calDet objects from CRU
  static constexpr std::string_view CalDetNames = "ITfraction,ITexpLambda,ThresholdMap,Pedestals,CMkValues";
  static constexpr int NCalDets = 5; ///< number of expected calDet objects

  CDBStorage mCDBStorage;
  CDBInterface::CalPadMapType mFEEPadData;   ///< pad-wise calibration data used in CRU
  std::bitset<NCalDets> mFEEPadDataReceived; ///< check if all pad-wise data was received
  std::string mOutputFileName;               ///< optional filename to dump data to
  bool mDumpToTextFiles = false;             ///< dump received data to text files

  void fillFEEPad(std::string_view configFileName, gsl::span<const char> configBuff);
  void dumpToTextFile(std::string_view configFileName, gsl::span<const char> configBuff);
};

void DCSConfigDevice::init(o2::framework::InitContext& ic)
{
  mDumpToTextFiles = ic.options().get<bool>("dump-to-text-files");
  mOutputFileName = ic.options().get<std::string>("dump-to-root-file");
  const auto calDetNamesVec = utils::tokenize(CalDetNames, ",");
  assert(NCalDets == calDetNamesVec);

  for (const auto& name : calDetNamesVec) {
    mFEEPadData[name.data()].setName(name.data());
  }
}

void DCSConfigDevice::run(o2::framework::ProcessingContext& pc)
{
  auto configBuff = pc.inputs().get<gsl::span<char>>("inputConfig");
  auto configFileName = pc.inputs().get<std::string>("inputConfigFileName");
  LOG(info) << "got input file " << configFileName << " of size " << configBuff.size();

  std::string objName = fs::path(configFileName).stem().c_str();
  objName = objName.substr(3, objName.size());
  if (CalDetNames.find(objName) != std::string_view::npos) {
    fillFEEPad(objName, configBuff);
  }

  if (mDumpToTextFiles) {
    dumpToTextFile(configFileName, configBuff);
  }

  if (mFEEPadDataReceived.all()) {
    LOGP(info, "Pad objects {} received", CalDetNames);
    updateCCDB(pc.outputs());
  }
}

template <typename T>
void DCSConfigDevice::sendObject(DataAllocator& output, T& obj, const CDBType calibType, const long start, const long end)
{
  LOGP(info, "Prepare CCDB for {}", CDBTypeMap.at(calibType));

  std::map<std::string, std::string> md = mCDBStorage.getMetaData();
  o2::ccdb::CcdbObjectInfo w;
  o2::calibration::Utils::prepareCCDBobjectInfo(obj, w, CDBTypeMap.at(calibType), md, start, end);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&obj, &w);

  LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", w.getPath(), w.getFileName(), image->size(), w.getStartValidityTimestamp(), w.getEndValidityTimestamp());
  output.snapshot(Output{CDBPayload, CDBDescMap.at(calibType), 0}, *image.get());
  output.snapshot(Output{CDBWrapper, CDBDescMap.at(calibType), 0}, w);
}

void DCSConfigDevice::updateCCDB(DataAllocator& output)
{
  using namespace std::literals::chrono_literals;
  const auto now = std::chrono::system_clock::now();
  const auto end = now + 1h * 24 * 365 * 3; // no + 3 years
  const long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  const long timeEnd = std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch()).count();

  sendObject(output, mFEEPadData, CDBType::ConfigFEEPad, timeStart, timeEnd);

  mFEEPadDataReceived.reset();

  if (mOutputFileName.size()) {
    writeRootFile();
  }
}

void DCSConfigDevice::fillFEEPad(std::string_view configFileName, gsl::span<const char> configBuff)
{
  if (configFileName == "ITfraction") {
    cru_calib_helpers::fillCalPad<0>(mFEEPadData[configFileName.data()], configBuff);
    mFEEPadDataReceived.set(0);
  } else if (configFileName == "ITexpLambda") {
    cru_calib_helpers::fillCalPad<0>(mFEEPadData[configFileName.data()], configBuff);
    mFEEPadDataReceived.set(1);
  } else if (configFileName == "ThresholdMap") {
    cru_calib_helpers::fillCalPad<2>(mFEEPadData[configFileName.data()], configBuff);
    mFEEPadDataReceived.set(2);
  } else if (configFileName == "Pedestals") {
    cru_calib_helpers::fillCalPad<2>(mFEEPadData[configFileName.data()], configBuff);
    mFEEPadDataReceived.set(3);
  } else if (configFileName == "CMkValues") {
    cru_calib_helpers::fillCalPad<6>(mFEEPadData[configFileName.data()], configBuff);
    mFEEPadDataReceived.set(4);
  }
  LOGP(info, "updating CalDet object {} ({})", configFileName, mFEEPadDataReceived.to_string());
}

void DCSConfigDevice::dumpToTextFile(std::string_view configFileName, gsl::span<const char> configBuff)
{
  const auto now = std::chrono::system_clock::now();
  const long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  std::string outputFileName = fs::path(configFileName).stem().c_str();
  outputFileName += fmt::format(".{}.txt", timeStart);

  std::ofstream outFile(outputFileName);
  outFile.write(configBuff.data(), configBuff.size());
  outFile.close();
}

void DCSConfigDevice::writeRootFile()
{
  std::unique_ptr<TFile> outFile(TFile::Open(mOutputFileName.data(), "recreate"));
  outFile->WriteObject(&mFEEPadData, "FEEPad");
}

DataProcessorSpec getDCSConfigSpec()
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_FEEPad"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_FEEPad"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-dcs-config",
    Inputs{{"inputConfig", o2::header::gDataOriginTPC, "DCS_CONFIG_FILE", Lifetime::Timeframe},
           {"inputConfigFileName", o2::header::gDataOriginTPC, "DCS_CONFIG_NAME", Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<DCSConfigDevice>()},
    Options{
      {"dump-to-text-files", VariantType::Bool, false, {"Optionally dump received data to text files"}},
      {"dump-to-root-file", VariantType::String, "", {"Optional root output filename of which to dump data to"}},
    }};
}

} // namespace o2::tpc
