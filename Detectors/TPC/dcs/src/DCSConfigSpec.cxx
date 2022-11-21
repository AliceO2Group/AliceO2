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
#include <sstream>
#include <string>
#include <string_view>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/StringUtils.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/NameConf.h"

#include "TPCBase/CDBInterface.h"
#include "TPCBase/CRUCalibHelpers.h"
#include "TPCBase/FEEConfig.h"
#include "TPCBase/FECInfo.h"

#include "TPCdcs/DCSConfigSpec.h"

namespace fs = std::filesystem;
using namespace o2::utils;
using namespace o2::framework;
constexpr auto CDBPayload = o2::calibration::Utils::gDataOriginCDBPayload;
constexpr auto CDBWrapper = o2::calibration::Utils::gDataOriginCDBWrapper;

namespace o2::tpc
{

const std::unordered_map<CDBType, o2::header::DataDescription> CDBDescMap{
  {CDBType::ConfigFEE, o2::header::DataDescription{"TPC_FEEConfig"}},
};

class DCSConfigDevice : public o2::framework::Task
{
 public:
  struct TagInfo {
    TagInfo(int t, std::string_view c)
    {
      tag = t;
      comment = c.data();
    }

    int tag;
    std::string comment;
  };
  using TagInfos = std::vector<TagInfo>;

  DCSConfigDevice() = default;

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void writeRootFile(int tag);

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
  }

  void stop() final
  {
    LOGP(info, "stop");
  }

  static TagInfos readTagInfos(gsl::span<const char> configBuff);

 private:
  struct membuf : std::streambuf {
    membuf(char* base, std::ptrdiff_t n)
    {
      this->setg(base, base, base + n);
    }
  };

  /// Names of calDet objects from CRU
  static constexpr std::string_view CalDetNames = "ITfraction,ITexpLambda,ThresholdMap,Pedestals,CMkValues";
  static constexpr int NCalDets = 5; ///< number of expected calDet objects

  static constexpr std::string_view CRUConfigName = "CRUdata";
  static constexpr std::string_view TagInfoName = "TagInfo";
  static constexpr std::string_view RunInfoFileName = "TPCRunInfo.txt";
  static constexpr int NLinksTotal = 2 * FECInfo::FECsTotal;

  CDBStorage mCDBStorage;
  o2::ccdb::CcdbApi mCCDBApi;
  FEEConfig mFEEConfig;
  std::unordered_map<std::string, std::vector<char>> mDataBuffer; ///< buffer for received data
  std::unordered_map<std::string, long> mDataBufferTimes;         ///< buffer for received data times
  std::bitset<NCalDets> mFEEPadDataReceived;                      ///< check if all pad-wise data was received
  bool mCRUConfigReceived = false;                                ///< if CRU config was received
  bool mDumpToRootFile = false;                                   ///< optional filename to dump data to
  bool mDumpToTextFiles = false;                                  ///< dump received data to text files
  bool mReadFromRootFile = false;                                 ///< read from root file instead of ccdb for testing
  bool mDontWriteRunInfo = false;                                 ///< read from root file instead of ccdb for testing

  void updateRunInfo(gsl::span<const char> configBuff);
  void fillFEEPad(std::string_view configFileName, gsl::span<const char> configBuff, bool update);
  void fillCRUConfig(gsl::span<const char> configBuff, bool update);
  void dumpToTextFile(std::string_view configFileName, gsl::span<const char> configBuff, std::string_view tagNames);
  bool loadFEEConfig(int tag);
  void updateTags(DataAllocator& output);
  void updateCCDB(DataAllocator& output, const TagInfo& tagInfo);
};

void DCSConfigDevice::init(o2::framework::InitContext& ic)
{
  mDumpToTextFiles = ic.options().get<bool>("dump-to-text-files");
  mDumpToRootFile = ic.options().get<bool>("dump-to-root-file");
  mReadFromRootFile = ic.options().get<bool>("read-from-root-file");
  mDontWriteRunInfo = ic.options().get<bool>("dont-write-run-info");

  const auto calDetNamesVec = Str::tokenize(CalDetNames.data(), ',');
  assert(NCalDets == calDetNamesVec.size());

  for (const auto& name : calDetNamesVec) {
    mFEEConfig.padMaps[name.data()].setName(name.data());
  }

  mCCDBApi.init(o2::base::NameConf::getCCDBServer());
  // set default meta data
  mCDBStorage.setResponsible("Jens Wiechula (jens.wiechula@cern.ch)");
  mCDBStorage.setIntervention(CDBIntervention::Automatic);
}

void DCSConfigDevice::run(o2::framework::ProcessingContext& pc)
{
  auto configBuff = pc.inputs().get<gsl::span<char>>("inputConfig");
  auto configFileName = pc.inputs().get<std::string>("inputConfigFileName");
  const auto creation = pc.services().get<o2::framework::TimingInfo>().creation;
  LOG(info) << "received input file " << configFileName << " of size " << configBuff.size();

  // either we receive a run info file or an update of a FEE config tag
  if (configFileName == RunInfoFileName) {
    updateRunInfo(configBuff);
  } else { // update tag
    std::string objName = fs::path(configFileName).stem().c_str();
    objName = objName.substr(3, objName.size());
    // first buffer all the datao
    auto& dataVec = mDataBuffer[objName];
    auto& objTime = mDataBufferTimes[objName];
    if (dataVec.size()) {
      LOGP(warning, "Another object with name {} was already received before: {}, now: {}, old object will be overwritten", objName, objTime, creation);
      dataVec.clear();
    }
    dataVec.insert(dataVec.begin(), configBuff.begin(), configBuff.end());
    objTime = creation;

    if (objName == TagInfoName) {
      updateTags(pc.outputs());
    }
  }
}

void DCSConfigDevice::updateRunInfo(gsl::span<const char> configBuff)
{
  const std::string line(configBuff.data(), configBuff.size());
  const auto data = Str::tokenize(line, ',');
  const std::string_view runInfoConf("Run number;SOX;Tag number;Run type");
  if (data.size() != 4) {
    LOGP(error, "{} has wrong format: {}, expected: {}, not writing RunInformation to CCDB", RunInfoFileName, line, runInfoConf);
    return;
  }
  char tempChar{};
  std::map<std::string, std::string> md;
  md[o2::base::NameConf::CCDBRunTag.data()] = data[0];
  md["Tag"] = data[2];
  md["RunType"] = data[3];
  md[o2::ccdb::CcdbObjectInfo::AdjustableEOV] = "true";

  const long startValRCT = std::stol(data[1]);
  const long endValRCT = startValRCT + 48l * 60l * 60l * 1000l;
  if (!mDontWriteRunInfo) {
    mCCDBApi.storeAsBinaryFile(&tempChar, sizeof(tempChar), "tmp.dat", "char", CDBTypeMap.at(CDBType::ConfigRunInfo), md, startValRCT, endValRCT);
  }

  std::string mdInfo = "[";
  for (const auto& [key, val] : md) {
    mdInfo += fmt::format("{} = {}, ", key, val);
  }
  mdInfo += "]";
  LOGP(info, "Updated {} with {} for validity range {}, {}", CDBTypeMap.at(CDBType::ConfigRunInfo), mdInfo, startValRCT, endValRCT);
}

void DCSConfigDevice::updateCCDB(DataAllocator& output, const TagInfo& tagInfo)
{
  mCDBStorage.setReason(tagInfo.comment);
  std::map<std::string, std::string> md = mCDBStorage.getMetaData();
  o2::ccdb::CcdbObjectInfo w(false);
  o2::calibration::Utils::prepareCCDBobjectInfo(mFEEConfig, w, CDBTypeMap.at(CDBType::ConfigFEE), md, tagInfo.tag, tagInfo.tag + 1);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&mFEEConfig, &w);

  LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", w.getPath(), w.getFileName(), image->size(), w.getStartValidityTimestamp(), w.getEndValidityTimestamp());
  output.snapshot(Output{CDBPayload, CDBDescMap.at(CDBType::ConfigFEE), 0}, *image.get());
  output.snapshot(Output{CDBWrapper, CDBDescMap.at(CDBType::ConfigFEE), 0}, w);

  mFEEPadDataReceived.reset();
  mCRUConfigReceived = false;

  if (mDumpToRootFile) {
    writeRootFile(tagInfo.tag);
  }

  mFEEConfig.clear();
}

void DCSConfigDevice::fillFEEPad(std::string_view configFileName, gsl::span<const char> configBuff, bool update)
{
  auto& calPad = mFEEConfig.padMaps[configFileName.data()];
  int nLines = 0;
  if (configFileName == "ITfraction") {
    nLines = cru_calib_helpers::fillCalPad<0>(calPad, configBuff);
    mFEEPadDataReceived.set(0);
  } else if (configFileName == "ITexpLambda") {
    nLines = cru_calib_helpers::fillCalPad<0>(calPad, configBuff);
    mFEEPadDataReceived.set(1);
  } else if (configFileName == "ThresholdMap") {
    nLines = cru_calib_helpers::fillCalPad<2>(calPad, configBuff);
    mFEEPadDataReceived.set(2);
  } else if (configFileName == "Pedestals") {
    nLines = cru_calib_helpers::fillCalPad<2>(calPad, configBuff);
    mFEEPadDataReceived.set(3);
  } else if (configFileName == "CMkValues") {
    nLines = cru_calib_helpers::fillCalPad<6>(calPad, configBuff);
    mFEEPadDataReceived.set(4);
  }

  if (!update && (nLines != NLinksTotal)) {
    LOGP(error, "Full FEEConfig expected, but only {} / {} lines read for object {}", nLines, NLinksTotal, configFileName);
  } else {
    LOGP(info, "updating CalDet object {} for {} links", configFileName, nLines);
  }
}

void DCSConfigDevice::fillCRUConfig(gsl::span<const char> configBuff, bool update)
{
  membuf sbuf((char*)configBuff.data(), configBuff.size());
  std::istream in(&sbuf);
  std::string line;

  int nLines = 0;
  while (std::getline(in, line)) {
    const auto cru = std::stoi(line.substr(0, line.find_first_of(' ')));
    if ((cru < 0) || (cru >= CRU::MaxCRU)) {
      LOGP(error, "unexpected CRU number {} in line {}", cru, line);
      continue;
    }

    const std::string cruData(line.substr(line.find_first_of(' ') + 1, line.size()));

    auto& cruConfig = mFEEConfig.cruConfig[cru];
    if (cruConfig.setValues(cruData)) {
      ++nLines;
    }
  }

  if (!update && (nLines != CRU::MaxCRU)) {
    LOGP(error, "Full FEEConfig expected, but only {} / {} lines read for CRUConfig", nLines, CRU::MaxCRU);
  } else {
    LOGP(info, "updating CRUConfig for {} crus", nLines);
  }
  mCRUConfigReceived = true;
}

void DCSConfigDevice::dumpToTextFile(std::string_view objName, gsl::span<const char> configBuff, std::string_view tagNames)
{
  const auto now = std::chrono::system_clock::now();
  const long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  const auto outputFileName = fmt::format("TPC{}.{}.{}.txt", objName, tagNames, timeStart);

  std::ofstream outFile(outputFileName);
  outFile.write(configBuff.data(), configBuff.size());
  outFile.close();
}

void DCSConfigDevice::updateTags(DataAllocator& outputs)
{
  const auto tagInfos = readTagInfos(mDataBuffer.at(TagInfoName.data()));
  std::string tagNames;

  for (const auto& tagInfo : tagInfos) {
    if (tagNames.size()) {
      tagNames += "_";
    }
    tagNames += std::to_string(tagInfo.tag);
    LOGP(info, "Updating TPC FEE config for tag {}", tagInfo.tag);
    const auto update = loadFEEConfig(tagInfo.tag);

    for (const auto& [objName, configBuff] : mDataBuffer) {
      if (CalDetNames.find(objName) != std::string_view::npos) {
        fillFEEPad(objName, configBuff, update);
      } else if (objName == CRUConfigName) {
        fillCRUConfig(configBuff, update);
      }
    }
    if (!update && ((mFEEPadDataReceived.count() != NCalDets) || !mCRUConfigReceived)) {
      std::string errorMessage;
      if (mFEEPadDataReceived.count() != NCalDets) {
        errorMessage = fmt::format("not all CalDet objects received: {} ({})", mFEEPadDataReceived.to_string(), CalDetNames);
      }
      if (!mCRUConfigReceived) {
        if (errorMessage.size()) {
          errorMessage += " and ";
        }
        errorMessage += "CRUConfig not reveived";
      }

      LOGP(error, "Full FEEConfig expected, but {}", errorMessage);
    }
    updateCCDB(outputs, tagInfo);
  }

  if (mDumpToTextFiles) {
    for (const auto& [objName, configBuff] : mDataBuffer) {
      dumpToTextFile(objName, configBuff, tagNames);
    }
  }

  mDataBuffer.clear();
}

void DCSConfigDevice::writeRootFile(int tag)
{
  const auto now = std::chrono::system_clock::now();
  const long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  const auto outputFileName = fmt::format("TPCFEEConfig.{}.{}.root", tag, timeStart);
  std::unique_ptr<TFile> outFile(TFile::Open(outputFileName.data(), "recreate"));
  outFile->WriteObject(&mFEEConfig, "FEEConfig");
}

DCSConfigDevice::TagInfos DCSConfigDevice::readTagInfos(gsl::span<const char> configBuff)
{

  membuf sbuf((char*)configBuff.data(), configBuff.size());
  std::istream in(&sbuf);
  std::string line;

  TagInfos infos;
  while (std::getline(in, line)) {
    const auto firstComma = line.find_first_of(',');
    int tag = -1;
    try {
      tag = std::stoi(std::string(line.substr(0, firstComma)));
    } catch (...) {
    }
    const std::string comment(line.substr(firstComma + 1, line.size()));
    if ((tag < 0) || comment.empty()) {
      LOGP(warning, "Ill formatted line in 'TPC{}.txt': {}, expecting <tag,comment>, skipping", TagInfoName, line);
      continue;
    }
    infos.emplace_back(tag, comment);
  }

  return infos;
}

bool DCSConfigDevice::loadFEEConfig(int tag)
{
  FEEConfig* feeConfig = nullptr;
  if (mReadFromRootFile) {
    // find last file of a tag in the local directory, assuming
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator("./")) {
      const auto fileName = entry.path().filename().string();
      // chek if root file and contains tag number
      if (entry.is_regular_file() &&
          fileName.find(".root") == (fileName.size() - 5) &&
          fileName.find(fmt::format(".{}.", tag)) != std::string::npos) {
        filenames.emplace_back(fileName);
      }
    }
    if (filenames.size()) {
      std::sort(filenames.begin(), filenames.end());
      const auto& lastFile = filenames.back();
      std::unique_ptr<TFile> inFile(TFile::Open(lastFile.data()));
      if (inFile->IsOpen() && !inFile->IsZombie()) {
        inFile->GetObject("FEEConfig", feeConfig);
      }
      if (feeConfig) {
        LOGP(info, "Read FEEConfig from file {} for updating", lastFile);
      }
    }
  } else {
    std::map<std::string, std::string> headers;
    feeConfig = mCCDBApi.retrieveFromTFileAny<FEEConfig>(CDBTypeMap.at(CDBType::ConfigFEE), std::map<std::string, std::string>(), tag, &headers);
    if (feeConfig) {
      LOGP(info, "Read FEEConfig from ccdb for updating: URL: {}, Validity: {} - {}, Last modfied: {}, ETag: {}", mCCDBApi.getURL(), headers["Valid-From"], headers["Valid-Until"], headers["Last-Modified"], headers["ETag"]);
    }
  }

  if (!feeConfig) {
    LOGP(warning, "Could not retrieve FEE config for tag {}, a new entry will be created, full config should be sent", tag);
    mFEEConfig.clear();
    return false;
  }
  mFEEConfig = *feeConfig;
  if (mReadFromRootFile) {
    delete feeConfig;
  }
  return true;
}

DataProcessorSpec getDCSConfigSpec()
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, CDBDescMap.at(CDBType::ConfigFEE)}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, CDBDescMap.at(CDBType::ConfigFEE)}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-dcs-config",
    Inputs{{"inputConfig", o2::header::gDataOriginTPC, "DCS_CONFIG_FILE", Lifetime::Timeframe},
           {"inputConfigFileName", o2::header::gDataOriginTPC, "DCS_CONFIG_NAME", Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<DCSConfigDevice>()},
    Options{
      {"dump-to-text-files", VariantType::Bool, false, {"Optionally dump received data to text files"}},
      {"dump-to-root-file", VariantType::Bool, false, {"Optionally write FEEConfig to root file"}},
      {"read-from-root-file", VariantType::Bool, false, {"For testing purposes: read configuration from local root file instead of ccdb"}},
      {"dont-write-run-info", VariantType::Bool, false, {"For testing purposes: skip writing RunInformation to ccdb"}},
    }};
}

} // namespace o2::tpc
