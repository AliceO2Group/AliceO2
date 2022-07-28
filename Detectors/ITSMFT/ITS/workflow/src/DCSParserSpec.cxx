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

/// @file   DCSParserSpec.cxx

#include "ITSWorkflow/DCSParserSpec.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace its
{

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSDCSParser::ITSDCSParser()
{
  this->mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSDCSParser");
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::init(InitContext& ic)
{
  LOGF(info, "ITSDCSParser init...", mSelfName);

  this->mCcdbUrl = ic.options().get<std::string>("ccdb-url");

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get DCSconfigObject_t from EPNs and aggregate them in 1 object
void ITSDCSParser::run(ProcessingContext& pc)
{
  // check O2/Detectors/DCS/testWorkflow/src/dcs-config-proxy.cxx for input format

  // Retrieve string from inputs
  const auto inString = pc.inputs().get<gsl::span<char>>("inString");
  std::string inStringConv;
  std::copy(inString.begin(), inString.end(), std::back_inserter(inStringConv));
  for (const std::string& line : this->vectorizeStringList(inStringConv, "\n")) {
    if (!line.length()) {
      continue;
    }
    this->updateMemoryFromInputString(line);
    this->saveToOutput();
    this->resetMemory();
  }

  if (this->mConfigDCS.size()) {
    this->pushToCCDB(pc);
    this->mConfigDCS.clear();
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::updateMemoryFromInputString(const std::string& inString)
{
  // Parse the individual parts of the string
  const std::string delimiter = "|";
  unsigned int pos = 0;
  unsigned int npos = inString.find(delimiter);

  // First is the stave name
  this->mStaveName = inString.substr(pos, npos);

  // Next is the run number
  std::string word = "RUN";
  pos += npos + delimiter.length() + word.length();
  npos = inString.find(delimiter, pos) - pos;
  this->updateAndCheck(this->mRunNumber, std::stoi(inString.substr(pos, npos)));

  // Next is the config
  word = "CONFIG";
  pos += npos + delimiter.length() + word.length();
  npos = inString.find(delimiter, pos) - pos;
  this->updateAndCheck(this->mConfigVersion, std::stoi(inString.substr(pos, npos)));

  // Then it's the run type
  word = "RUNTYPE";
  pos += npos + delimiter.length() + word.length();
  npos = inString.find(delimiter, pos) - pos;
  this->updateAndCheck(this->mRunType, std::stoi(inString.substr(pos, npos)));

  // Then there is a semicolon-delineated list of disabled chips
  word = "DISABLED_CHIPS";
  pos += npos + delimiter.length() + word.length();
  npos = inString.find(delimiter, pos) - pos;
  std::string disabledChipsStr = inString.substr(pos, npos);
  if (disabledChipsStr.length()) {
    this->mDisabledChips = this->vectorizeStringListInt(disabledChipsStr, ";");
  }

  // Then there is a 2D list of masked double-columns
  word = "MASKED_DCOLS";
  pos += npos + delimiter.length() + word.length();
  npos = inString.find(delimiter, pos) - pos;
  std::string maskedDoubleColsStr = inString.substr(pos, npos);
  if (maskedDoubleColsStr.length()) {
    std::vector<std::string> chipVect = this->vectorizeStringList(maskedDoubleColsStr, ";");
    for (const std::string& str : chipVect) {
      // Element 0 in each subvector is chip ID, rest are double column numbers
      this->mMaskedDoubleCols.push_back(this->vectorizeStringListInt(str, ":"));
    }
  }

  // Finally, there are double columns which are disabled at EOR
  word = "DCOLS_EOR";
  pos += npos + delimiter.length() + word.length();
  std::string doubleColsEORstr = inString.substr(pos);
  if (doubleColsEORstr.length()) {
    std::vector<std::string> bigVect = this->vectorizeStringList(doubleColsEORstr, "&");
    for (const std::string& bigStr : bigVect) {
      std::vector<std::string> bigVectSplit = this->vectorizeStringList(bigStr, "|");
      if (!bigVectSplit.size()) {
        continue;
      }

      // First, update map of disabled double columns at EOR
      std::vector<std::string> doubleColDisable = this->vectorizeStringList(bigVectSplit[0], ";");
      for (const std::string& str : doubleColDisable) {
        std::vector<unsigned short int> doubleColDisableVector = this->vectorizeStringListInt(str, ":");
        this->mDoubleColsDisableEOR[doubleColDisableVector[0]].push_back(doubleColDisableVector[1]);
      }
      // Second, update map of flagged pixels at EOR
      std::vector<std::string> pixelFlagsEOR = this->vectorizeStringList(bigVectSplit[1], ";");
      for (const std::string& str : pixelFlagsEOR) {
        std::vector<unsigned short int> pixelFlagsVector = this->vectorizeStringListInt(str, ":");
        this->mPixelFlagsEOR[pixelFlagsVector[0]].push_back(pixelFlagsVector[1]);
      }
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Update memValue using newValue if memValue is not yet set
void ITSDCSParser::updateAndCheck(int& memValue, const int newValue)
{
  // Check if value in memory is nonsense, meaning it has never been updated
  if (memValue == UNSET_INT) {
    // Save value in memory for the first time (should always be the same)
    memValue = newValue;
  } else if (memValue != newValue) {
    // Different value received than the one saved in memory -- throw error
    throw newValue;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::updateAndCheck(short int& memValue, const short int newValue)
{
  // Check if value in memory is nonsense, meaning it has never been updated
  if (memValue == UNSET_SHORT) {
    // Save value in memory for the first time (should always be the same)
    memValue = newValue;
  } else if (memValue != newValue) {
    // Different value received than the one saved in memory -- throw error
    throw newValue;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Take string delimited by delimiter and parse into vector of objects
std::vector<std::string> ITSDCSParser::vectorizeStringList(
  const std::string& str, const std::string& delimiter)
{
  std::vector<std::string> str_vect;
  std::string::size_type prev_pos = 0, pos = 0;
  while ((pos = str.find(delimiter, pos)) != std::string::npos) {
    std::string substr = str.substr(prev_pos, pos - prev_pos);
    if (substr.length()) {
      str_vect.push_back(substr);
    }
    pos += delimiter.length();
    prev_pos = pos;
  }
  std::string substr = str.substr(prev_pos, pos - prev_pos);
  if (substr.length()) {
    str_vect.push_back(substr);
  }
  return str_vect;
}

//////////////////////////////////////////////////////////////////////////////
std::vector<unsigned short int> ITSDCSParser::vectorizeStringListInt(
  const std::string& str, const std::string& delimiter)
{
  std::vector<unsigned short int> int_vect;
  std::string::size_type prev_pos = 0, pos = 0;
  while ((pos = str.find(delimiter, pos)) != std::string::npos) {
    unsigned short int substr = std::stoi(str.substr(prev_pos, pos - prev_pos));
    int_vect.push_back(substr);
    pos += delimiter.length();
    prev_pos = pos;
  }
  int_vect.push_back(std::stoi(str.substr(prev_pos, pos - prev_pos)));
  return int_vect;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::saveToOutput()
{
  // First loop through the disabled chips to write these to the string
  for (const unsigned short int& chipID : this->mDisabledChips) {
    // Write basic chip info
    this->writeChipInfo(this->mConfigDCS, this->mStaveName, chipID);

    // Mark chip as disabled
    o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "1");

    // Mark other information with nonsense value
    o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked_eor", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Pixel_flags", "-1");

    // Ensure that chips are removed from the maps
    mDoubleColsDisableEOR.erase(chipID);
    mPixelFlagsEOR.erase(chipID);
  }

  // Second, loop through all the chips with disabled double columns
  for (std::vector<unsigned short int> maskedDoubleCols : this->mMaskedDoubleCols) {
    unsigned short int chipID = maskedDoubleCols[0];
    maskedDoubleCols.erase(maskedDoubleCols.begin());

    // Write basic chip info
    this->writeChipInfo(this->mConfigDCS, this->mStaveName, chipID);
    o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "0");

    // Write information for disabled double columns
    o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", this->intVecToStr(maskedDoubleCols, "|"));

    // Retrieve information from maps, if any, and then erase
    o2::dcs::addConfigItem(
      this->mConfigDCS, "Dcol_masked_eor", this->intVecToStr(this->mDoubleColsDisableEOR[chipID], "|"));
    this->mDoubleColsDisableEOR.erase(chipID);
    o2::dcs::addConfigItem(
      this->mConfigDCS, "Pixel_flags", this->intVecToStr(this->mPixelFlagsEOR[chipID], "|"));
    this->mPixelFlagsEOR.erase(chipID);
  }

  // Finally, loop through any remaining chips
  for (const auto& [chipID, v] : this->mDoubleColsDisableEOR) {
    std::string s = this->intVecToStr(v, "|");
    if (s != "-1") { // Ensure no meaningless entries
      this->writeChipInfo(this->mConfigDCS, this->mStaveName, chipID);
      o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "0");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", "-1");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked_eor", this->intVecToStr(v, "|"));
      o2::dcs::addConfigItem(
        this->mConfigDCS, "Pixel_flags", this->intVecToStr(this->mPixelFlagsEOR[chipID], "|"));
      this->mPixelFlagsEOR.erase(chipID);
    }
  }

  for (const auto& [chipID, v] : this->mPixelFlagsEOR) {
    std::string s = this->intVecToStr(v, "|");
    if (s != "-1") { // Ensure no meaningless entries
      this->writeChipInfo(this->mConfigDCS, this->mStaveName, chipID);
      o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "0");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", "-1");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked_eor", "-1");
      o2::dcs::addConfigItem(this->mConfigDCS, "Pixel_flags", this->intVecToStr(v, "|"));
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Reset memory for reading another line from input string
void ITSDCSParser::resetMemory()
{
  mStaveName = "";
  mDisabledChips.clear();
  mMaskedDoubleCols.clear();
  mDoubleColsDisableEOR.clear();
  mPixelFlagsEOR.clear();
  return;
}

//////////////////////////////////////////////////////////////////////////////
// Convert vector of integers to one single string, delineated by delin
std::string ITSDCSParser::intVecToStr(
  const std::vector<unsigned short int>& v, const std::string& delin)
{
  // Return nonsense value if there are no entries
  if (v.empty()) {
    return "-1";
  }

  std::string bigStr = std::to_string(v[0]);
  for (int i = 1; i < v.size(); i++) {
    bigStr += delin + std::to_string(v[i]);
  }

  return bigStr;
}

//////////////////////////////////////////////////////////////////////////////
// Write the Stave, Hs_pos, Hic_pos, and ChipID to the config object
void ITSDCSParser::writeChipInfo(
  o2::dcs::DCSconfigObject_t& configObj, const std::string& staveName,
  const unsigned short int chipID)
{
  // First save the Stave to the string
  o2::dcs::addConfigItem(configObj, "Stave", staveName);

  // Hs_pos: 0 is lower, 1 is upper
  // Hic_pos: from 1 to 7 (module number)
  unsigned short int hicPos = chipID / 16;
  bool hsPos = (hicPos > 7);
  if (hsPos) {
    hicPos -= 8;
  }
  o2::dcs::addConfigItem(configObj, "Hs_pos", std::to_string(hsPos));
  o2::dcs::addConfigItem(configObj, "Hic_pos", std::to_string(hicPos));

  // Chip ID inside the module
  o2::dcs::addConfigItem(configObj, "ChipID", std::to_string(chipID % 16));

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::pushToCCDB(ProcessingContext& pc)
{
  // Timestamps for CCDB entry
  long tstart = o2::ccdb::getCurrentTimestamp();
  long tend = tstart + 365L * 24 * 3600 * 1000;

  auto class_name = o2::utils::MemFileHelper::getClassName(mConfigDCS);

  // Create metadata for database object
  std::map<std::string, std::string> md = {
    {"runtype", std::to_string(this->mRunType)}, {"confDBversion", std::to_string(this->mConfigVersion)}};
  if (!mCcdbUrl.empty()) { // add only if we write here otherwise ccdb-populator-wf add it already
    md.insert({"runNumber", std::to_string(this->mRunNumber)});
  }
  std::string path("ITS/DCS_CONFIG/");
  const char* filename = "dcs_config.root";
  o2::ccdb::CcdbObjectInfo info(path, "dcs_config", filename, md, tstart, tend);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&mConfigDCS, &info);
  info.setFileName(filename);

  // Send to ccdb-populator wf
  LOG(info) << "Class Name: " << class_name << " | File Name: " << filename
            << "\nSending to ccdb-populator the object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size() << " bytes, valid for "
            << info.getStartValidityTimestamp() << " : "
            << info.getEndValidityTimestamp();

  if (mCcdbUrl.empty()) {

    pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "DCS_CONFIG", 0}, *image);
    pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "DCS_CONFIG", 0}, info);

  } else { // if url is specified, send object to ccdb from THIS wf

    LOG(info) << "Sending object " << info.getFileName() << " to " << mCcdbUrl << "/browse/"
              << info.getPath() << " from the ITS string parser workflow";
    o2::ccdb::CcdbApi mApi;
    mApi.init(mCcdbUrl);
    mApi.storeAsBinaryFile(
      &image->at(0), image->size(), info.getFileName(), info.getObjectType(), info.getPath(),
      info.getMetaData(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSDCSParserSpec()
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("inString", detOrig, "DCS_CONFIG_FILE", 0, Lifetime::Timeframe);
  inputs.emplace_back("nameString", detOrig, "DCS_CONFIG_NAME", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "DCS_CONFIG"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "DCS_CONFIG"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "its-parser",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::its::ITSDCSParser>()},
    Options{
      {"ccdb-url", VariantType::String, "", {"CCDB url, default is empty (i.e. send output to CCDB populator workflow)"}}}};
}
} // namespace its
} // namespace o2
