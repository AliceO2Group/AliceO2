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

  this->mCcdbUrlRct = ic.options().get<std::string>("ccdb-url-rct");

  this->mVerboseOutput = ic.options().get<bool>("verbose");

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

  // Check for DOS vs. Unix line ending
  std::string line_ending = "\n";
  size_t newline_pos = inStringConv.find(line_ending);
  if (newline_pos && newline_pos != std::string::npos &&
      inStringConv[newline_pos - 1] == '\r') {
    line_ending = "\r\n";
  }

  // Initialize Dead Chips map
  this->mDeadMap = o2::itsmft::NoiseMap(mp.getNChips());

  // Loop over lines in the input file
  for (const std::string& line : this->vectorizeStringList(inStringConv, line_ending)) {
    if (!line.length()) {
      continue;
    }
    this->updateMemoryFromInputString(line);
    this->appendDeadChipObj();
    this->saveToOutput();
    this->resetMemory();
  }

  this->saveMissingToOutput();

  if (this->mConfigDCS.size() && this->mDeadMap.size()) {
    LOG(info) << "Pushing to CCDB...\n";
    this->pushToCCDB(pc);
    this->mConfigDCS.clear();
  }

  // Reset saved information for the next EOR file
  this->mRunNumber = UNSET_INT;
  this->mConfigVersion = UNSET_INT;
  this->mRunType = UNSET_SHORT;

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::updateMemoryFromInputString(const std::string& inString)
{

  // Print the entire string if verbose mode is requested
  if (this->mVerboseOutput) {
    LOG(info) << "Parsing string: " << inString;
  }

  // Parse the individual parts of the string
  const std::string delimiter = "|";
  const std::string terminator = "!";
  size_t pos = 0;
  size_t npos = inString.find(delimiter);
  if (npos == std::string::npos) {
    LOG(error) << "Delimiter not found, possibly corrupted data!";
    return;
  }

  // First is the stave name
  this->mStaveName = inString.substr(pos, npos);
  this->mSavedStaves.push_back(this->mStaveName);

  // Control the integrity of the string
  this->mTerminationString = true;
  if (inString.back() != '!') {
    this->mTerminationString = false;
    LOG(warning) << "Terminator not found, possibly incomplete data for stave " << this->mStaveName << " !";
  }

  // Next is the run number
  if (!this->updatePosition(pos, npos, delimiter, "RUN", inString)) {
    return;
  }
  this->updateAndCheck(this->mRunNumber, std::stoi(inString.substr(pos, npos)));

  // Next is the config
  if (!this->updatePosition(pos, npos, delimiter, "CONFIG", inString)) {
    return;
  }
  this->updateAndCheck(this->mConfigVersion, std::stoi(inString.substr(pos, npos)));

  // Then it's the run type
  if (!this->updatePosition(pos, npos, delimiter, "RUNTYPE", inString)) {
    return;
  }
  this->updateAndCheck(this->mRunType, std::stoi(inString.substr(pos, npos)));

  // Then there is a semicolon-delineated list of disabled chips
  if (!this->updatePosition(pos, npos, delimiter, "DISABLED_CHIPS", inString)) {
    return;
  }
  std::string disabledChipsStr = inString.substr(pos, npos);
  if (disabledChipsStr.length()) {
    this->mDisabledChips = this->vectorizeStringListInt(disabledChipsStr, ";");
  }

  // Then there is a 2D list of masked double-columns
  if (!this->updatePosition(pos, npos, delimiter, "MASKED_DCOLS", inString)) {
    return;
  }
  std::string maskedDoubleColsStr = inString.substr(pos, npos);
  if (maskedDoubleColsStr.length()) {
    std::vector<std::string> chipVect = this->vectorizeStringList(maskedDoubleColsStr, ";");
    for (const std::string& str : chipVect) {
      // Element 0 in each subvector is chip ID, rest are double column numbers
      this->mMaskedDoubleCols.push_back(this->vectorizeStringListInt(str, ":"));
    }
  }

  // Finally, there are double columns which are disabled at EOR
  if (!this->updatePosition(pos, npos, delimiter, "DCOLS_EOR", inString, true)) {
    return;
  }
  std::string doubleColsEORstr = inString.substr(pos);

  // In this case the terminator is missing and the DCOLS_EOR field is empty:
  // the stave is passed whitout meaningful infos
  if (doubleColsEORstr == "|") {
    this->writeChipInfo(this->mStaveName, -1);
    o2::dcs::addConfigItem(this->mConfigDCS, "String_OK", this->mTerminationString);
  }
  // Eliminate all the chars after the last ";":
  // If the string is complete this has no effect on the saved object
  // If the string is incomplete this avoids saving wrong data
  size_t pos_del = doubleColsEORstr.rfind(';');
  if (pos_del != std::string::npos) {
    doubleColsEORstr = doubleColsEORstr.erase(pos_del);
  }

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
        if (str == '!') {
          continue; // protection needed to avoid to pass '!' to std::stoi() in vectorizeStringListInt()
        }
        std::vector<unsigned short int> doubleColDisableVector = this->vectorizeStringListInt(str, ":");
        this->mDoubleColsDisableEOR[doubleColDisableVector[0]].push_back(doubleColDisableVector[1]);
      }
      // Second, update map of flagged pixels at EOR
      if (bigVectSplit.size() > 1) {
        std::vector<std::string> pixelFlagsEOR = this->vectorizeStringList(bigVectSplit[1], ";");
        for (const std::string& str : pixelFlagsEOR) {
          std::vector<unsigned short int> pixelFlagsVector = this->vectorizeStringListInt(str, ":");
          this->mPixelFlagsEOR[pixelFlagsVector[0]].push_back(pixelFlagsVector[1]);
        }
      }
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Update pos and npos and check for validity. Return false if there is error
bool ITSDCSParser::updatePosition(size_t& pos, size_t& npos,
                                  const std::string& delimiter, const char* word,
                                  const std::string& inString, bool ignoreNpos /*=false*/)
{
  pos += npos + delimiter.length() + std::string(word).length();
  if (!ignoreNpos) {
    npos = inString.find(delimiter, pos);

    // Check that npos does not go out-of-bounds
    if (npos == std::string::npos) {
      LOG(error) << "Delimiter not found, possibly corrupted data for stave " << this->mStaveName << " !";
      // If the last word is not complete and the terminator is missing the stave is saved as a stave
      this->writeChipInfo(this->mStaveName, -1);
      o2::dcs::addConfigItem(this->mConfigDCS, "String_OK", this->mTerminationString);

      return false;
    }

    npos -= pos;
  }

  return true;
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
    throw std::runtime_error(fmt::format("New value {} differs from old value {}", newValue, memValue));
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
    throw std::runtime_error(fmt::format("New value {} differs from old value {}", newValue, memValue));
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Take string delimited by delimiter and parse into vector of objects
std::vector<std::string> ITSDCSParser::vectorizeStringList(
  const std::string& str, const std::string& delimiter)
{
  std::vector<std::string> str_vect;
  size_t prev_pos = 0, pos = 0;
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
  size_t prev_pos = 0, pos = 0;
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
    this->writeChipInfo(this->mStaveName, chipID);

    // Mark chip as disabled
    o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "1");

    // Mark other information with nonsense value
    o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked_eor", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Pixel_flags", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "String_OK", this->mTerminationString);

    // Ensure that chips are removed from the maps
    mDoubleColsDisableEOR.erase(chipID);
    mPixelFlagsEOR.erase(chipID);
  }

  // Second, loop through all the chips with disabled double columns
  for (std::vector<unsigned short int> maskedDoubleCols : this->mMaskedDoubleCols) {
    unsigned short int chipID = maskedDoubleCols[0];
    maskedDoubleCols.erase(maskedDoubleCols.begin());

    // Write basic chip info
    this->writeChipInfo(this->mStaveName, chipID);
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
    o2::dcs::addConfigItem(this->mConfigDCS, "String_OK", this->mTerminationString);
  }

  // Finally, loop through any remaining chips
  for (const auto& [chipID, v] : this->mDoubleColsDisableEOR) {
    std::string s = this->intVecToStr(v, "|");
    if (s != "-1") { // Ensure no meaningless entries
      this->writeChipInfo(this->mStaveName, chipID);
      o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "0");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", "-1");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked_eor", this->intVecToStr(v, "|"));
      o2::dcs::addConfigItem(
        this->mConfigDCS, "Pixel_flags", this->intVecToStr(this->mPixelFlagsEOR[chipID], "|"));
      this->mPixelFlagsEOR.erase(chipID);
      o2::dcs::addConfigItem(this->mConfigDCS, "String_OK", this->mTerminationString);
    }
  }

  for (const auto& [chipID, v] : this->mPixelFlagsEOR) {
    std::string s = this->intVecToStr(v, "|");
    if (s != "-1") { // Ensure no meaningless entries
      this->writeChipInfo(this->mStaveName, chipID);
      o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "0");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", "-1");
      o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked_eor", "-1");
      o2::dcs::addConfigItem(this->mConfigDCS, "Pixel_flags", this->intVecToStr(v, "|"));
      o2::dcs::addConfigItem(this->mConfigDCS, "String_OK", this->mTerminationString);
    }
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::saveMissingToOutput()
{
  // Loop on the missing staves
  std::vector<string> missingStaves;
  std::vector<string> listStaves = this->listStaves();
  std::vector<string> savedStaves = this->mSavedStaves;
  std::sort(savedStaves.begin(), savedStaves.end());
  std::set_difference(listStaves.begin(), listStaves.end(), savedStaves.begin(), savedStaves.end(),
                      std::inserter(missingStaves, missingStaves.begin()));

  for (std::string stave : missingStaves) {
    this->writeChipInfo(stave, -1);
    o2::dcs::addConfigItem(this->mConfigDCS, "String_OK", "-1");
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
  const std::string& staveName, const short int chipID)
{
  // Stave present in the EOR data and all the "worlds" correctly read
  if (chipID != -1) {
    // First save the Stave to the string
    o2::dcs::addConfigItem(this->mConfigDCS, "Stave", staveName);
    unsigned short int hicPos = getModule(chipID);
    bool hsPos = getHS(chipID);
    o2::dcs::addConfigItem(this->mConfigDCS, "Hs_pos", std::to_string(hsPos));
    o2::dcs::addConfigItem(this->mConfigDCS, "Hic_pos", std::to_string(hicPos));

    // Chip ID inside the module
    o2::dcs::addConfigItem(this->mConfigDCS, "ChipID", std::to_string(chipID % 16));
  }

  // Stave missing in the EOR data or "word" non cottectly read
  else {
    o2::dcs::addConfigItem(this->mConfigDCS, "Stave", staveName);
    o2::dcs::addConfigItem(this->mConfigDCS, "Hs_pos", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Hic_pos", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "ChipID", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Disabled", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Dcol_masked_eor", "-1");
    o2::dcs::addConfigItem(this->mConfigDCS, "Pixel_flags", "-1");
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::pushToCCDB(ProcessingContext& pc)
{
  // Timestamps for CCDB entry
  long tstart = 0, tend = 0;
  // retireve run start/stop times from CCDB
  o2::ccdb::CcdbApi api;
  api.init(mCcdbUrlRct);
  // Initialize empty metadata object for search
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers = api.retrieveHeaders(
    "RCT/Info/RunInformation", metadata, this->mRunNumber);
  if (headers.empty()) { // No CCDB entry is found
    LOG(error) << "Failed to retrieve headers from CCDB with run number " << this->mRunNumber
               << "\nWill default to using the current time for timestamp information";
    tstart = o2::ccdb::getCurrentTimestamp();
    tend = tstart + 365L * 24 * 3600 * 1000;
  } else {
    tstart = std::stol(headers["SOR"]);
    tend = std::stol(headers["EOR"]);
  }

  auto class_name = o2::utils::MemFileHelper::getClassName(mConfigDCS);
  auto class_name_deadMap = o2::utils::MemFileHelper::getClassName(mDeadMap);

  // Create metadata for database object
  metadata = {{"runtype", std::to_string(this->mRunType)}, {"confDBversion", std::to_string(this->mConfigVersion)}, {"runNumber", std::to_string(this->mRunNumber)}};

  std::string path("ITS/Calib/DCS_CONFIG");
  std::string path_deadMap("ITS/Calib/DeadMap");
  const char* filename = "dcs_config.root";
  long current_time = o2::ccdb::getCurrentTimestamp();
  std::string filename_deadMap = "o2-itsmft-NoiseMap_" + std::to_string(current_time) + ".root";
  o2::ccdb::CcdbObjectInfo info(path, "dcs_config", filename, metadata, tstart, tend);
  o2::ccdb::CcdbObjectInfo info_deadMap(path_deadMap, "noise_map", filename_deadMap, metadata, tstart - o2::ccdb::CcdbObjectInfo::MINUTE, tend + 5 * o2::ccdb::CcdbObjectInfo::MINUTE);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&mConfigDCS, &info);
  auto image_deadMap = o2::ccdb::CcdbApi::createObjectImage(&mDeadMap, &info_deadMap);
  info.setFileName(filename);
  info_deadMap.setFileName(filename_deadMap);

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

  // Send dead chips map to ccdb-populator wf
  LOG(info) << "Class Name: " << class_name_deadMap << " | File Name: " << filename_deadMap
            << "\nSending to ccdb-populator the object " << info_deadMap.getPath() << "/" << info_deadMap.getFileName()
            << " of size " << image_deadMap->size() << " bytes, valid for "
            << info_deadMap.getStartValidityTimestamp() << " : "
            << info_deadMap.getEndValidityTimestamp();

  if (mCcdbUrl.empty()) {

    pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "DCS_CONFIG", 1}, *image_deadMap);
    pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "DCS_CONFIG", 1}, info_deadMap);

  } else { // if url is specified, send object to ccdb from THIS wf

    LOG(info) << "Sending object " << info_deadMap.getFileName() << " to " << mCcdbUrl << "/browse/"
              << info_deadMap.getPath() << " from the ITS string parser workflow";
    o2::ccdb::CcdbApi mApi;
    mApi.init(mCcdbUrl);
    mApi.storeAsBinaryFile(
      &image_deadMap->at(0), image_deadMap->size(), info_deadMap.getFileName(), info_deadMap.getObjectType(), info_deadMap.getPath(),
      info_deadMap.getMetaData(), info_deadMap.getStartValidityTimestamp(), info_deadMap.getEndValidityTimestamp());
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
std::vector<std::string> ITSDCSParser::listStaves()
{
  std::vector<std::string> vecStaves = {};
  int stavesPerLayer[] = {12, 16, 20, 24, 30, 42, 48};
  std::string stavenum = "";
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < stavesPerLayer[i]; j++) {
      string stavestring = std::to_string(j);
      int precision = 2 - std::min(2, (int)(stavestring.size()));
      stavenum = std::string(precision, '0').append(std::to_string(j));
      std::string stave = "L" + std::to_string(i) + "_" + stavenum;
      vecStaves.push_back(stave);
    }
  }
  return vecStaves;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSParser::appendDeadChipObj()
{
  // Append an object to the deadMap

  for (auto ch : this->mDisabledChips) {

    unsigned short int hicPos = getModule(ch);
    bool hS = getHS(ch);
    unsigned short int chipInMod = ch % 16;

    unsigned short int globchipID = getGlobalChipID(hicPos, hS, chipInMod);
    this->mDeadMap.maskFullChip(globchipID);
    if (mVerboseOutput) {
      LOG(info) << "Masking dead chip " << globchipID;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
unsigned short int ITSDCSParser::getGlobalChipID(unsigned short int hicPos, bool hS, unsigned short int chipInMod)
{
  // Find the global ID of a chip (0->24119)
  std::vector<unsigned short int> stavesPerLayer = {12, 16, 20, 24, 30, 42, 48};
  std::vector<unsigned short int> chipPerStave = {9, 9, 9, 112, 112, 196, 196};
  std::vector<unsigned short int> maxChipIDlayer = {0};
  int maxChip = 0;
  for (int i = 0; i < 7; i++) {
    maxChip += stavesPerLayer[i] * chipPerStave[i];
    maxChipIDlayer.push_back(maxChip - 1);
  }

  unsigned short int layerNum = std::stoi(this->mStaveName.substr(1, 1));
  unsigned short int staveNum = std::stoi(this->mStaveName.substr(3, 2));

  unsigned short int chipid_in_HIC = ((layerNum > 2 && chipInMod > 7) || layerNum == 0) ? chipInMod : chipInMod + 1;
  unsigned short int modPerLayer = (layerNum > 2 && layerNum < 5) ? 4 : 7;
  unsigned short int add_HSL = (hS) ? (14 * modPerLayer) : 0;

  unsigned short int chipIDglob = chipid_in_HIC + maxChipIDlayer[layerNum] + (staveNum)*chipPerStave[layerNum] + (hicPos - 1) * 14 + add_HSL;

  return chipIDglob;
}

//////////////////////////////////////////////////////////////////////////////
unsigned short int ITSDCSParser::getModule(unsigned short int chipID)
{
  // Get the number of the module (1->7)
  unsigned short int hicPos = 0;
  if (std::stoi((this->mStaveName).substr(1, 1)) > 2) { // OB case
    if (chipID / 16 <= 7) {
      hicPos = chipID / 16;
    } else {
      hicPos = (chipID / 16) - 8;
    }
  } else {
    hicPos = 1; // IB case
  }
  return hicPos;
}

//////////////////////////////////////////////////////////////////////////////
bool ITSDCSParser::getHS(unsigned short int chipInMod)
{
  // Return 0 if the chip is in the HSL, 1 if the chip is in the HSU
  bool hS;
  if (chipInMod / 16 <= 7) {
    hS = 0; // L
  } else {
    hS = 1; // U
  }
  return hS;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSDCSParserSpec()
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("inString", detOrig, "DCS_CONFIG_FILE", 0, Lifetime::Sporadic);
  inputs.emplace_back("nameString", detOrig, "DCS_CONFIG_NAME", 0, Lifetime::Sporadic);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "DCS_CONFIG"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "DCS_CONFIG"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "its-parser",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::its::ITSDCSParser>()},
    Options{
      {"verbose", VariantType::Bool, false, {"Use verbose output mode"}},
      {"ccdb-url", VariantType::String, "", {"CCDB url, default is empty (i.e. send output to CCDB populator workflow)"}},
      {"ccdb-url-rct", VariantType::String, "", {"CCDB url from where to get RCT object for headers, default is o2-ccdb.internal. Use http://alice-ccdb.cern.ch for local tests"}}}};
}
} // namespace its
} // namespace o2
