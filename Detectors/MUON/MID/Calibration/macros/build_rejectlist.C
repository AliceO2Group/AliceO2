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

/// \file   MID/Calibration/macros/build_rejectlist.C
/// \brief  Analyse QC output and build reject list
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 July 2024

#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <limits>
#include <algorithm>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TGraph.h"
#include "TTimeStamp.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDBase/ColumnDataHandler.h"
#include "MIDGlobalMapping/ExtendedMappingInfo.h"
#include "MIDGlobalMapping/GlobalMapper.h"
#include "MIDFiltering/ChannelMasksHandler.h"

// ...
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#endif

static const std::string sPathQCQuality = "qc/MID/MO/MIDQuality/Trends/global/MIDQuality/MIDQuality";

/// @brief  Reject list object
struct RejectListStruct {
  long start = 0;                                /// Start validity
  long end = 0;                                  /// End validity
  std::vector<o2::mid::ColumnData> rejectList{}; /// Bad channels
};

/// @brief Useful metadata
struct MDStruct {
  long start = 0;      /// Start validity
  long end = 0;        /// End validity
  int runNumber = 0;   /// Run number
  std::string runType; /// Run Type

  bool operator<(const MDStruct& other) const { return start < other.start; }
  bool operator==(const MDStruct& other) const { return start == other.start; }
};

/// @brief Get timestamp in milliseconds
/// @param timestamp Input timestamp (in s or ms)
/// @return Timestamp in ms
long getTSMS(long timestamp)
{
  if (timestamp < 1000000000000) {
    return timestamp * 1000;
  }
  return timestamp;
}

/// @brief Get timestamp in seconds
/// @param timestamp Input timestamp (in s or ms)
/// @return Timestamp in s
long getTSS(long timestamp)
{
  if (timestamp < 1000000000000) {
    return timestamp;
  }
  return timestamp / 1000;
}

/// @brief Converts timestamp to human-readable string
/// @param timestamp Timestamp as long
/// @return Timestamp as human-readable string
std::string tsToString(long timestamp)
{
  return TTimeStamp(getTSS(timestamp)).AsString("l");
}

/// @brief Converts time range in a human-readable string
/// @param start Start time
/// @param end End time
/// @return Time range as human-readable string
std::string timeRangeToString(long start, long end)
{
  std::stringstream ss;
  ss << start << " - " << end << " (" << tsToString(start) << " - " << tsToString(end) << ")";
  return ss.str();
}

/// @brief Query the CDB path and search for initial validity of objects
/// @param start Query objects created not before
/// @param end Query objects created not after
/// @param api CDB api
/// @param path CDB path
/// @return Vector of metadata in ascending order
std::vector<MDStruct> findObjectsMDInPeriod(long start, long end, const o2::ccdb::CcdbApi& api, const char* path)
{
  std::vector<MDStruct> mds;
  long creationDelayMS = 300000; // The objects can be created up to 5 minutes after the end of run
  auto out = api.list(path, false, "application/json", getTSMS(end) + creationDelayMS, getTSMS(start));
  rapidjson::Document doc;
  doc.Parse(out.c_str());
  for (auto& obj : doc["objects"].GetArray()) {
    MDStruct md;
    md.start = obj["validFrom"].GetInt64();
    if (getTSMS(end) < getTSMS(md.start)) {
      // Since we query on the creation time, adding a delay
      // we need to cross-check here that we are within the run
      continue;
    }
    md.end = obj["validUntil"].GetInt64();
    md.runNumber = std::atoi(obj["RunNumber"].GetString());
    md.runType = obj["RunType"].GetString();
    mds.emplace_back(md);
  }
  mds.erase(std::unique(mds.begin(), mds.end()), mds.end());
  // Sort timestamps in ascending order
  std::sort(mds.begin(), mds.end());
  return mds;
}

/// @brief Gets the quality trend graph from the quality canvas
/// @param qcQuality MID QC quality canvas
/// @return Quality trend graph
TGraph* getQualityTrend(const TCanvas* qcQuality)
{
  return static_cast<TGraph*>(qcQuality->GetListOfPrimitives()->FindObject("Graph"));
}

/// @brief Find the first and last time when the quality was good or bad
/// @param qcQuality MID QC quality canvas
/// @param selectBad When true select bad quality
/// @return Pair with first and last time
std::pair<uint64_t, uint64_t> findTSRange(TCanvas* qcQuality, bool selectBad = true)
{
  // Gets the plot with the quality flags
  // The flag values are:
  // Good: 3.5
  // Medium: 2.5
  // Bad: 1.5
  // Null: 0.5
  auto* gr = getQualityTrend(qcQuality);
  double xp, yp;
  std::pair<uint64_t, uint64_t> range{std::numeric_limits<uint64_t>::max(), 0};
  for (int ip = 0; ip < gr->GetN(); ++ip) {
    gr->GetPoint(ip, xp, yp);
    if (selectBad == (yp > 1 && yp < 3)) {
      uint64_t xpi = static_cast<uint64_t>(1000 * xp);
      range.first = std::min(range.first, xpi);
      range.second = std::max(range.second, xpi);
    }
  }
  if (range.first == std::numeric_limits<uint64_t>::max()) {
    range.first = 0;
  }
  return range;
}

/// @brief Gets the first and last timestamp in the quality
/// @param qcQuality MID QC quality canvas
/// @return Pair with the first and last timestamp in the quality trend
std::pair<uint64_t, uint64_t> getFirstLast(const TCanvas* qcQuality)
{
  auto* gr = getQualityTrend(qcQuality);
  double xp1, xp2, yp;
  gr->GetPoint(0, xp1, yp);
  gr->GetPoint(gr->GetN() - 1, xp2, yp);
  return {static_cast<uint64_t>(xp1 * 1000), static_cast<uint64_t>(xp2 * 1000)};
}

/// @brief Update the selected range of timestamp
/// @param selectedTSRange Reference to the selected range to be modified
/// @param qcTSRange Range of the MID quality trend
/// @param runRange Run range
void updateRange(std::pair<uint64_t, uint64_t>& selectedTSRange, const std::pair<uint64_t, uint64_t> qcTSRange, const std::pair<uint64_t, uint64_t> runRange)
{
  if (selectedTSRange.first == qcTSRange.first) {
    selectedTSRange.first = runRange.first;
  }
  if (selectedTSRange.second == qcTSRange.second) {
    selectedTSRange.second = runRange.second;
  }
}

/// @brief Find bad channels from the occupancy histograms
/// @param hits Occupancy histogram
/// @param infos Mapping
/// @return Vector of bad channels
std::vector<o2::mid::ColumnData> findBadChannels(TH1* hits, std::vector<o2::mid::ExtendedMappingInfo>& infos)
{
  std::map<uint16_t, o2::mid::ColumnData> badChannelsMap;
  for (int ibin = 1; ibin <= hits->GetNbinsX(); ++ibin) {
    if (hits->GetBinContent(ibin) == 0) {
      auto info = infos[ibin - 1];
      auto uniqueId = o2::mid::getColumnDataUniqueId(info.deId, info.columnId);
      o2::mid::ColumnData col;
      col.deId = info.deId;
      col.columnId = info.columnId;
      auto result = badChannelsMap.emplace(uniqueId, col);
      result.first->second.addStrip(info.stripId, info.cathode, info.lineId);
    }
  }

  std::vector<o2::mid::ColumnData> badChannels;
  for (auto& item : badChannelsMap) {
    badChannels.emplace_back(item.second);
  }
  return badChannels;
}

/// @brief Converts the bad channels from the occupancy into a reject list (removing the ones from CCDB)
/// @param badChannels Bad channels from the analysis of the occupancy histogram
/// @param badChannelsCCDB Bad channels in the CCDB
/// @return Reject list
std::vector<o2::mid::ColumnData> getRejectList(std::vector<o2::mid::ColumnData> badChannels, const std::vector<o2::mid::ColumnData>& badChannelsCCDB)
{
  o2::mid::ChannelMasksHandler mh;
  mh.switchOffChannels(badChannelsCCDB);
  for (auto& bad : badChannels) {
    mh.applyMask(bad);
  }
  badChannels.erase(std::remove_if(badChannels.begin(), badChannels.end(), [](const o2::mid::ColumnData col) { return col.isEmpty(); }),
                    badChannels.end());
  return badChannels;
}

/// @brief Builds the reject list for the selected timestamp
/// @param md MD structure
/// @param qcdbApi QCDB api
/// @param ccdbApi CCDB api
/// @param outCCDBApi api of the CCDB where the reject list will be uploaded
/// @return Reject list
RejectListStruct build_rejectlist(const MDStruct& md, const o2::ccdb::CcdbApi& qcdbApi, const o2::ccdb::CcdbApi& ccdbApi)
{
  RejectListStruct rl;
  if (md.runType != "PHYSICS") {
    std::cout << "Run " << md.runNumber << " is of type " << md.runType << ": skip" << std::endl;
    return rl;
  }

  std::map<std::string, std::string> metadata;
  auto* qcQuality = qcdbApi.retrieveFromTFileAny<TCanvas>(sPathQCQuality, metadata, getTSMS(md.start));
  if (!qcQuality) {
    std::cerr << "Cannot find QC quality for " << tsToString(md.start) << std::endl;
    return rl;
  }

  // Find the first and last timestamp where the quality was bad (if any)
  auto badTSRange = findTSRange(qcQuality);
  if (badTSRange.second == 0) {
    std::cout << "All good" << std::endl;
    return rl;
  }

  // Find the first and last timestamp where the quality flag was set
  auto qualityTSRange = getFirstLast(qcQuality);
  // Search for the last timestamp for which the run quality was good
  auto goodTSRange = findTSRange(qcQuality, false);

  auto runRange = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, md.runNumber);
  updateRange(badTSRange, qualityTSRange, runRange);
  updateRange(goodTSRange, qualityTSRange, runRange);

  // Search for hits histogram in the period where the QC quality was bad
  auto mdVector = findObjectsMDInPeriod(badTSRange.first, badTSRange.second, qcdbApi, "qc/MID/MO/QcTaskMIDDigits/Hits");
  if (mdVector.empty()) {
    std::cerr << "Cannot find hits in period " << tsToString(badTSRange.first) << " - " << tsToString(badTSRange.second) << std::endl;
    return {};
  }
  // Focus on the last object found
  // We chose the last instead of the first because it might happen that
  // we lose additional boards before the EOR
  // If we build the reject list for the first object, we would therefore miss some boards
  TH1* occupancy = qcdbApi.retrieveFromTFileAny<TH1F>("qc/MID/MO/QcTaskMIDDigits/Hits", metadata, getTSMS(mdVector.back().start));
  o2::mid::GlobalMapper gm;
  auto infos = gm.buildStripsInfo();
  auto badChannels = findBadChannels(occupancy, infos);
  auto badChannelsCCDB = *ccdbApi.retrieveFromTFileAny<std::vector<o2::mid::ColumnData>>("MID/Calib/BadChannels", metadata, getTSMS(md.start));
  rl.rejectList = getRejectList(badChannels, badChannelsCCDB);
  if (rl.rejectList.empty()) {
    std::cout << "Warning: reject list was empty. It probably means that an entire board is already masked in calibration for run " << md.runNumber << std::endl;
    return rl;
  }

  // Print some useful information
  std::cout << "Reject list:" << std::endl;
  for (auto& col : rl.rejectList) {
    std::cout << col << std::endl;
  }
  std::cout << "Run number: " << md.runNumber << std::endl;
  std::cout << "SOT - EOT: " << timeRangeToString(runRange.first, runRange.second) << std::endl;
  std::cout << "Good:      " << timeRangeToString(goodTSRange.first, goodTSRange.second) << std::endl;
  std::cout << "Bad:       " << timeRangeToString(badTSRange.first, badTSRange.second) << std::endl;
  std::cout << "Fraction bad: " << static_cast<double>(badTSRange.second - badTSRange.first) / static_cast<double>(runRange.second - runRange.first) << std::endl;

  // Set the start of the reject list to the last timestamp in which the occupancy was ok
  rl.start = goodTSRange.second;
  rl.end = badTSRange.second;
  return rl;
}

/// @brief Loads the reject list from a json file
/// @param ccdbApi CCDB api
/// @param filename json filename
/// @return Reject list structure
RejectListStruct load_from_json(const o2::ccdb::CcdbApi& ccdbApi, const char* filename = "rejectlist.json")
{
  // Open the JSON file
  std::cout << "Reading reject list from file " << filename << std::endl;
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    std::cerr << "Could not open the file!" << std::endl;
    return {};
  }

  // Create an IStreamWrapper for file input stream
  rapidjson::IStreamWrapper isw(inFile);
  rapidjson::Document doc;
  if (doc.ParseStream(isw).HasParseError()) {
    std::cerr << "Problem parsing " << filename << std::endl;
    return {};
  }

  // manual-validity interval in ms:
  int64_t startTSms = 0;
  int64_t endTSms = 0;

  // run numbers from the json
  int startRun = doc["startRun"].GetInt();
  int endRun = doc["endRun"].GetInt();

  // check if there are non-zero timestamps in the json
  bool hasStartTT = doc.HasMember("startTT") && doc["startTT"].IsInt64() && doc["startTT"].GetInt64() != 0;
  bool hasEndTT = doc.HasMember("endTT") && doc["endTT"].IsInt64() && doc["endTT"].GetInt64() != 0;
  if (hasStartTT && hasEndTT) {
    startTSms = doc["startTT"].GetInt64();
    endTSms = doc["endTT"].GetInt64();

    // sanity check against the run boundaries
    auto runStart = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, startRun).first;
    auto runEnd = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, endRun).second;
    if (startTSms < runStart || endTSms > runEnd) {
      std::cout
        << "\n\nWarning: manual timestamps [" << startTSms << " - " << endTSms
        << "] lie outside run interval [" << runStart << " - " << runEnd << "]\n\n\n";
    }
  } else {
    // use run start/end if there are no timestamps in the json
    startTSms = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, startRun).first;
    endTSms = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, endRun).second;
  }

  RejectListStruct rl;
  rl.start = startTSms;
  rl.end = endTSms;
  std::cout << "Manual RL validity: " << timeRangeToString(rl.start, rl.end) << std::endl;
  auto rlArray = doc["rejectList"].GetArray();
  for (auto& ar : rlArray) {
    o2::mid::ColumnData col;
    col.deId = ar["deId"].GetInt();
    col.columnId = ar["columnId"].GetInt();
    auto patterns = ar["patterns"].GetArray();
    for (size_t iar = 0; iar < 5; ++iar) {
      col.patterns[iar] = std::strtol(patterns[iar].GetString(), NULL, 16);
    }
    rl.rejectList.emplace_back(col);
    std::cout << col << std::endl;
  }
  return rl;
}

/// @brief Merges the manual and automatic reject lists
/// @param manualRL Manual reject list from json file
/// @param rls Reject list from QCDB and CCDB
/// @return Merged reject list
std::vector<RejectListStruct> merge_rejectlists(const RejectListStruct& manualRL, const std::vector<RejectListStruct>& rls)
{
  std::vector<RejectListStruct> merged;
  if (rls.empty()) {
    merged.emplace_back(manualRL);
    return merged;
  }
  o2::mid::ColumnDataHandler ch;
  RejectListStruct tmpRL;
  long lastEnd = manualRL.start;
  for (auto& rl : rls) {
    std::cout << "Checking rl with validity:      " << timeRangeToString(rl.start, rl.end) << std::endl;
    if (rl.start >= manualRL.start && rl.end <= manualRL.end) {
      // The period is included in the validity of the manual reject list
      if (rl.start > lastEnd) {
        // Fill holes between periods
        tmpRL = manualRL;
        tmpRL.start = lastEnd;
        tmpRL.end = rl.start;
        merged.emplace_back(tmpRL);
        std::cout << "Adding manual RL with validity: " << timeRangeToString(tmpRL.start, tmpRL.end) << std::endl;
      }
      lastEnd = rl.end;

      // merge
      ch.clear();
      ch.merge(rl.rejectList);
      ch.merge(manualRL.rejectList);
      tmpRL = rl;
      tmpRL.rejectList = ch.getMerged();
      std::sort(tmpRL.rejectList.begin(), tmpRL.rejectList.end(), [](const o2::mid::ColumnData& col1, const o2::mid::ColumnData& col2) { return o2::mid::getColumnDataUniqueId(col1.deId, col1.columnId) < o2::mid::getColumnDataUniqueId(col2.deId, col2.columnId); });
      merged.emplace_back(tmpRL);
      std::cout << "Merging RL with validity:       " << timeRangeToString(tmpRL.start, tmpRL.end) << std::endl;
      // std::cout << "Before: " << std::endl;
      // for (auto& col : rl.rejectList) {
      //   std::cout << col << std::endl;
      // }
      // std::cout << "After: " << std::endl;
      // for (auto& col : tmpRL.rejectList) {
      //   std::cout << col << std::endl;
      // }
    } else {
      if (rl.start > manualRL.end && lastEnd < manualRL.end) {
        // Close manual period
        tmpRL = manualRL;
        tmpRL.start = lastEnd;
        merged.emplace_back(tmpRL);
        std::cout << "Adding manual RL with validity: " << timeRangeToString(tmpRL.start, tmpRL.end) << std::endl;
        lastEnd = manualRL.end;
      }
      // Add current reject list as it is
      merged.emplace_back(rl);
      std::cout << "Adding RL with validity: " << timeRangeToString(rl.start, rl.end) << std::endl;
    }
  }
  return merged;
}

/// @brief Builds the reject list in a time range
/// @param start Start time for query
/// @param end End time for query
/// @param qcdbUrl QCDB URL
/// @param ccdbUrl CCDB URL
/// @param outCCDBUrl URL of the CCDB where the reject lists will be uploaded
void build_rejectlist(long start, long end, const char* qcdbUrl = "http://ali-qcdb-gpn.cern.ch:8083", const char* ccdbUrl = "http://alice-ccdb.cern.ch", const char* outCCDBUrl = "http://localhost:8080", const char* json_rejectlist = "")
{
  // Query the MID QC quality objects
  o2::ccdb::CcdbApi qcdbApi;
  qcdbApi.init(qcdbUrl);
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  std::vector<RejectListStruct> rls;
  auto objectsMD = findObjectsMDInPeriod(start, end, qcdbApi, sPathQCQuality.c_str());
  for (auto md : objectsMD) {
    auto rl = build_rejectlist(md, qcdbApi, ccdbApi);
    if (rl.start != rl.end) {
      rls.emplace_back(rl);
    }
  }

  if (!std::string(json_rejectlist).empty()) {
    auto rlManual = load_from_json(ccdbApi, json_rejectlist);
    rls = merge_rejectlists(rlManual, rls);
  }

  o2::ccdb::CcdbApi outCCDBApi;
  outCCDBApi.init(outCCDBUrl);
  std::map<std::string, std::string> metadata;
  for (auto& rl : rls) {
    // Ask if you want to upload the object to the CCDB
    std::cout << "Upload reject list with validity: " << rl.start << " - " << rl.end << " to " << outCCDBApi.getURL() << "? [y/n]" << std::endl;
    std::string answer;
    std::cin >> answer;
    if (answer == "y") {
      std::cout << "Storing RejectList valid from " << rl.start << " to " << rl.end << std::endl;
      outCCDBApi.storeAsTFileAny(&rl.rejectList, "MID/Calib/RejectList", metadata, rl.start, rl.end);
    }
  }
}
