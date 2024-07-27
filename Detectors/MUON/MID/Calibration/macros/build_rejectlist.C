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
#include "TCanvas.h"
#include "TH1.h"
#include "TGraph.h"
#include "TTimeStamp.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDGlobalMapping/ExtendedMappingInfo.h"
#include "MIDGlobalMapping/GlobalMapper.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#endif

static const std::string sPathQCQuality = "qc/MID/MO/MIDQuality/Trends/global/MIDQuality/MIDQuality";

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
/// @return Vector of start validity of each object sorted in ascending way
std::vector<long> findObjectsTSInPeriod(long start, long end, const o2::ccdb::CcdbApi& api, const char* path)
{
  std::vector<long> ts;
  auto out = api.list(path, false, "text/plain", getTSMS(end), getTSMS(start));
  std::stringstream ss(out);
  std::string token;
  while (ss >> token) {
    if (token.find("Validity") != std::string::npos) {
      ss >> token;
      ts.emplace_back(std::atol(token.c_str()));
    }
  }
  ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
  // Sort timestamps in ascending order
  std::sort(ts.begin(), ts.end());
  return ts;
}

/// @brief Find the first and last time when the quality was good or bad
/// @param qcQuality MID QC quality canvas
/// @param selectBad When true select bad quality
/// @return Pair with first and last time
std::pair<uint64_t, uint64_t> findTSRange(TCanvas* qcQuality, bool selectBad = true)
{
  auto* gr = static_cast<TGraph*>(qcQuality->GetListOfPrimitives()->FindObject("Graph"));
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
/// @param timestamp Timestamp for query
/// @param qcdbApi QCDB api
/// @param ccdbApi CCDB api
/// @param outCCDBApi api of the CCDB where the reject list will be uploaded
/// @return Reject list
std::vector<o2::mid::ColumnData> build_rejectlist(long timestamp, const o2::ccdb::CcdbApi& qcdbApi, const o2::ccdb::CcdbApi& ccdbApi, const o2::ccdb::CcdbApi& outCCDBApi)
{
  std::map<std::string, std::string> metadata;
  auto* qcQuality = qcdbApi.retrieveFromTFileAny<TCanvas>(sPathQCQuality, metadata, getTSMS(timestamp));
  if (!qcQuality) {
    std::cerr << "Cannot find QC quality for " << tsToString(timestamp) << std::endl;
    return {};
  }
  // Find the first and last timestamp where the quality was bad (if any)
  auto badTSRange = findTSRange(qcQuality);
  if (badTSRange.second == 0) {
    std::cout << "All good" << std::endl;
    return {};
  }
  // Search for the last timestamp for which the run quality was good
  auto goodTSRange = findTSRange(qcQuality, false);
  // Query the CCDB to see to which run the timestamp corresponds
  auto oldestTSInQCQuality = (goodTSRange.first == 0) ? badTSRange.first : goodTSRange.first;
  auto grpecs = *ccdbApi.retrieveFromTFileAny<o2::parameters::GRPECSObject>("GLO/Config/GRPECS", metadata, getTSMS(oldestTSInQCQuality));
  if (!grpecs.isDetReadOut(o2::detectors::DetID::MID)) {
    std::cout << "Error: we are probably reading a parallel run" << std::endl;
    grpecs.print();
    return {};
  }
  if (grpecs.getRunType() != o2::parameters::GRPECS::PHYSICS) {
    std::cout << "This is not a physics run: skip" << std::endl;
    grpecs.print();
    return {};
  }

  auto runRange = o2::ccdb::BasicCCDBManager::getRunDuration(ccdbApi, grpecs.getRun());
  long margin = 120000;      // Add a two minutes safety margin
  runRange.first -= margin;  // Subtract margin
  runRange.second += margin; // Add margin

  // Search for hits histogram in the period where the QC quality was bad
  auto tsVector = findObjectsTSInPeriod(badTSRange.first, badTSRange.second, qcdbApi, "qc/MID/MO/QcTaskMIDDigits/Hits");
  if (tsVector.empty()) {
    std::cerr << "Cannot find hits in period " << tsToString(badTSRange.first) << " - " << tsToString(badTSRange.second) << std::endl;
    return {};
  }
  // Focus on the first object found
  TH1* occupancy = qcdbApi.retrieveFromTFileAny<TH1F>("qc/MID/MO/QcTaskMIDDigits/Hits", metadata, getTSMS(tsVector.front()));
  o2::mid::GlobalMapper gm;
  auto infos = gm.buildStripsInfo();
  auto badChannels = findBadChannels(occupancy, infos);
  auto badChannelsCCDB = *ccdbApi.retrieveFromTFileAny<std::vector<o2::mid::ColumnData>>("MID/Calib/BadChannels", metadata, getTSMS(timestamp));
  auto rejectList = getRejectList(badChannels, badChannelsCCDB);
  if (rejectList.empty()) {
    std::cout << "Warning: reject list was empty. It probably means that an entire board is already masked in calibration for run " << grpecs.getRun() << std::endl;
    return {};
  }

  // Print some useful information
  std::cout << "Reject list:" << std::endl;
  for (auto& col : rejectList) {
    std::cout << col << std::endl;
  }
  std::cout << "Run number: " << grpecs.getRun() << std::endl;
  std::cout << "SOR - EOR: " << timeRangeToString(grpecs.getTimeStart(), grpecs.getTimeEnd()) << std::endl;
  std::cout << "SOT - EOT: " << timeRangeToString(runRange.first, runRange.second) << std::endl;
  std::cout << "Good:      " << timeRangeToString(goodTSRange.first, goodTSRange.second) << std::endl;
  std::cout << "Bad:       " << timeRangeToString(badTSRange.first, badTSRange.second) << std::endl;

  // Set the start of the reject list to the last timestamp in which the occupancy was ok
  auto startRL = goodTSRange.second;
  if (goodTSRange.first == 0) {
    // If the quality was bad for the full run, set the start of the reject list to the SOR
    std::cout << "CAVEAT: no good TS found. Will use SOT instead" << std::endl;
    startRL = runRange.first;
  }
  // Set the end of the reject list to the end of run
  auto endRL = runRange.second;
  // Ask if you want to upload the object to the CCDB
  std::cout << "Upload reject list with validity: " << startRL << " - " << endRL << " to " << outCCDBApi.getURL() << "? [y/n]" << std::endl;
  std::string answer;
  std::cin >> answer;
  if (answer == "y") {
    std::cout << "Storing RejectList valid from " << startRL << " to " << endRL << std::endl;
    outCCDBApi.storeAsTFileAny(&rejectList, "MID/Calib/RejectList", metadata, startRL, endRL);
  }
  return rejectList;
}

/// @brief Builds the reject list for the selected timestamp
/// @param timestamp Timestamp for query
/// @param qcdbUrl QCDB URL
/// @param ccdbUrl CCDB URL
/// @param outCCDBUrl URL of the CCDB where the reject list will be uploaded
/// @return Reject list
std::vector<o2::mid::ColumnData> build_rejectlist(long timestamp, const char* qcdbUrl = "http://ali-qcdb-gpn.cern.ch:8083", const char* ccdbUrl = "http://alice-ccdb.cern.ch", const char* outCCDBUrl = "http://localhost:8080")
{
  // Get the QC quality object for the selected timestamp
  o2::ccdb::CcdbApi qcdbApi;
  qcdbApi.init(qcdbUrl);
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  o2::ccdb::CcdbApi outCCDBApi;
  outCCDBApi.init(outCCDBUrl);
  return build_rejectlist(timestamp, qcdbApi, ccdbApi, outCCDBApi);
}

/// @brief Builds the reject list iin a time range
/// @param start Start time for query
/// @param end End time for query
/// @param qcdbUrl QCDB URL
/// @param ccdbUrl CCDB URL
/// @param outCCDBUrl URL of the CCDB where the reject lists will be uploaded
void build_rejectlist(long start, long end, const char* qcdbUrl = "http://ali-qcdb-gpn.cern.ch:8083", const char* ccdbUrl = "http://alice-ccdb.cern.ch", const char* outCCDBUrl = "http://localhost:8080")
{
  // Query the MID QC quality objects
  o2::ccdb::CcdbApi qcdbApi;
  qcdbApi.init(qcdbUrl);
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  o2::ccdb::CcdbApi outCCDBApi;
  outCCDBApi.init(outCCDBUrl);
  auto objectsTS = findObjectsTSInPeriod(start, end, qcdbApi, sPathQCQuality.c_str());
  for (auto ts : objectsTS) {
    build_rejectlist(ts, qcdbApi, ccdbApi, outCCDBApi);
  }
}