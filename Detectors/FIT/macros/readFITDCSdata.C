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

/// \file readFITDCSdata.C
/// \brief ROOT macro for reading the FIT DCS data from CCDB
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsFIT/DCSDPValues.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DeliveryType.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TMultiGraph.h"
#include "TStyle.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <time.h>
#include <unordered_map>
#include <vector>

#endif

#include "Framework/Logger.h"

#include <boost/algorithm/string.hpp>

// Helper functions

const std::string epochToReadable(const long timestamp);
std::vector<std::string> getAliases(const std::string& input, const o2::ccdb::CcdbApi& ccdbApi, const std::string& detectorName, const long timestamp);
void plotFITDCSmultigraph(const TMultiGraph& multiGraph);

/// ROOT macro for reading FIT DCS data from CCDB.
///
/// The macro can:
/// - Plot the trends (default ON)
/// - Store a TMultiGraph of the trends to file (default OFF)
/// - Store the trend values to file (default OFF). The data format is std::map<std::string, o2::fit::DCSDPValues>, where the key is the data point alias.
/// - Print the trend values to the output (default OFF).
///
/// Please note that query timestamps behaves a bit "approximately" sometimes. If values in the ends of the queried time period are missing,
/// try to include some extra time in the end where values are missing.
///
/// \param detectorName     FIT subdetector, i.e. FT0/FV0/FDD, for which to query the data points.
/// \param dataPointAliases A string specifying what data points to query.
///                         It can be a semicolon separated list of data point aliases, e.g. "FT0/HV/FT0_A/MCP_A1/actual/iMon;FT0/HV/FT0_A/MCP_A2/actual/iMon"
///                         or a relative path to a file where the aliases are listed (one alias on each line), e.g. "aliases.txt" or "path/to/aliases.txt".
///                         If left empty, all data points defined in [ccdbUrl]/[detectorName]/Config/DCSDPconfig are queried.
/// \param timeStart        UNIX timestamp (in ms) for start of data point query. If omitted, one hour before the end time is used.
/// \param timeEnd          UNIX timestamp (in ms) for end of data point query. If omitted, current time is used.
/// \param ccdbUrl          CCDB url.
/// \param plot             Whether to plot the data point values.
/// \param rootOutput       If specified, a plot and raw values are stored in [rootOutput].
/// \param print            Whether to print the data point values.
/// \param textOutput       If specified the data point values are stored in '[textOutput]'. TODO: implement
/// \param verbose          Verbose mode for debugging.
void readFITDCSdata(std::string detectorName = "FT0",
                    const std::string& dataPointAliases = "",
                    long timeStart = -1,
                    long timeEnd = -1,
                    const std::string& ccdbUrl = "https://alice-ccdb.cern.ch",
                    const bool plot = true,
                    const std::string& rootOutput = "",
                    const bool print = false,
                    const std::string& textOutput = "",
                    const bool verbose = true)
{
  // Parse and check detector name
  boost::to_upper(detectorName);
  if (detectorName != "FT0" && detectorName != "FV0" && detectorName != "FDD") {
    LOGP(fatal, "Invalid detector name provided: '{}'. Please use [FT0/FV0/FDD].", detectorName);
    return;
  }
  LOGP(info, "Fetching DCS data points for {}.", detectorName);

  // Init CCDB stuff
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  const std::string ccdbPath = detectorName + "/Calib/DCSDPs";
  const std::map<std::string, std::string> metadata;

  // Set query time interval
  long timeNow = o2::ccdb::getCurrentTimestamp();
  if (timeEnd < 0 || timeEnd > timeNow) {
    timeEnd = timeNow;
  }
  if (timeStart < 0 || timeStart > timeNow) {
    timeStart = std::max(timeEnd - 1 * o2::ccdb::CcdbObjectInfo::HOUR, 0L);
  }
  if (timeStart > timeEnd) {
    long timeTmp = timeEnd;
    timeEnd = timeStart;
    timeStart = timeTmp;
  }
  LOG(info) << "Querying data points for time interval:";
  LOGP(info, "START {} ({})", timeStart, epochToReadable(timeStart));
  LOGP(info, "END   {} ({})", timeEnd, epochToReadable(timeEnd));

  // Define what data points to query
  std::vector<std::string> requestedDPaliases = getAliases(dataPointAliases, ccdbApi, detectorName, timeStart);

  LOG(info) << "Querying datapoint(s):";
  for (auto& alias : requestedDPaliases) {
    LOG(info) << alias;
  }

  // Set up data point value storage
  std::map<std::string, o2::fit::DCSDPValues> dataSeries;
  for (std::string& alias : requestedDPaliases) {
    dataSeries[alias] = o2::fit::DCSDPValues();
  }

  // Query data points

  long queryTimeStamp = timeStart;
  // Headers are used to compare fetched CCDB objects
  std::map<std::string, std::string> headers;
  std::map<std::string, std::string> headersPrev;
  std::map<std::string, std::string> headersLatest = ccdbApi.retrieveHeaders(ccdbPath, metadata, timeEnd);
  std::unordered_map<o2::dcs::DataPointIdentifier, o2::fit::DCSDPValues>* ccdbMap = nullptr;     // Pointer to the CCDB object
  std::unordered_map<o2::dcs::DataPointIdentifier, o2::fit::DCSDPValues>* ccdbMapPrev = nullptr; // Pointer to the previously queried CCDB object
  o2::dcs::DataPointIdentifier dpIdTmp;                                                          // DataPointIdentifier object used as CCDB object map key
  o2::fit::DCSDPValues* ccdbDPValuesPointer = nullptr;                                           // Pointer to the CCDB object map values

  while (queryTimeStamp <= timeEnd) {
    ccdbMap = ccdbApi.retrieveFromTFileAny<std::unordered_map<o2::dcs::DataPointIdentifier, o2::fit::DCSDPValues>>(ccdbPath, metadata, queryTimeStamp, &headers);

    if (!ccdbMap) {
      // TODO: Improve functionality when data not found
      if (queryTimeStamp == timeStart && headersLatest.empty()) {
        LOGP(fatal, "No CCDB object found for start time {} ({}) or end time {} ({}).\nThe macro can't handle this, aborting.", queryTimeStamp, epochToReadable(queryTimeStamp), timeEnd, epochToReadable(timeEnd));
        return;
      } else if (headersLatest.empty()) {
        LOGP(error, "No CCDB object found for timestamp {} ({}) or end time {} ({}).", queryTimeStamp, epochToReadable(queryTimeStamp), timeEnd, epochToReadable(timeEnd));
        LOGP(error, "The O2 workflow might have been stopped, and there are periods without valid objects in CCDB. The macro can't handle this, stopping here.");
        break;
      } else {
        LOGP(error, "No valid CCDB object for timestamp {} ({}).", queryTimeStamp, epochToReadable(queryTimeStamp));
        queryTimeStamp += 10 * o2::ccdb::CcdbObjectInfo::MINUTE;
        continue;
      }
    } else if (headers["Location"] == headersPrev["Location"]) {
      // CCDB was maybe not updated as it should after the last CCDB object (i.e. there should always be a new object after 10 mins)

      if ((headers["Location"] == headersLatest["Location"])) {
        LOGP(warning, "No newer CCDB objects for query period found. The O2 workflow might have been stopped.");
        break;
      } else {
        LOGP(warning, "No new CCDB object for time {} ({}). The O2 workflow might have been stopped temporarily.", queryTimeStamp, epochToReadable(queryTimeStamp));
        // There are newer objects for the query period, move on to the next query
        // Currently the CCDB is updated every 10 mins
        queryTimeStamp += 10 * o2::ccdb::CcdbObjectInfo::MINUTE;
        continue;
      }
    } else {
      if (verbose) {
        LOGP(info, "CCDB object for timestamp {} ({}) found.", queryTimeStamp, epochToReadable(queryTimeStamp));
      }
      headersPrev = headers;
    }

    // The CCDB object should always contain values for all datapoints. This is just to check that.
    if ((detectorName == "FT0" && ccdbMap->size() != 477) || (detectorName == "FV0" && ccdbMap->size() != 147) || (detectorName == "FDD" && ccdbMap->size() != 76)) {
      LOGP(error,
           "Wrong number of DCS datapoints fetched for {}, got {}. There is a bug, please send output of this script, with input parameters, to andreas.molander@cern.ch.",
           detectorName, ccdbMap->size());
    }

    for (std::string& alias : requestedDPaliases) {
      // Need to fetch data point based on its type. To avoid specifying/knowing the type of each data point, try all used types instead.
      // TODO: This is not very nice, the dataformat in CCDB should be changed.
      o2::dcs::DataPointIdentifier::FILL(dpIdTmp, alias, o2::dcs::DeliveryType::DPVAL_DOUBLE);
      if (ccdbMap->find(dpIdTmp) != ccdbMap->end()) {
        ccdbDPValuesPointer = &ccdbMap->at(dpIdTmp);
      } else {
        o2::dcs::DataPointIdentifier::FILL(dpIdTmp, alias, o2::dcs::DeliveryType::DPVAL_UINT);
        if (ccdbMap->find(dpIdTmp) != ccdbMap->end()) {
          ccdbDPValuesPointer = &ccdbMap->at(dpIdTmp);
        } else {
          ccdbDPValuesPointer = nullptr;
        }
      }

      if (ccdbDPValuesPointer) {
        // If there are no values read for this DP yet, or there are newer values in the current CCDB object
        if (dataSeries[alias].empty() || (dataSeries[alias].values.back().first < ccdbDPValuesPointer->values.back().first)) {
          // if (verbose) {
          //   LOGP(info, "Newer values found.");
          // }

          // Iterate through the values in the current CCDB object
          for (auto& it : ccdbDPValuesPointer->values) {
            if ((it.first >= timeStart && it.first <= timeEnd) && (dataSeries[alias].empty() || it.first > dataSeries[alias].values.back().first)) {
              // if (dataSeries[alias].empty() || it.first > dataSeries[alias].values.back().first) {
              dataSeries[alias].add(it.first, it.second);
            }
          }
        }
      } else {
        LOGP(error, "Requested DP '{}' not found in CCDB object for timestamp {}.", alias, queryTimeStamp);
      }
    }

    // Currently the CCDB is updated every 10 mins
    queryTimeStamp += 10 * o2::ccdb::CcdbObjectInfo::MINUTE;
  }

  if (print) {
    LOG(info) << "Printing data point values:";
    for (auto& it : dataSeries) {
      LOGP(info, "{}", it.first);
      LOGP(info, "{} value(s):", it.second.values.size());
      if (verbose) {
        for (auto& value : it.second.values) {
          LOGP(info, "TIME = {} ({}), VALUE = {}", value.first, epochToReadable(value.first), value.second);
        }
      } else {
        LOGP(info, "First value:");
        LOGP(info, "TIME = {} ({}), VALUE = {}", it.second.values.front().first, epochToReadable(it.second.values.front().first), it.second.values.front().second);
        LOGP(info, "Last value:");
        LOGP(info, "TIME = {} ({}), VALUE = {}", it.second.values.back().first, epochToReadable(it.second.values.back().first), it.second.values.back().second);
      }
      LOG(info);
    }
  }

  if (plot || !rootOutput.empty()) {
    std::unique_ptr<TMultiGraph> multiGraph(new TMultiGraph);
    multiGraph->SetName("mgDCSDPTrends");
    std::vector<TGraph*> graphs;
    int pointCounter = 0;

    for (auto& dp : dataSeries) {
      if (!dp.second.empty()) {
        graphs.push_back(new TGraph);
        graphs.back()->SetName(dp.first.c_str());
        graphs.back()->GetXaxis()->SetTimeDisplay(1);
        graphs.back()->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
        graphs.back()->SetMarkerStyle(20);
        pointCounter = 0;

        for (size_t i = 0; i < dp.second.values.size(); i++) {
          if (i > 0) {
            graphs.back()->SetPoint(pointCounter++, dp.second.values.at(i).first / 1000., dp.second.values.at(i - 1).second);
          }
          graphs.back()->SetPoint(pointCounter++, dp.second.values.at(i).first / 1000., dp.second.values.at(i).second);
        }
        // for (auto& dpValue : dp.second.values) {
        //   graphs.back()->SetPoint(pointCounter++, dpValue.first / 1000., dpValue.second);
        // }
        multiGraph->Add(graphs.back());
      } else {
        LOGP(warning, "No CCDB data found for alias '{}'. Data point excluded from graph.", dp.first);
      }
    }

    multiGraph->GetXaxis()->SetTimeDisplay(1);
    multiGraph->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");

    if (plot) {
      plotFITDCSmultigraph(*multiGraph);
    }

    if (!rootOutput.empty()) {
      LOGP(info, "Storing trends in '{}'", rootOutput);
      std::unique_ptr<TFile> rootFile(TFile::Open(rootOutput.c_str(), "RECREATE"));
      multiGraph->Write();
      rootFile->WriteObject(&dataSeries, "DCSDPValues");
    }

    if (!textOutput.empty()) {
      LOG(info) << "Storing data point values to text file is not implemented yet.";
    }
  }
}

void plotFITDCSdataFromFile(const std::string& fileName)
{
  std::unique_ptr<TFile> file(TFile::Open(fileName.c_str()));
  if (!file || file->IsZombie()) {
    LOGP(fatal, "Error opening file '{}'", fileName);
    return;
  }

  std::unique_ptr<TMultiGraph> multiGraph(file->Get<TMultiGraph>("mgDCSDPTrends"));
  if (!multiGraph) {
    LOGP(fatal, "Cannot find TMultiGraph 'mgDCSDPTrends' in {}", fileName);
    return;
  }

  plotFITDCSmultigraph(*multiGraph);
}

/// ROOT macro for printing the contents of one CCDB object. Used for debugging.
void printCCDBObject(const std::string detectorName = "FT0",
                     long timestamp = -1,
                     const std::string ccdbUrl = "https://alice-ccdb.cern.ch")
{
  // // Parse and check detector name
  // boost::to_upper(detectorName);
  // if (detectorName != "FT0" && detectorName != "FV0" && detectorName != "FDD") {
  //   LOGP(fatal, "Invalid detector name provided: '{}'. Please use [FT0/FV0/FDD].", detectorName);
  //   return;
  // }
  LOGP(info, "Querying CCDB object for {}.", detectorName);

  // Init CCDB stuff
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  const std::string ccdbPath = detectorName + "/Calib/DCSDPs";
  const std::map<std::string, std::string> metadata;

  if (timestamp < 0) {
    timestamp = o2::ccdb::getCurrentTimestamp();
  }

  std::unordered_map<o2::dcs::DataPointIdentifier, o2::fit::DCSDPValues>* map = ccdbApi.retrieveFromTFileAny<std::unordered_map<o2::dcs::DataPointIdentifier, o2::fit::DCSDPValues>>(ccdbPath, metadata, timestamp);

  if (!map) {
    LOGP(fatal, "CCDB object not found in {}/{} for timestamp {}", ccdbUrl, ccdbPath, timestamp);
    return;
  }

  int nEmptyDPs = 0;

  for (auto& it : *map) {
    if (it.second.empty()) {
      nEmptyDPs++;
    }
    std::stringstream tmp;
    tmp << it.first;
    LOGP(info, "DPID = {}", tmp.str());
    it.second.print();
  }

  LOGP(info, "Size of map = {}", map->size());
  LOGP(info, "Empty DPs   = {}", nEmptyDPs);
}

// Helper functions

void plotFITDCSmultigraph(const TMultiGraph& multiGraph)
{
  gStyle->SetPalette(kRainBow);
  gStyle->SetTimeOffset(0);
  gStyle->SetLabelOffset(0.03);

  TCanvas* canvas = new TCanvas("canvas", "FIT DCS DP trends");
  multiGraph.DrawClone("apl pmc plc");
  canvas->BuildLegend();
}

const std::string epochToReadable(const long timestamp)
{
  std::string readableTime;
  time_t timeSeconds = timestamp / 1000;
  readableTime = std::string(asctime(localtime(&timeSeconds)));
  readableTime.pop_back();
  return readableTime;
}

std::vector<std::string> getAliases(const std::string& input, const o2::ccdb::CcdbApi& ccdbApi, const std::string& detectorName, const long timestamp)
{
  std::vector<std::string> aliases;
  std::string alias;
  std::ifstream file(input);

  if (file.is_open()) {
    LOGP(info, "Parsing input file '{}'", input);
    while (std::getline(file, alias)) {
      if (!alias.empty()) {
        aliases.push_back(alias);
      }
    }
  } else if (!input.empty()) {
    LOGP(info, "Parsing input string '{}'", input);
    std::stringstream ss(input);
    while (std::getline(ss, alias, ';')) {
      aliases.push_back(alias);
    }
  } else {
    // Input string empty, fetching all datapoints
    LOGP(info, "Data point input empty, fetching data point definitions from CCDB.");
    std::string ccdbPath = detectorName + "/Config/DCSDPconfig";
    std::map<std::string, std::string> metadata;
    std::unordered_map<o2::dcs::DataPointIdentifier, std::string>* dpConfig = ccdbApi.retrieveFromTFileAny<std::unordered_map<o2::dcs::DataPointIdentifier, std::string>>(ccdbPath, metadata, timestamp);
    if (dpConfig) {
      for (auto& it : *dpConfig) {
        aliases.push_back(it.first.get_alias());
      }
    } else {
      LOGP(fatal, "No data point input provided, and can't fetch data point definitions from {}/{} for timestamp {}", ccdbApi.getURL(), ccdbPath, timestamp);
    }
  }
  return aliases;
}
