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

//
// Created by Sandro Wenzel on 2019-08-20.
//
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include <chrono>
#include <ctime>

namespace o2::ccdb
{

long getFutureTimestamp(int secondsInFuture)
{
  std::chrono::seconds sec(secondsInFuture);
  auto future = std::chrono::system_clock::now() + sec;
  auto future_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(future);
  auto epoch = future_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
  return value.count();
}

/// returns the timestamp in long corresponding to "now"
long getCurrentTimestamp()
{
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto epoch = now_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
  return value.count();
}

/// \brief Converting time into numerical time stamp representation
long createTimestamp(int year, int month, int day, int hour, int minutes, int seconds)
{
  struct tm timeinfo;
  timeinfo.tm_year = year;
  timeinfo.tm_mon = month;
  timeinfo.tm_mday = day;
  timeinfo.tm_hour = hour;
  timeinfo.tm_min = minutes;
  timeinfo.tm_sec = seconds;

  time_t timeformat = mktime(&timeinfo);
  return static_cast<long>(timeformat);
}

int adjustOverriddenEOV(CcdbApi& api, const CcdbObjectInfo& infoNew)
{
  int res = 0;
  if (infoNew.getStartValidityTimestamp() > 0) {
    std::map<std::string, std::string> dummyMD, prevHeader;
    dummyMD[o2::ccdb::CcdbObjectInfo::AdjustableEOV] = "true";
    prevHeader = api.retrieveHeaders(infoNew.getPath(), dummyMD, infoNew.getStartValidityTimestamp() - 1); // is there an adjustable object to override?
    const auto itETag = prevHeader.find("ETag");
    if (itETag != prevHeader.end() &&
        prevHeader.find(o2::ccdb::CcdbObjectInfo::AdjustableEOV) != prevHeader.end() &&
        prevHeader.find(o2::ccdb::CcdbObjectInfo::DefaultObj) == prevHeader.end()) {
      std::string etag = itETag->second;
      etag.erase(remove(etag.begin(), etag.end(), '\"'), etag.end());
      LOGP(info, "Adjusting EOV of previous {}/{}/{} to {} (id:{})", infoNew.getPath(), prevHeader["Valid-From"], prevHeader["Valid-Until"], infoNew.getStartValidityTimestamp() - 1, etag);
      // equivalent of std::string cmd = fmt::format("sh -c \"curl -X PUT {}{}{}/{}/{}\"", api.getURL(), api.getURL().back() == '/' ? "" : "/", infoNew.getPath(), infoNew.getStartValidityTimestamp() - 1, infoNew.getStartValidityTimestamp());
      api.updateMetadata(infoNew.getPath(), {}, infoNew.getStartValidityTimestamp() - 1, etag, infoNew.getStartValidityTimestamp());
    }
  }
  return res;
}

} // namespace o2::ccdb
