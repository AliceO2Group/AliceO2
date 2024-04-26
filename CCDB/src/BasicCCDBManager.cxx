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
// Created by Sandro Wenzel on 2019-08-14.
//
#include "CCDB/BasicCCDBManager.h"
#include <boost/lexical_cast.hpp>
#include <fairlogger/Logger.h>
#include <string>

namespace o2
{
namespace ccdb
{

void CCDBManagerInstance::setURL(std::string const& url)
{
  mCCDBAccessor.init(url);
}

void CCDBManagerInstance::reportFatal(std::string_view err)
{
  LOG(fatal) << err;
}

std::pair<int64_t, int64_t> CCDBManagerInstance::getRunDuration(o2::ccdb::CcdbApi const& api, int runnumber, bool fatal)
{
  auto response = api.retrieveHeaders("RCT/Info/RunInformation", std::map<std::string, std::string>(), runnumber);
  if (response.size() != 0) {
    std::string report{};
    auto strt = response.find("STF");
    auto stop = response.find("ETF");
    long valStrt = (strt == response.end()) ? -1L : boost::lexical_cast<int64_t>(strt->second);
    long valStop = (stop == response.end()) ? -1L : boost::lexical_cast<int64_t>(stop->second);
    if (valStrt < 0 || valStop < 0) {
      report += "Missing STF/EFT -> use SOX/EOX;";
      strt = response.find("SOX");
      valStrt = (strt == response.end()) ? -1L : boost::lexical_cast<int64_t>(strt->second);
      if (valStrt < 1) {
        report += fmt::format(" Missing/invalid SOX -> use SOR");
        strt = response.find("SOR");
        valStrt = (strt == response.end()) ? -1L : boost::lexical_cast<int64_t>(strt->second);
      }
      stop = response.find("EOX");
      valStop = (stop == response.end()) ? -1L : boost::lexical_cast<int64_t>(stop->second);
      if (valStop < 1) {
        report += fmt::format(" | Missing/invalid EOX -> use EOR");
        stop = response.find("EOR");
        valStop = (stop == response.end()) ? -1L : boost::lexical_cast<int64_t>(stop->second);
      }
      if (!report.empty()) {
        LOGP(warn, "{}", report);
      }
    }
    if (valStrt > 0 && valStop >= valStrt) {
      return std::make_pair(valStrt, valStop);
    }
  }
  // failure
  if (fatal) {
    LOG(fatal) << "Empty, missing or invalid response from query to RCT/Info/RunInformation for run " << runnumber;
  }
  return std::make_pair(-1L, -1L);
}

std::pair<int64_t, int64_t> CCDBManagerInstance::getRunDuration(int runnumber, bool fatal)
{
  mQueries++;
  if (!isCachingEnabled()) {
    return CCDBManagerInstance::getRunDuration(mCCDBAccessor, runnumber, fatal);
  }
  auto& cached = mCache["RCT-Run-Info HeaderOnly"];
  std::pair<int64_t, int64_t> rd;
  cached.queries++;
  if (cached.startvalidity != runnumber) { // need to fetch
    rd = CCDBManagerInstance::getRunDuration(mCCDBAccessor, runnumber, fatal);
    cached.objPtr = std::make_shared<std::pair<int64_t, int64_t>>(rd);
    cached.startvalidity = runnumber;
    cached.endvalidity = runnumber + 1;
    cached.minSize = cached.maxSize = 0;
    cached.fetches++;
  } else {
    rd = *reinterpret_cast<std::pair<int64_t, int64_t>*>(cached.objPtr.get());
  }
  return rd;
}

std::string CCDBManagerInstance::getSummaryString() const
{
  std::string res = fmt::format("{} queries, {} bytes", mQueries, fmt::group_digits(mFetchedSize));
  if (mCachingEnabled) {
    res += fmt::format(" for {} objects", mCache.size());
  }
  res += fmt::format(", {} good fetches (and {} failed ones", mFetches, mFailures);
  if (mCachingEnabled && mFailures) {
    int nfailObj = 0;
    for (const auto& obj : mCache) {
      if (obj.second.failures) {
        nfailObj++;
      }
    }
    res += fmt::format(" for {} objects", nfailObj);
  }
  res += fmt::format(") in {} ms, instance: {}", fmt::group_digits(mTimerMS), mCCDBAccessor.getUniqueAgentID());
  return res;
}

void CCDBManagerInstance::report(bool longrep)
{
  LOG(info) << "CCDBManager summary: " << getSummaryString();
  if (longrep && mCachingEnabled) {
    LOGP(info, "CCDB cache miss/hit/failures");
    for (const auto& obj : mCache) {
      LOGP(info, "  {}: {}/{}/{} ({}-{} bytes)", obj.first, obj.second.fetches, obj.second.queries - obj.second.fetches - obj.second.failures, obj.second.failures, obj.second.minSize, obj.second.maxSize);
    }
  }
}

void CCDBManagerInstance::endOfStream()
{
  report(true);
}

} // namespace ccdb
} // namespace o2
