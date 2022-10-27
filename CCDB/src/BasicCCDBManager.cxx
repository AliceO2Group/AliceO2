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

std::pair<uint64_t, uint64_t> CCDBManagerInstance::getRunDuration(int runnumber) const
{
  auto response = mCCDBAccessor.retrieveHeaders("RCT/Info/RunInformation", std::map<std::string, std::string>(), runnumber);
  if (response.size() == 0 || response.find("SOR") == response.end() || response.find("EOR") == response.end()) {
    LOG(fatal) << "Empty or missing response from query to RCT/Info/RunInformation for run " << runnumber;
  }
  auto sor = boost::lexical_cast<uint64_t>(response["SOR"]);
  auto eor = boost::lexical_cast<uint64_t>(response["EOR"]);
  return std::make_pair(sor, eor);
}

std::string CCDBManagerInstance::getSummaryString() const
{
  std::string res = fmt::format("{} queries", mQueries);
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

void CCDBManagerInstance::endOfStream()
{
  LOG(info) << "CCDBManager summary: " << getSummaryString();
}

} // namespace ccdb
} // namespace o2
