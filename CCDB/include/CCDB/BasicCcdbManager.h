// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Created by Sandro Wenzel on 2019-08-14.
//

#ifndef O2_BASICCDBMANAGER_H
#define O2_BASICCDBMANAGER_H

#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbTimeStampUtils.h"
#include <string>
#include <map>
// #include <FairLogger.h>

namespace o2::ccdb
{

/// A simple (singleton) class offering simplified access to CCDB (mainly for MC simulation)
/// The class encapsulates timestamp and URL and is easily usable from detector code.
class BasicCcdbManager
{
 public:
  static BasicCcdbManager& instance()
  {
    const std::string ccdbUrl{"http://ccdb-test.cern.ch:8080"};
    static BasicCcdbManager inst{ccdbUrl};
    return inst;
  }

  /// set a URL to query from
  void setURL(const std::string& url);

  /// set timestamp cache for all queries
  void setTimestamp(long t)
  {
    if (t >= 0) {
      mTimestamp = t;
    }
  }

  /// query current URL
  std::string const& getURL() const { return mCcdbAccessor.getURL(); }

  /// query timestamp
  long getTimestamp() const { return mTimestamp; }

  /// retrieve an object of type T from CCDB as stored under path and timestamp
  template <typename T>
  T* getForTimeStamp(std::string const& path, long timestamp) const;

  /// retrieve an object of type T from CCDB as stored under path; will use the timestamp member
  template <typename T>
  T* get(std::string const& path) const
  {
    // TODO: add some error info/handling when failing
    return getForTimeStamp<T>(path, mTimestamp);
  }

  bool isHostReachable() const { return mCcdbAccessor.isHostReachable(); }

 private:
  BasicCcdbManager(std::string const& path) : mCcdbAccessor{}
  {
    mCcdbAccessor.init(path);
  }

  // we access the CCDB via the CURL based C++ API
  o2::ccdb::CcdbApi mCcdbAccessor;
  std::map<std::string, std::string> mMetaData;     // some dummy object needed to talk to CCDB API
  long mTimestamp{o2::ccdb::getCurrentTimestamp()}; // timestamp to be used for query (by default "now")
  bool mCanDefault = false;                         // whether default is ok --> useful for testing purposes done standalone/isolation
};

template <typename T>
T* BasicCcdbManager::getForTimeStamp(std::string const& path, long timestamp) const
{
  return mCcdbAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp);
}

} // namespace o2::ccdb

#endif //O2_BASICCCDBMANAGER_H
