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

#ifndef O2_MCSIMMANAGER_H
#define O2_MCSIMMANAGER_H

#include "CCDB/CcdbApi.h"
#include <string>
#include <map>
#include <FairLogger.h>

namespace o2::ccdb
{

/// A simple singleton class offering generic access to CCDB (mainly for MC simulation)
class BasicCCDBManager
{
 public:
  static BasicCCDBManager& instance()
  {
    const std::string ccdbUrl{"http://ccdb-test.cern.ch:8080"};
    static BasicCCDBManager inst{ccdbUrl};
    return inst;
  }

  /// retrieve an object of type T from CCDB as stored under path and timestamp
  template <typename T>
  T* getForTimeStamp(std::string const& path, long timestamp) const;

  /// retrieve an object of type T from CCDB as stored under path and timestamp
  /// if there is not object, the function can create one on the fly by calling the constructor of class T with the provided arguments
  template <typename T, typename... Args>
  T* getOrDefaultForTimestamp(std::string const& path, long timestamp, Args&&...) const;

  /// retrieve an object of type T from CCDB as stored under path; will use timestamp member
  template <typename T>
  T* get(std::string const& path) const
  {
    return getForTimeStamp<T>(path, mTimestamp);
  }

  /// retrieve an object of type T from CCDB as stored under path; will use timestamp member
  /// if there is not object, the function can create one on the fly by calling the constructor of class T with the provided arguments
  template <typename T, typename... Args>
  T* getOrDefault(std::string const& path, Args&&... args) const
  {
    return getOrDefaultForTimestamp<T>(path, mTimestamp, args...);
  }

  void setCanDefault(bool d) { mCanDefault = d; }

  /// set timestamp cache for all queries
  void setTimestamp(unsigned long t) { mTimestamp = t; }

 private:
  BasicCCDBManager(std::string const& path) : mCCDBAccessor{}
  {
    mCCDBAccessor.init(path);
  }

  // we access the CCDB via the CURL based API
  o2::ccdb::CcdbApi mCCDBAccessor;
  std::map<std::string, std::string> mMetaData;
  long mTimestamp = 0;
  bool mCanDefault = false; // whether default is ok --> useful for testing purposes done standalone/isolation
};

template <typename T>
T* BasicCCDBManager::getForTimeStamp(std::string const& path, long timestamp) const
{
  return mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp);
}

template <typename T, typename... Args>
T* BasicCCDBManager::getOrDefaultForTimestamp(std::string const& path, long timestamp, Args&&... args) const
{
  T* obj = getForTimeStamp<T>(path, timestamp);
  if (!obj && mCanDefault) {
    // NOTIFY ABOUT DEFAULT
    LOG(WARN) << " DEFAULTING AN OBJECT FROM CCDB";
    // return a default obj
    return new T(std::forward<Args>(args)...);
  }
  return obj;
}
} // namespace o2::ccdb

#endif //O2_MCSIMMANAGER_H
