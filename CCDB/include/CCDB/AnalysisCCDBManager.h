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
// Created by Nicolo' Jacazio on 2020-06-15.
//

#ifndef O2_ANALYSISCCDBMANAGER_H
#define O2_ANALYSISCCDBMANAGER_H

#include "CCDB/BasicCCDBManager.h"

namespace o2::ccdb
{

/// A simple (singleton) class extending the BasicCCDBManager in order to limit the number of CCDB API queries.
// This class is intended mostly for analysis purposes
class AnalysisCCDBManager
{
  struct CachedObject {
    long startvalidity = 0;
    long endvalidity = 0;
    bool isValid(long ts) { return ts < endvalidity && ts > startvalidity; }
  };

 public:
  static AnalysisCCDBManager& instance()
  {
    static AnalysisCCDBManager inst;
    return inst;
  }

  /// retrieve an object of type T from CCDB as stored under path and timestamp
  template <typename T>
  T* getForTimeStamp(std::string const& path, long timestamp);

 private:
  AnalysisCCDBManager() {}
  std::unordered_map<std::string, CachedObject> mCache; //! map for {path, CachedObject} associations
};

template <typename T>
T* AnalysisCCDBManager::getForTimeStamp(std::string const& path, long timestamp)
{
  auto& cached = mCache[path];
  if (cached.isValid(timestamp))
    return cached;
  BasicCCDBManager man = BasicCCDBManager::instance();
  T obj = man.getForTimeStamp<T>(path, timestamp);
  mCache[path].startvalidity = man.getLastObjValidityStart();
  mCache[path].endvalidity = man.getLastObjValidityEnd();
  return obj;
}

} // namespace o2::ccdb

#endif //O2_ANALYSISCCDBMANAGER_H
