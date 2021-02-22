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
#include "CCDB/CCDBTimeStampUtils.h"
#include <string>
#include <map>
#include <unordered_map>
#include <memory>

// #include <FairLogger.h>

namespace o2::ccdb
{

/// A simple class offering simplified access to CCDB (mainly for MC simulation)
/// The class encapsulates timestamp and URL and is easily usable from detector code.
///
/// The CDBManager allowing caching of retrieved objects is by definition not thread safe,
/// therefore, to provide a possibility of multithread processing, one should foresee possibility
/// of multiple instances of the manager. CCDBManagerInstance serves to this purpose
///
/// In cases where caching is not needed or just 1 instance of the manager is enough, one case use
/// a singleton version BasicCCDBManager

class CCDBManagerInstance
{
  struct CachedObject {
    std::shared_ptr<void> objPtr;
    std::string uuid;
    long startvalidity = 0;
    long endvalidity = 0;
    bool isValid(long ts) { return ts < endvalidity && ts > startvalidity; }
  };

 public:
  CCDBManagerInstance(std::string const& path) : mCCDBAccessor{}
  {
    mCCDBAccessor.init(path);
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
  std::string const& getURL() const { return mCCDBAccessor.getURL(); }

  /// query timestamp
  long getTimestamp() const { return mTimestamp; }

  /// retrieve an object of type T from CCDB as stored under path and timestamp
  template <typename T>
  T* getForTimeStamp(std::string const& path, long timestamp);

  /// retrieve an object of type T from CCDB as stored under path; will use the timestamp member
  template <typename T>
  T* get(std::string const& path)
  {
    // TODO: add some error info/handling when failing
    return getForTimeStamp<T>(path, mTimestamp);
  }

  bool isHostReachable() const { return mCCDBAccessor.isHostReachable(); }

  /// clear all entries in the cache
  void clearCache() { mCache.clear(); }

  /// clear particular entry in the cache
  void clearCache(std::string const& path) { mCache.erase(path); }

  /// check if caching is enabled
  bool isCachingEnabled() const { return mCachingEnabled; }

  /// disable or enable caching
  void setCaching(bool v)
  {
    mCachingEnabled = v;
    if (!v) {
      clearCache();
    }
  }

  /// check if checks of object validity before CCDB query is enabled
  bool isLocalObjectValidityCheckingEnabled() const { return mCheckObjValidityEnabled; }

  /// set the flag to check object validity before CCDB query
  void setLocalObjectValidityChecking(bool v = true) { mCheckObjValidityEnabled = v; }

  /// set the object upper validity limit
  void setCreatedNotAfter(long v) { mCreatedNotAfter = v; }

  /// get the object upper validity limit
  long getCreatedNotAfter() const { return mCreatedNotAfter; }

  /// reset the object upper validity limit
  void resetCreatedNotAfter() { mCreatedNotAfter = 0; }

  /// set the object upper validity limit
  void setCreatedNotBefore(long v) { mCreatedNotBefore = v; }

  /// get the object upper validity limit
  long getCreatedNotBefore() const { return mCreatedNotBefore; }

  /// reset the object upper validity limit
  void resetCreatedNotBefore() { mCreatedNotBefore = 0; }

 private:
  // we access the CCDB via the CURL based C++ API
  o2::ccdb::CcdbApi mCCDBAccessor;
  std::unordered_map<std::string, CachedObject> mCache; //! map for {path, CachedObject} associations
  std::map<std::string, std::string> mMetaData;         // some dummy object needed to talk to CCDB API
  std::map<std::string, std::string> mHeaders;          // headers to retrieve tags
  long mTimestamp{o2::ccdb::getCurrentTimestamp()};     // timestamp to be used for query (by default "now")
  bool mCanDefault = false;                             // whether default is ok --> useful for testing purposes done standalone/isolation
  bool mCachingEnabled = true;                          // whether caching is enabled
  bool mCheckObjValidityEnabled = false;                // wether the validity of cached object is checked before proceeding to a CCDB API query
  long mCreatedNotAfter = 0;                            // upper limit for object creation timestamp (TimeMachine mode) - If-Not-After HTTP header
  long mCreatedNotBefore = 0;                           // lower limit for object creation timestamp (TimeMachine mode) - If-Not-Before HTTP header
};

template <typename T>
T* CCDBManagerInstance::getForTimeStamp(std::string const& path, long timestamp)
{
  if (!isCachingEnabled()) {
    return mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp, nullptr, "",
                                                 mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                                                 mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
  }
  auto& cached = mCache[path];
  if (mCheckObjValidityEnabled && cached.isValid(timestamp)) {
    return reinterpret_cast<T*>(cached.objPtr.get());
  }

  T* ptr = mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp, &mHeaders, cached.uuid,
                                                 mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                                                 mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
  if (ptr) { // new object was shipped, old one (if any) is not valid anymore
    cached.objPtr.reset(ptr);
    cached.uuid = mHeaders["ETag"];
    cached.startvalidity = std::stol(mHeaders["Valid-From"]);
    cached.endvalidity = std::stol(mHeaders["Valid-Until"]);
  } else if (mHeaders.count("Error")) { // in case of errors the pointer is 0 and headers["Error"] should be set
    clearCache(path);                   // in case of any error clear cache for this object
  } else {                              // the old object is valid
    ptr = reinterpret_cast<T*>(cached.objPtr.get());
  }
  mHeaders.clear();
  return ptr;
}

class BasicCCDBManager : public CCDBManagerInstance
{
 public:
  static BasicCCDBManager& instance()
  {
    const std::string ccdbUrl{"http://ccdb-test.cern.ch:8080"};
    static BasicCCDBManager inst{ccdbUrl};
    return inst;
  }

 private:
  using CCDBManagerInstance::CCDBManagerInstance;
};

} // namespace o2::ccdb

#endif //O2_BASICCCDBMANAGER_H
