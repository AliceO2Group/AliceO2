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

#ifndef O2_BASICCDBMANAGER_H
#define O2_BASICCDBMANAGER_H

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/NameConf.h"
#include <string>
#include <map>
#include <unordered_map>
#include <memory>

class TGeoManager; // we need to forward-declare those classes which should not be cleaned up

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
    void* noCleanupPtr = nullptr; // if assigned instead of objPtr, no cleanup will be done on exit (for global objects cleaned up by the root, e.g. gGeoManager)
    std::string uuid;
    long startvalidity = 0;
    long endvalidity = -1;
    bool isBlob = false; // is this raw blob or parsed object?
    bool isValid(long ts) { return ts < endvalidity && ts > startvalidity; }
    void clear()
    {
      objPtr.reset();
      uuid = "";
      startvalidity = 0;
      endvalidity = -1;
      isBlob = false;
    }
  };

 public:
  using BLOB = std::vector<char>;
  using MD = std::map<std::string, std::string>;

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

  /// retrieve an object of type T from CCDB as stored under path, timestamp and metaData
  template <typename T>
  T* getSpecific(std::string const& path, long timestamp = -1, MD metaData = MD())
  {
    // TODO: add some error info/handling when failing
    mMetaData = metaData;
    return getForTimeStamp<T>(path, timestamp);
  }

  /// retrieve an object of type T from CCDB as stored under path; will use the timestamp member
  template <typename T>
  T* get(std::string const& path)
  {
    return getForTimeStamp<T>(path, mTimestamp);
  }

  /// aliases for BLOB retrieval
  BLOB* getBlobForTimeStamp(std::string const& path, long timestamp) { return getForTimeStamp<BLOB>(path, timestamp); }
  BLOB* getSpecificBlob(std::string const& path, long timestamp = -1, MD metaData = MD()) { return getSpecific<BLOB>(path, timestamp, metaData); }
  BLOB* getBlob(std::string const& path) { return get<BLOB>(path); }

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

  /// get the fatalWhenNull state
  bool getFatalWhenNull() const { return mFatalWhenNull; }
  /// set the fatal property (when false; nullptr object responses will not abort)
  void setFatalWhenNull(bool b) { mFatalWhenNull = b; }

 private:
  // method to print (fatal) error
  void reportFatal(std::string_view s);
  BLOB* createBlob(std::string const& path,
                   MD const& metadata, long timestamp,
                   MD* headers, std::string const& etag,
                   const std::string& createdNotAfter, const std::string& createdNotBefore);
  // we access the CCDB via the CURL based C++ API
  o2::ccdb::CcdbApi mCCDBAccessor;
  std::unordered_map<std::string, CachedObject> mCache; //! map for {path, CachedObject} associations
  MD mMetaData;                                         // some dummy object needed to talk to CCDB API
  MD mHeaders;                                          // headers to retrieve tags
  long mTimestamp{o2::ccdb::getCurrentTimestamp()};     // timestamp to be used for query (by default "now")
  bool mCanDefault = false;                             // whether default is ok --> useful for testing purposes done standalone/isolation
  bool mCachingEnabled = true;                          // whether caching is enabled
  bool mCheckObjValidityEnabled = false;                // wether the validity of cached object is checked before proceeding to a CCDB API query
  long mCreatedNotAfter = 0;                            // upper limit for object creation timestamp (TimeMachine mode) - If-Not-After HTTP header
  long mCreatedNotBefore = 0;                           // lower limit for object creation timestamp (TimeMachine mode) - If-Not-Before HTTP header
  bool mFatalWhenNull = true;                           // if nullptr blob replies should be treated as fatal (can be set by user)

  ClassDefNV(CCDBManagerInstance, 1);
};

template <typename T>
T* CCDBManagerInstance::getForTimeStamp(std::string const& path, long timestamp)
{
  T* ptr = nullptr;
  if (!isCachingEnabled()) {
    if constexpr (std::is_same<T, BLOB>::value) {
      ptr = createBlob(path, mMetaData, timestamp, nullptr, "",
                       mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                       mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
    } else {
      ptr = mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp, nullptr, "",
                                                  mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                                                  mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
    }
    if (!ptr && mFatalWhenNull) {
      reportFatal(std::string("Got nullptr from CCDB for path ") + path + std::string(" and timestamp ") + std::to_string(timestamp));
    }
    return ptr;
  }
  auto& cached = mCache[path];
  if constexpr (std::is_same<T, BLOB>::value) { // check if cached object type is consistent with requested one
    if (!cached.isBlob) {
      cached.clear();
    }
  } else {
    if (cached.isBlob) {
      cached.clear();
    }
  }
  if (mCheckObjValidityEnabled && cached.isValid(timestamp)) {
    return reinterpret_cast<T*>(cached.noCleanupPtr ? cached.noCleanupPtr : cached.objPtr.get());
  }
  if constexpr (std::is_same<T, BLOB>::value) {
    ptr = createBlob(path, mMetaData, timestamp, &mHeaders, cached.uuid,
                     mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                     mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
  } else {
    ptr = mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp, &mHeaders, cached.uuid,
                                                mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                                                mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
  }
  if (ptr) { // new object was shipped, old one (if any) is not valid anymore
    if constexpr (std::is_same<TGeoManager, T>::value) { // some special objects cannot be cached to shared_ptr since root may delete their raw global pointer
      cached.noCleanupPtr = ptr;
    } else {
      cached.objPtr.reset(ptr);
    }
    if constexpr (std::is_same<T, BLOB>::value) {
      cached.isBlob = true;
    } else {
      cached.isBlob = false;
    }
    cached.uuid = mHeaders["ETag"];
    cached.startvalidity = std::stol(mHeaders["Valid-From"]);
    cached.endvalidity = std::stol(mHeaders["Valid-Until"]);
  } else if (mHeaders.count("Error")) { // in case of errors the pointer is 0 and headers["Error"] should be set
    clearCache(path);                   // in case of any error clear cache for this object
  } else {                              // the old object is valid
    ptr = reinterpret_cast<T*>(cached.noCleanupPtr ? cached.noCleanupPtr : cached.objPtr.get());
  }
  mHeaders.clear();
  mMetaData.clear();
  if (!ptr && mFatalWhenNull) {
    reportFatal(std::string("Got nullptr from CCDB for path ") + path + std::string(" and timestamp ") + std::to_string(timestamp));
  }
  return ptr;
}

class BasicCCDBManager : public CCDBManagerInstance
{
 public:
  static BasicCCDBManager& instance()
  {
    const std::string ccdbUrl{o2::base::NameConf::getCCDBServer()};
    static BasicCCDBManager inst{ccdbUrl};
    return inst;
  }

 private:
  using CCDBManagerInstance::CCDBManagerInstance;
};

} // namespace o2::ccdb

#endif //O2_BASICCCDBMANAGER_H
