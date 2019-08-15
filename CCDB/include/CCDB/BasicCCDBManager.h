//
// Created by Sandro Wenzel on 2019-08-14.
//

#ifndef O2_MCSIMMANAGER_H
#define O2_MCSIMMANAGER_H

#include "CCDB/CcdbApi.h"
#include <string>
#include <map>

namespace o2::ccdb {

/// A singleton class offering access to CCDB (mainly for MC simulation)
class BasicCCDBManager
{
 public:

  static BasicCCDBManager& instance()
  {
      const std::string ccdbUrl{"http://ccdb-test.cern.ch:8080"};
    static BasicCCDBManager inst{ccdbUrl};
    return inst;
  }


  /// retrieve an object of type T from CCDB as stored under path
  template <typename T>
  T* get(std::string const& path) const;


 private:
  BasicCCDBManager(std::string const& path) : mCCDBAccessor{} {
      mCCDBAccessor.init(path);
  }

  // we access the CCDB via the CURL based API
  o2::ccdb::CcdbApi mCCDBAccessor;
  std::map<std::string, std::string> mMetaData;
};

template <typename T>
T* BasicCCDBManager::get(std::string const& path) const {
    mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData);
}

}


#endif //O2_MCSIMMANAGER_H
