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
#include "FairLogger.h"
#include <string>

namespace o2
{
namespace ccdb
{

// Create blob pointer from the vector<char> containing the CCDB file
CCDBManagerInstance::BLOB* CCDBManagerInstance::createBlob(std::string const& path, MD const& metadata, long timestamp, MD* headers, std::string const& etag,
                                                           const std::string& createdNotAfter, const std::string& createdNotBefore)
{
  o2::vector<char> v;
  mCCDBAccessor.loadFileToMemory(v, path, metadata, timestamp, headers, etag, createdNotAfter, createdNotBefore);
  if ((headers && headers->count("Error")) || !v.size()) {
    return nullptr;
  }
  // Do a copy to avoid changing the API of createBlob, at least for now.
  BLOB* b = new BLOB();
  b->reserve(v.size());
  std::copy(v.begin(), v.end(), b->end());
  return b;
}

void CCDBManagerInstance::setURL(std::string const& url)
{
  mCCDBAccessor.init(url);
}

void CCDBManagerInstance::reportFatal(std::string_view err)
{
  LOG(fatal) << err;
}

} // namespace ccdb
} // namespace o2
