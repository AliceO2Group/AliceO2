// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AODNative/ByteSource.h"
#include "AODNative/Errors.h"
#include "LocalByteSource.h"
#ifdef AODNATIVE_WITH_XROOTD
#include "XRootDByteSource.h"
#endif
#ifdef AODNATIVE_WITH_CURL
#include "HttpByteSource.h"
#endif

namespace o2::aod::native
{

ByteSource::~ByteSource() = default;

std::unique_ptr<ByteSource> ByteSource::open(std::string url, const Options& options)
{
  if (url.starts_with("file://")) {
    return std::make_unique<LocalByteSource>(url.substr(7), options);
  }
  for (auto scheme : {"root://", "roots://", "xroot://", "xroots://"}) {
    if (url.starts_with(scheme)) {
#ifdef AODNATIVE_WITH_XROOTD
      return std::make_unique<XRootDByteSource>(std::move(url), options);
#else
      throw IoError(url, "the native reader was built without XRootD");
#endif
    }
  }
  for (auto scheme : {"http://", "https://"}) {
    if (url.starts_with(scheme)) {
#ifdef AODNATIVE_WITH_CURL
      return std::make_unique<HttpByteSource>(std::move(url), options);
#else
      throw IoError(url, "the native reader was built without curl");
#endif
    }
  }
  // alien:// names are resolved to root:// replicas by the caller.
  for (auto scheme : {"alien://"}) {
    if (url.starts_with(scheme)) {
      throw IoError(url, "alien:// names must be resolved to a replica first");
    }
  }
  if (url.find("://") != std::string::npos) {
    throw IoError(url, "unknown URL scheme");
  }
  return std::make_unique<LocalByteSource>(std::move(url), options);
}

} // namespace o2::aod::native
