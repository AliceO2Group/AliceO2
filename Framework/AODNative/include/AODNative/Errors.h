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

#ifndef O2_FRAMEWORK_AODNATIVE_ERRORS_H_
#define O2_FRAMEWORK_AODNATIVE_ERRORS_H_

#include <cstdint>
#include <stdexcept>
#include <string>

namespace o2::aod::native
{

/// Base of every error raised by the native reader. In "auto" mode, any of
/// these during open or metadata parsing makes the file fall back to ROOT.
struct FormatError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// A class version we have no hardcoded layout for.
struct UnsupportedVersion : FormatError {
  UnsupportedVersion(std::string className_, int version_, std::string file_)
    : FormatError("unsupported version " + std::to_string(version_) + " of class " + className_ + " in " + file_),
      className(std::move(className_)),
      version(version_),
      file(std::move(file_))
  {
  }
  std::string className;
  int version;
  std::string file;
};

/// A valid ROOT construct outside of the AO2D subset (split objects,
/// unknown leaf types, legacy compression, ...).
struct UnsupportedLayout : FormatError {
  explicit UnsupportedLayout(std::string what_)
    : FormatError("unsupported layout: " + what_),
      detail(std::move(what_))
  {
  }
  std::string detail;
};

/// Bytes which contradict themselves: truncated buffers, bad checksums,
/// sizes which do not add up.
struct CorruptData : FormatError {
  CorruptData(std::string file_, uint64_t offset_, std::string what_)
    : FormatError("corrupt data in " + file_ + " at offset " + std::to_string(offset_) + ": " + what_),
      file(std::move(file_)),
      offset(offset_),
      detail(std::move(what_))
  {
  }
  std::string file;
  uint64_t offset;
  std::string detail;
};

/// Failure of the underlying transport.
struct IoError : FormatError {
  IoError(std::string url_, std::string status_)
    : FormatError("I/O error on " + url_ + ": " + status_),
      url(std::move(url_)),
      status(std::move(status_))
  {
  }
  std::string url;
  std::string status;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_ERRORS_H_
