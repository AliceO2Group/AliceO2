// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <filesystem/path.h>

#include <stdexcept>
#include <string>

template <bool isDirectory>
class FileError : std::exception
{

 public:
  FileError(const filesystem::path& f)
    : file(f)
  {
    std::stringstream ss;
    ss << "Could not find " << ((isDirectory) ? "directory" : "file")
       << file.str() << ".";

    msg = ss.str();
  }

  const char* what() const noexcept override
  {
    return msg.c_str();
  }

 private:
  std::string msg;
  filesystem::path file;
};

using FileNotFoundError = FileError<false>;
using DirectoryNotFoundError = FileError<true>;

// vim: set ts=4 sw=4 sts=4 expandtab:
