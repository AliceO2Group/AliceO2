// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "DataSet.h"

#include <gpucf/common/log.h>
#include <gpucf/errors/FileErrors.h>

#include <filesystem/path.h>

#include <fstream>

using namespace gpucf;
namespace fs = filesystem;

void DataSet::read(const fs::path& file)
{
  log::Info() << "Reading file " << file;

  if (!file.exists()) {
    throw FileNotFoundError(file);
  }

  objs.clear();

  std::ifstream in(file.str());

  for (std::string line;
       std::getline(in, line);) {
    nonstd::optional<Object> obj = Object::tryParse(line);

    if (obj.has_value()) {
      objs.push_back(*obj);
    }
  }
}

void DataSet::write(const fs::path& file) const
{
  log::Info() << "Writing to file " << file;

  std::ofstream out(file.str());

  for (const Object& obj : objs) {
    out << obj.str() << "\n";
  }
}

std::vector<Object> DataSet::get() const
{
  return objs;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
