// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsCommonDataFormats/NameConf.h"
#include <sys/stat.h>

bool pathExists(const std::string_view p)
{
  struct stat buffer;
  return (stat(p.data(), &buffer) == 0);
}

using namespace o2::base;

// Filename to store geometry file
std::string NameConf::getGeomFileName(const std::string_view prefix)
{
  // check if the prefix is an existing path
  const bool prefixispath = pathExists(prefix);
  if (prefixispath) {
    return o2::utils::concat_string(prefix, "/", STANDARDSIMPREFIX, "_", GEOM_FILE_STRING, ".root");
  } else {
    return o2::utils::concat_string(prefix.empty() ? STANDARDSIMPREFIX : prefix, "_", GEOM_FILE_STRING, ".root");
  }
}
