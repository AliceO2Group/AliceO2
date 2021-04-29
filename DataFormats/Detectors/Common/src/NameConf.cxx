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
#include <fmt/format.h>
#include <memory>

using namespace o2::base;
using DId = o2::detectors::DetID;

// Check if the path exists
bool NameConf::pathExists(const std::string_view p)
{
  return o2::utils::pathExists(p);
}

// Check if the path is a directory
bool NameConf::pathIsDirectory(const std::string_view p)
{
  return o2::utils::pathIsDirectory(p);
}

std::string NameConf::getFullPath(const std::string_view p)
{
  return o2::utils::getFullPath(p);
}

std::string NameConf::rectifyDirectory(const std::string& _dir)
{
  return o2::utils::rectifyDirectory(_dir);
}

// Filename to store geometry file
std::string NameConf::getGeomFileName(const std::string_view prefix)
{
  // check if the prefix is an existing path
  if (pathIsDirectory(prefix)) {
    return o2::utils::concat_string(prefix, "/", STANDARDSIMPREFIX, "_", GEOM_FILE_STRING, ".root");
  } else if (pathExists(prefix)) {
    return std::string(prefix); // it is a full file
  }
  return o2::utils::concat_string(prefix.empty() ? STANDARDSIMPREFIX : prefix, "_", GEOM_FILE_STRING, ".root");
}

// Filename to store simulation cuts/process summary
std::string NameConf::getCutProcFileName(std::string_view prefix)
{
  // check if the prefix is an existing path
  if (pathIsDirectory(prefix)) {
    return o2::utils::concat_string(prefix, "/", STANDARDSIMPREFIX, "_", CUT_FILE_STRING, ".dat");
  } else if (pathExists(prefix)) {
    return std::string(prefix); // it is a full file
  }
  return o2::utils::concat_string(prefix.empty() ? STANDARDSIMPREFIX : prefix, "_", CUT_FILE_STRING, ".dat");
}

// Filename to store ITSMFT dictionary
std::string NameConf::getAlpideClusterDictionaryFileName(DId det, const std::string_view prefix, const std::string_view ext)
{
  // check if the prefix is an existing path
  if (pathIsDirectory(prefix)) {
    return o2::utils::concat_string(prefix, "/", det.getName(), DICTFILENAME, ext);
  } else if (pathExists(prefix)) {
    return std::string(prefix); // it is a full file
  }
  return o2::utils::concat_string(prefix, det.getName(), DICTFILENAME, ext);
}

// Filename to store material LUT file
std::string NameConf::getMatLUTFileName(const std::string_view prefix)
{
  // check if the prefix is an existing path
  if (pathIsDirectory(prefix)) {
    return o2::utils::concat_string(prefix, "/", MATBUDLUT, ".root");
  } else if (pathExists(prefix)) {
    return std::string(prefix); // it is a full file
  }
  return o2::utils::concat_string(prefix, MATBUDLUT, ".root");
}

std::string NameConf::getCTFFileName(uint32_t run, uint32_t orb, uint32_t id, const std::string_view prefix)
{
  return o2::utils::concat_string(prefix, '_', fmt::format("run{:08d}_orbit{:010d}_tf{:010d}", run, orb, id), ".root");
}

std::string NameConf::getCTFDictFileName()
{
  return o2::utils::concat_string(CTFDICT, ".root");
}
