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

#include "CommonUtils/NameConf.h"
#include <fmt/format.h>
#include <memory>

O2ParamImpl(o2::base::NameConf);

using namespace o2::base;

std::string NameConf::buildFileName(const std::string_view prefix, const std::string_view delimiter, const std::string_view defPrefix, const std::string_view defName,
                                    const std::string_view extension, const std::string_view optDir)
{
  if (o2::utils::Str::pathIsDirectory(prefix)) { // if path is directory, just add to default name, ignoring optional directory optDir argument
    return o2::utils::Str::concat_string(prefix, "/", defPrefix, delimiter, defName, '.', extension);
  } else if (!prefix.empty() && o2::utils::Str::pathExists(prefix)) { // explicit file path is provided, use it directly
    return std::string(prefix);
  }
  auto dir = o2::utils::Str::rectifyDirectory(optDir); // directory might have been provided
  // is the prefix really prefix or a file-name
  if (!prefix.empty()) {
    auto path = o2::utils::Str::concat_string(dir, prefix);
    if (o2::utils::Str::pathExists(path)) {
      return path;
    }
  }
  return o2::utils::Str::concat_string(dir, prefix.empty() ? defPrefix : prefix, delimiter, defName, '.', extension);
}

// Filename to store geometry file
std::string NameConf::getGeomFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "_", STANDARDSIMPREFIX, GEOM_FILE_STRING, ROOT_EXT_STRING, Instance().mDirGeom);
}

// Filename to store geometry file
std::string NameConf::getAlignedGeomFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "_", STANDARDSIMPREFIX, ALIGNEDGEOM_FILE_STRING, ROOT_EXT_STRING, Instance().mDirGeom);
}

// Filename to store general run parameters (GRP)
std::string NameConf::getCollisionContextFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "", "", COLLISIONCONTEXT, ROOT_EXT_STRING, Instance().mDirCollContext);
}

// Filename to store general run parameters (GRP)
std::string NameConf::getGRPFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "_", STANDARDSIMPREFIX, GRP_STRING, ROOT_EXT_STRING, Instance().mDirGRP);
}

// Filename to store general run parameters from ECS (GRPECS)
std::string NameConf::getGRPECSFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "_", STANDARDSIMPREFIX, GRPECS_STRING, ROOT_EXT_STRING, Instance().mDirGRPECS);
}

// Filename to store general run parameters from LHCIF (GRPLHCIF)
std::string NameConf::getGRPLHCIFFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "_", STANDARDSIMPREFIX, GRPLHCIF_STRING, ROOT_EXT_STRING, Instance().mDirGRPLHCIF);
}

// Filename to store general run parameters from ECS (GRPECS)
std::string NameConf::getGRPMagFieldFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "_", STANDARDSIMPREFIX, GRPMAGFIELD_STRING, ROOT_EXT_STRING, Instance().mDirGRPMagField);
}

// Filename to store simulation cuts/process summary
std::string NameConf::getCutProcFileName(std::string_view prefix)
{
  return buildFileName(prefix, "_", STANDARDSIMPREFIX, CUT_FILE_STRING, DAT_EXT_STRING);
}

// Filename to store material LUT file
std::string NameConf::getMatLUTFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "", "", MATBUDLUT, ROOT_EXT_STRING, Instance().mDirMatLUT);
}

std::string NameConf::getCTFFileName(uint32_t run, uint32_t orb, uint32_t id, const std::string& host, const std::string_view prefix)
{
  return o2::utils::Str::concat_string(prefix, '_', fmt::format("run{:08d}_orbit{:010d}_tf{:010d}_{}", run, orb, id, host), ".root");
}

std::string NameConf::getCTFDictFileName()
{
  return o2::utils::Str::concat_string(CTFDICT, ".root");
}

// Filename to store general run parameters (GRP)
std::string NameConf::getTFIDInfoFileName(const std::string_view prefix)
{
  return buildFileName(prefix, "_", "o2", TFIDINFO, ROOT_EXT_STRING, Instance().mDirTFIDINFO);
}

// Default CCDB server
std::string NameConf::getCCDBServer()
{
  return Instance().mCCDBServer;
}
