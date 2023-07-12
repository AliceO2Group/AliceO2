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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsBase/MatLayerCylSet.h"
#include "Framework/Logger.h"
#include "CCDB/BasicCCDBManager.h"
#include <regex>
#endif

// Macro to extract layers covering selected radial range into the separate LUT file.

o2::base::MatLayerCylSet* extractLUTLayers(const o2::base::MatLayerCylSet* src, const std::string& outName, float rmin = 0., float rmax = 84., float toler = 1e-6)
{
  auto* cp = src->extractCopy(rmin, rmax, toler);
  if (!cp) {
    LOGP(error, "failed to extract layers for {} < r < {}", rmin, rmax);
  }
  if (outName.size()) {
    cp->writeToFile(outName);
  }
  return cp;
}

void extractLUTLayers(const std::string& fname, float rmin = 0., float rmax = 84., float toler = 1e-6)
{
  const auto src = o2::base::MatLayerCylSet::loadFromFile(fname);
  if (!src) {
    LOGP(error, "failed to open source LUT from {}", fname);
    return;
  }
  auto fnameOut = std::regex_replace(fname, std::regex(R"(.root)"), fmt::format("_r{}_{}.root", rmin, rmax));
  auto cp = extractLUTLayers(src, fnameOut, rmin, rmax, toler);
}

void extractLUTLayers(long timestamp, float rmin = 0., float rmax = 84., float toler = 1e-6)
{
  auto& mg = o2::ccdb::BasicCCDBManager::instance();
  mg.setTimestamp(timestamp);
  const auto src = o2::base::MatLayerCylSet::rectifyPtrFromFile(mg.get<o2::base::MatLayerCylSet>("GLO/Param/MatLUT"));
  if (!src) {
    LOGP(error, "failed to open load LUT from CCDB for timestamp {}", timestamp);
    return;
  }
  auto fnameOut = fmt::format("matbudLUT_ts{}_r{}_{}.root", timestamp, rmin, rmax);
  auto cp = extractLUTLayers(src, fnameOut, rmin, rmax, toler);
}
