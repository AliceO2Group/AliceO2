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

void rescaleLUT(o2::base::MatLayerCylSet* src, const std::string& outName)
{
  struct RescRange {
    float rMin, rMax, factor;
  };
  std::vector<RescRange> task = {
    // put here radial ranges in increasing order with corresponding factors to rescale
    {0.1f, 6.f, 1.05},  // e.g. rescale layers covering 0.1<r<6 by factor 1.05
    {30.f, 40.f, 1.15}, // e.g. rescale layers covering 30.f<r<40.f by factor 1.15
  };

  // check if there are no overlaps in ranges, to avoid double rescaling
  for (size_t il = 1; il < task.size(); il++) {
    short lmax, lmin;
    float rmin = task[il - 1].rMax, rmax = task[il].rMin;
    if (rmin > rmax) {
      LOGP(error, "rMax={:.2f} of range {} is larger then rMin={:.2f} of range {}, must be in increasing order", rmin, il - 1, rmax, il);
      return;
    }
    o2::base::Ray ray(std::max(src->getRMin(), rmin), 0., 0., std::min(src->getRMax(), rmax), 0., 0.);
    if (!src->getLayersRange(ray, lmin, lmax)) {
      LOGP(error, "No layers found for {:.2f} < r < {:.2f}", rmin, rmax);
      return;
    }
    if (lmin == lmax) {
      LOGP(error, "rMax={:.2f} of range {} and rMin={:.2f} of range {}, correspond to the same slice {} with {:.2f}<r<{:.2f}",
           rmin, il - 1, rmax, il, lmin, src->getLayer(lmin).getRMin(), src->getLayer(lmin).getRMax());
      return;
    }
  }

  for (size_t il = 0; il < task.size(); il++) {
    src->scaleLayersByR(task[il].rMin, task[il].rMax, task[il].factor);
  }
  if (outName.size()) {
    src->writeToFile(outName);
  }
}

void rescaleLUT(const std::string& fname)
{
  auto src = o2::base::MatLayerCylSet::loadFromFile(fname);
  if (!src) {
    LOGP(error, "failed to open source LUT from {}", fname);
    return;
  }
  auto fnameOut = std::regex_replace(fname, std::regex(R"(.root)"), "_rescaled.root");
  rescaleLUT(src, fnameOut);
}

void rescaleLUT(long timestamp = -1)
{
  auto& mg = o2::ccdb::BasicCCDBManager::instance();
  mg.setTimestamp(timestamp);
  auto src = o2::base::MatLayerCylSet::rectifyPtrFromFile(mg.get<o2::base::MatLayerCylSet>("GLO/Param/MatLUT"));
  if (!src) {
    LOGP(error, "failed to open load LUT from CCDB for timestamp {}", timestamp);
    return;
  }
  auto fnameOut = fmt::format("matbudLUT_ts{}_rescaled.root", timestamp);
  rescaleLUT(src, fnameOut);
}
