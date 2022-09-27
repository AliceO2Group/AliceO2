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

/// \file makeDefaultBadChannelMap.C
/// \brief Macro for uploading default bad channel maps to CCDB
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "CCDB/CcdbApi.h"
#include "DataFormatsFIT/BadChannelMap.h"
#include "TFile.h"

#include <chrono>
#include <string>
#include <memory>

#endif

#include "Framework/Logger.h"

#include <boost/algorithm/string.hpp>

void makeDefaultBadChannelMap(std::string detectorName,
                              const std::string ccdbUrl = "http://localhost:8080",
                              const std::string fileName = "")
{
  boost::to_upper(detectorName);

  int nChannels;

  if (detectorName == "FT0") {
    nChannels = 212;
  } else if (detectorName == "FV0") {
    nChannels = 49;
  } else if (detectorName == "FDD") {
    nChannels = 19;
  } else {
    LOGP(fatal, "Invalid detector name provided: '{}'. Please use [FT0/FV0/FDD].", detectorName);
    return;
  }

  LOGP(info, "Creating default bad channel map for {}.", detectorName);

  o2::fit::BadChannelMap badChannelMap;

  for (int iChannel = 0; iChannel < nChannels; iChannel++) {
    badChannelMap.setChannelGood(iChannel, true);
  }

  if (!ccdbUrl.empty()) {
    const std::string ccdbPath = detectorName + "/Calib/BadChannelMap";
    std::map<std::string, std::string> metadata;
    metadata["default"] = "true";
    metadata["comment"] = "Default bad channel map, all channels are good.";
    LOGP(info, "Storing default bad channel map on {}/{}.", ccdbUrl, ccdbPath);
    o2::ccdb::CcdbApi api;
    api.init(ccdbUrl);
    api.storeAsTFileAny(&badChannelMap, ccdbPath, metadata, 1, 99999999999999);
  }

  if (!fileName.empty()) {
    LOGP(info, "Storing default bad channel map locally in {}.", fileName);
    std::unique_ptr<TFile> file(TFile::Open(fileName.c_str(), "RECREATE"));
    file->WriteObject(&badChannelMap, "badChannelMap");
  }
  return;
}
