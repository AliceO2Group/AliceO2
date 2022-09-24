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

/// \file CreateLinkToHCIDMapping.C
/// \brief Create CCDB object required by TRD raw reader of mapping link indices to half-chamber IDs

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "FairLogger.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbObjectInfo.h"

#include <chrono>
#include <map>

#endif

using namespace o2::trd;

void createLinkToHCIDMapping(bool isDefault = true)
{

  LinkToHCIDMapping mapping;
  for (int i = 0; i < constants::MAXHALFCHAMBER; ++i) {
    mapping.linkIDToHCID.insert({i, HelperMethods::getHCIDFromLinkID(i)});
    mapping.hcIDToLinkID.insert({i, HelperMethods::getLinkIDfromHCID(i)});
  }

  if (!isDefault) {
    // in case we have swapped links, they should be manually entered here
    // the default map should correspond to the foreseen mapping
    mapping.swapLinks(139, 175);
    mapping.swapLinks(170, 171);
    mapping.swapLinks(156, 162); // To be confirmed, only saw data from HCID 166 so far (corresponding to link 162 with default mapping)
    mapping.swapLinks(584, 590);
    mapping.swapLinks(644, 650);
    mapping.swapLinks(665, 671);
    mapping.swapLinks(675, 681); // To be confirmed, only saw data from HCID 677 so far (corresponding to link 681 with default mapping)
    mapping.swapLinks(728, 729);
    mapping.swapLinks(794, 800);
    mapping.swapLinks(855, 856);
    mapping.swapLinks(858, 861);
  }

  if (!mapping.isOK()) {
    LOG(error) << "Failed to create proper mapping. Not creating CCDB object";
    return;
  }

  o2::ccdb::CcdbApi ccdb;
  ccdb.init("https://alice-ccdb.cern.ch");
  // ccdb.init("http://ccdb-test.cern.ch:8080");

  std::map<std::string, std::string> metadata;
  metadata.emplace(std::make_pair("UploadedBy", "marten")); // FIXME change to the name of the acual uploader
  metadata.emplace(std::make_pair("Description", "Mapping from Link ID to HCID and vice versa"));
  metadata.emplace(std::make_pair("Responsible", "Ole Schmidt (ole.schmidt@cern.ch)"));

  if (isDefault) {
    metadata.emplace(std::make_pair("default", "true"));
  }
  uint64_t timeStampStart = (isDefault) ? 1UL : 1577833200000UL;        //  00:00 2020-01-01 in case of the non-default map
  uint64_t timeStampEnd = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; // the mapping is always valid until changed manually, at which point we upload a new objected which will take precedence

  ccdb.storeAsTFileAny(&mapping, "TRD/Config/LinkToHCIDMapping", metadata, timeStampStart, timeStampEnd);
}
