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

#include "AnalysisDataModelHelpers.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/StringHelpers.h"
#include "Framework/Logger.h"

std::string str_tolower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); } // correct
  );
  return s;
}

namespace o2::aod::datamodel
{
std::string getTreeName(header::DataHeader dh)
{
  std::string description = std::string(dh.dataDescription.str);
  std::string origin = std::string(dh.dataOrigin.str);

  // lower case of first part of description
  auto found = description.find_first_of(":");
  std::string treeName = str_tolower(description).substr(0, found);

  // add prefix according to origin
  if (origin == "AOD") {
    treeName = "O2" + treeName;
  } else if (origin == "RN2") {
    treeName = "Run2" + treeName;
  }

  // exceptions from this
  if (origin == "AOD" && description == "MCCOLLISLABEL") {
    treeName = "O2mccollisionlabel";
  }

  return treeName;
}


} // namespace o2::aod::datamodel
