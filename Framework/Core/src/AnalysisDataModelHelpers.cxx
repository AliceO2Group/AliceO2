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

#include "Framework/AnalysisDataModelHelpers.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisSupportHelpers.h"
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
  auto description = std::string(dh.dataDescription.str);
  auto iver = (float)dh.subSpecification;

  // lower case of first part of description
  auto found = description.find_first_of(':');
  std::string treeName = str_tolower(description).substr(0, found);
  if (iver > 0) {
    treeName += std::string{"_"}.append(std::string(2 - (int)std::log10(iver), '0')).append(std::to_string((int)iver));
  }

  // add prefix according to origin
  for (auto possibleOrigin : framework::AODOrigins) {
    if (dh.dataOrigin == possibleOrigin) {
      treeName = "O2" + treeName;
      break;
    }
  }

  // exceptions from this
  auto origin = std::string(dh.dataOrigin.str);
  if (origin == "AOD" && description == "MCCOLLISLABEL") {
    treeName = "O2mccollisionlabel";
  }

  return treeName;
}

} // namespace o2::aod::datamodel
