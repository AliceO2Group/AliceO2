// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITCalibration/FITObjectViewer.h"

using namespace o2::calibration::fit;

void FITObjectViewer::_emplaceResultIntoContainers(std::vector<std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TObject>>>&& result)
{
  for( auto&[infoObject, visObject] : result ){
    if(visObject){
      mInfoVector.emplace_back(std::move(infoObject));
      mVisualizationObjects.emplace_back(std::move(visObject));
    }
  }
}

void FITObjectViewer::clear()
{
  mInfoVector.clear();
  mVisualizationObjects.clear();
}