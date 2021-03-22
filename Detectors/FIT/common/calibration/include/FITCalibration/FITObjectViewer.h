// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITOBJECTVIEWER_H
#define O2_FITOBJECTVIEWER_H

#include "FITViewObjectGenerator.h"
#include "Rtypes.h"

namespace o2::calibration::fit
{

class FITObjectViewer
{

 public:

  template <typename... ObjectsToVis>
  void generateViewObjects(ObjectsToVis&&... objectsToVis)
  {
    ( _emplaceResultIntoContainers(
       FITViewObjectGenerator::generateViewObjects(std::forward<ObjectsToVis>(objectsToVis))), ... );
  }

  [[nodiscard]] const std::vector<o2::ccdb::CcdbObjectInfo>& getInfoVector() const { return mInfoVector; }
  [[nodiscard]] const std::vector<std::shared_ptr<TObject>>& getObjectsVector() const { return mVisualizationObjects; }
  [[nodiscard]] std::vector<o2::ccdb::CcdbObjectInfo>& getInfoVector() { return mInfoVector; }
  void clear();

 private:
    void _emplaceResultIntoContainers(std::vector<std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TObject>>>&& result);

 private:
  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector;
  std::vector<std::shared_ptr<TObject>> mVisualizationObjects;


};




}



#endif //O2_FITOBJECTVIEWER_H
