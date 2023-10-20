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

///
/// \file    GeometryManager.cxx
/// \author  Jeremi Niedziela
/// \author  Julian Myrcha

#include "EventVisualisationBase/GeometryManager.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include <fairlogger/Logger.h>

#include <TFile.h>
#include <TGLViewer.h>
#include <TEveGeoShapeExtract.h>
#include <TEveManager.h>
#include <TEveProjectionManager.h>
#include <TSystem.h>

using namespace std;

namespace o2
{
namespace event_visualisation
{

GeometryManager& GeometryManager::getInstance()
{
  static GeometryManager instance;
  return instance;
}

TEveGeoShape* GeometryManager::getGeometryForDetector(string detectorName)
{

  // read geometry path from config file
  string geomPath = ConfigurationManager::getSimpleGeomR3Path();

  // load ROOT file with geometry
  TFile* f = TFile::Open(Form("%s/simple_geom_%s.root", geomPath.c_str(), detectorName.c_str()));
  if (!f) {
    LOGF(error, "GeometryManager::GetSimpleGeom -- no file with geometry found for: ", detectorName, "!");
    return nullptr;
  }
  LOGF(info, "GeometryManager::GetSimpleGeom for: ", detectorName, " from ",
       Form("%s/simple_geom_%s.root", geomPath.c_str(), detectorName.c_str()));

  auto geomShapreExtract = dynamic_cast<TEveGeoShapeExtract*>(f->Get(detectorName.c_str()));
  TEveGeoShape* geomShape = TEveGeoShape::ImportShapeExtract(geomShapreExtract);
  f->Close();

  geomShape->SetName(detectorName.c_str());

  // prepare geometry to be drawn including all children
  drawDeep(geomShape,
           ConfigurationManager::getInstance().getSettings().GetValue(Form("%s.color", detectorName.c_str()), -1),
           ConfigurationManager::getInstance().getSettings().GetValue(Form("%s.trans", detectorName.c_str()), -1),
           ConfigurationManager::getInstance().getSettings().GetValue(Form("%s.line.color", detectorName.c_str()), -1));

  gEve->GetDefaultGLViewer()->UpdateScene();

  return geomShape;
}

void GeometryManager::drawDeep(TEveGeoShape* geomShape, Color_t color, Char_t transparency, Color_t lineColor)
{
  if (geomShape->HasChildren()) {
    geomShape->SetRnrSelf(false);

    if (strcmp(geomShape->GetElementName(), "TPC_Drift_1") == 0) { // hack for TPC drift chamber
      geomShape->SetRnrSelf(kTRUE);
      if (color >= 0) {
        geomShape->SetMainColor(color);
      }
      if (lineColor >= 0) {
        geomShape->SetLineColor(lineColor);
        geomShape->SetLineWidth(1); // 0.1
        geomShape->SetDrawFrame(true);
      } else {
        geomShape->SetDrawFrame(false);
      }
      if (transparency >= 0) {
        geomShape->SetMainTransparency(transparency);
      }
    }

    for (TEveElement::List_i i = geomShape->BeginChildren(); i != geomShape->EndChildren(); ++i) {
      drawDeep(static_cast<TEveGeoShape*>(*i), color, transparency, lineColor);
    }
  } else {
    geomShape->SetRnrSelf(true);
    if (color >= 0) {
      geomShape->SetMainColor(color);
    }
    if (lineColor >= 0) {
      geomShape->SetLineColor(lineColor);
      geomShape->SetLineWidth(1); // 0.1
      geomShape->SetDrawFrame(true);
    } else {
      geomShape->SetDrawFrame(false);
    }
    if (transparency >= 0) {
      geomShape->SetMainTransparency(transparency);
    }

    if (strcmp(geomShape->GetElementName(), "PHOS_5") == 0) { // hack for PHOS module which is not installed
      geomShape->SetRnrSelf(false);
    }
  }
}

} // namespace event_visualisation
} // namespace o2
