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

#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/MaterialManager.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "MCHGeometryCreator/Geometry.h"
#include "MCHGeometryMisAligner/MisAligner.h"
#include "MCHGeometryTest/Helpers.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "CCDB/CcdbApi.h"

#include "Math/GenVector/Cartesian3D.h"
#include "TGLRnrCtx.h"
#include "TGLViewer.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TH2F.h"
#include "TFile.h"
#include "TPRegexp.h"
#include "TVirtualPad.h"
#include <iostream>

namespace o2
{
namespace mch
{
namespace test
{

TGeoVolume* createAirVacuumCave(const char* name)
{
  // create the air medium (only used for the geometry test)
  auto& mgr = o2::base::MaterialManager::Instance();

  const int nAir = 4;
  Float_t aAir[nAir] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[nAir] = {6., 7., 8., 18.};
  Float_t wAir[nAir] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAirVacuum = 1.20479E-10;
  const int kID = 90; // to avoid conflicts with definitions of other MCH materials

  mgr.Mixture("MCH", kID, "Air", aAir, zAir, dAirVacuum, nAir, wAir);
  mgr.Medium("MCH", kID, "Air", kID,
             false, /* isvol */
             0,     /* ifield */
             -1.0,  /* fieldm */
             -1.0,  /* tmaxfd */
             -1.0,  /* stemax */
             -1.0,  /* deemax */
             -1.0,  /* epsil */
             -1.0 /* stmin */);
  return gGeoManager->MakeBox(name, gGeoManager->GetMedium("MCH_Air"), 2000.0, 2000.0, 3000.0);
}

void dump(std::ostream& out, const TGeoNode& n, int level, int maxdepth, std::string prefix)
{
  if (level >= maxdepth) {
    return;
  }

  if (level == 0) {
    out << n.GetName() << "\n";
  }

  if (level < maxdepth) {
    for (int i = 0; i < n.GetNdaughters(); i++) {
      TGeoNode* d = n.GetDaughter(i);
      if (i == n.GetNdaughters() - 1) {
        out << prefix + "└──" << d->GetName()
            << "\n";
        dump(out, *d, level + 1, maxdepth, prefix + "   ");
      } else {
        out << prefix + "├──" << d->GetName()
            << "\n";
        dump(out, *d, level + 1, maxdepth, prefix + "│  ");
      }
    }
  }
}

void showGeometryAsTextTree(const char* fromPath, int maxdepth, std::ostream& out)
{
  if (!gGeoManager) {
    return;
  }

  TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();

  if (strlen(fromPath)) {
    if (!nav->cd(fromPath)) {
      std::cerr << "Could not get path " << fromPath << "\n";
      return;
    }
  }

  TGeoNode* node = nav->GetCurrentNode();

  dump(out, *node, 0, maxdepth, "");
}

void createStandaloneGeometry()
{
  if (gGeoManager && gGeoManager->GetTopVolume()) {
    std::cerr << "Can only call this function with an empty geometry, i.e. gGeoManager==nullptr "
              << " or gGeoManager->GetTopVolume()==nullptr\n";
  }
  TGeoManager* g = new TGeoManager("MCH-ONLY", "ALICE MCH Standalone Geometry");
  TGeoVolume* top = createAirVacuumCave("cave");
  g->SetTopVolume(top);
  o2::mch::geo::createGeometry(*g, *top);
}

void addAlignableVolumes()
{
  if (!gGeoManager) {
    std::cerr << "gGeoManager == nullptr, must create a geometry first\n";
    return;
  }
  // If not closed, we need to close it
  if (!gGeoManager->IsClosed()) {
    gGeoManager->CloseGeometry();
  }
  // Then add the alignable volumes
  o2::mch::geo::addAlignableVolumes(*gGeoManager);
}

void zeroMisAlignGeometry(const std::string& ccdbHost, const std::string& fileName)
{
  // create a regular geometry
  createStandaloneGeometry();
  if (!gGeoManager) {
    std::cerr << "gGeoManager == nullptr, must create a geometry first\n";
    return;
  }
  // If not closed, we need to close it
  if (!gGeoManager->IsClosed()) {
    gGeoManager->CloseGeometry();
  }
  // Then add the alignable volumes
  addAlignableVolumes();

  std::vector<o2::detectors::AlignParam> params;

  // The misaligner
  o2::mch::geo::MisAligner aGMA;
  aGMA.misAlign(params);

  o2::detectors::DetID detMCH("MCH");

  long tmin = 0;
  long tmax = -1;
  const std::string& objectPath = "";

  if (!ccdbHost.empty()) {
    std::string path = objectPath.empty() ? o2::base::DetectorNameConf::getAlignmentPath(detMCH) : objectPath;
    LOGP(info, "Storing alignment object on {}/{}", ccdbHost, path);
    o2::ccdb::CcdbApi api;
    map<string, string> metadata; // can be empty
    api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
    // store abitrary user object in strongly typed manner
    api.storeAsTFileAny(&params, path, metadata, tmin, tmax);
  }

  if (!fileName.empty()) {
    LOGP(info, "Storing MCH alignment in local file {}", fileName);
    TFile algFile(fileName.c_str(), "recreate");
    algFile.WriteObjectAny(&params, "std::vector<o2::detectors::AlignParam>", "alignment");
    algFile.Close();
  }
}

void misAlignGeometry()
{
  if (!gGeoManager) {
    std::cerr << "gGeoManager == nullptr, must create a geometry first\n";
    return;
  }
  // If not closed, we need to close it
  if (!gGeoManager->IsClosed()) {
    gGeoManager->CloseGeometry();
  }
  // Check for Alignable Volumes?
  // addAlignableVolumes();

  std::vector<o2::detectors::AlignParam> params;

  // The misaligner
  o2::mch::geo::MisAligner aGMA;
  aGMA.setModuleCartMisAlig(0.1, 0., 0.2, 0., 0.3, 0.);
  aGMA.setModuleAngMisAlig(0.0, 0., 0., 0., 0., 0.);
  aGMA.setCartMisAlig(0.01, 0., 0.02, 0., 0.03, 0.);
  aGMA.setAngMisAlig(0.0, 0.0, 0., 0., 0., 0.);
  aGMA.misAlign(params);
}

void setVolumeVisibility(const char* pattern, bool visible, bool visibleDaughters)
{
  TPRegexp re(pattern);
  TIter next(gGeoManager->GetListOfVolumes());
  TGeoVolume* vol;

  while ((vol = static_cast<TGeoVolume*>(next()))) {
    if (TString(vol->GetName()).Contains(re)) {
      vol->SetVisibility(visible);
      vol->SetVisDaughters(visibleDaughters);
    }
  }
}

void setVolumeColor(const char* pattern, int lineColor, int fillColor)
{
  TPRegexp re(pattern);
  TIter next(gGeoManager->GetListOfVolumes());
  TGeoVolume* vol;

  while ((vol = static_cast<TGeoVolume*>(next()))) {
    if (TString(vol->GetName()).Contains(re)) {
      vol->SetFillColor(fillColor);
      vol->SetLineColor(lineColor);
    }
  }
}

void drawOptionPresetBasic()
{
  gGeoManager->SetVisLevel(4);

  setVolumeVisibility("cave", false, true);

  // Hide to half-chamber top volumes
  setVolumeVisibility("^SC", false, true);

  // Hide St345 support panels
  setVolumeVisibility("support panel", false, false);

  // Hide St345 LV wires
  setVolumeVisibility(" LV ", false, false);

  // Make St345 carbon panels dark gray
  setVolumeColor("panel carbon", kGray + 3);

  // Make St345 insulators dark green
  setVolumeColor("insulator", kGreen + 3);

  // Hide most of St1
  setVolumeVisibility("SQ", false, true);

  // Only reveal gas module
  setVolumeVisibility("SA", true, true);
  setVolumeColor("SA", kCyan - 10);
}

void drawGeometry()
{
  // minimal macro to test setup of the geometry

  createStandaloneGeometry();

  drawOptionPresetBasic();

  gGeoManager->GetTopVolume()->Draw("ogl");

  TGLViewer* gl = static_cast<TGLViewer*>(gPad->GetViewer3D("ogl"));
  TGLCamera& c = gl->CurrentCamera();

  // gl->SetStyle(TGLRnrCtx::kWireFrame);
  gl->SetStyle(TGLRnrCtx::kOutline);
  // gl->SetStyle(TGLRnrCtx::kFill);
}

o2::base::GeometryManager::MatBudgetExt getMatBudgetExt(const o2::math_utils::Transform3D& t, math_utils::Vector3D<double>& n, float x, float y, float thickness)
{
  math_utils::Point3D<double> point;
  t.LocalToMaster(math_utils::Point3D<double>{x, y, 0}, point);
  return o2::base::GeometryManager::meanMaterialBudgetExt(math_utils::Point3D<double>{point + n * thickness / 2.0}, math_utils::Point3D<double>{point - n * thickness / 2.0});
}

std::ostream& operator<<(std::ostream& os, o2::base::GeometryManager::MatBudgetExt m)
{
  os << "L=" << m.length << " <Rho>=" << m.meanRho << " <A>=" << m.meanA
     << " <Z>=" << m.meanZ << " <x/x0>=" << m.meanX2X0 << " nCross=" << m.nCross;
  return os;
}

math_utils::Vector3D<double> getNormalVector(const o2::math_utils::Transform3D& t)
{
  math_utils::Point3D<double> px, py, po;
  t.LocalToMaster(math_utils::Point3D<double>{0, 1, 0}, py);
  t.LocalToMaster(math_utils::Point3D<double>{1, 0, 0}, px);
  t.LocalToMaster(math_utils::Point3D<double>{0, 0, 0}, po);
  math_utils::Vector3D<double> a{px - po};
  math_utils::Vector3D<double> b{py - po};
  return a.Cross(b).Unit();
}

TH2* getRadio(int detElemId, float xmin, float ymin, float xmax, float ymax, float xstep, float ystep, float thickness)
{
  if (xmin >= xmax || ymin >= ymax) {
    std::cerr << "incorrect limits\n";
    return nullptr;
  }
  TH2* hmatb = new TH2F("hmatb", "hmatb", (int)((xmax - xmin) / xstep), xmin, xmax, (int)((ymax - ymin) / ystep), ymin, ymax);

  auto transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
  auto t = transformation(detElemId);

  auto normal = getNormalVector(t);

  for (auto x = xmin; x < xmax; x += xstep) {
    for (auto y = ymin; y < ymax; y += ystep) {
      auto matb = getMatBudgetExt(t, normal, x, y, thickness);
      if (std::isfinite(matb.meanX2X0)) {
        hmatb->Fill(x, y, matb.meanX2X0);
      }
    }
  }
  return hmatb;
}
} // namespace test
} // namespace mch
} // namespace o2
