// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/GeometryTest.h"

#include "DetectorsBase/GeometryManager.h"
#include "MCHSimulation/Geometry.h"
#include "Math/GenVector/Cartesian3D.h"
#include "TGeoManager.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"
#include "TH2F.h"
#include <iostream>
#include "TPRegexp.h"
#include "TGLViewer.h"
#include "TGLRnrCtx.h"
#include "TVirtualPad.h"

namespace o2
{
namespace mch
{
namespace test
{

TGeoVolume* createAirVacuumCave(const char* name)
{
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAirVacuum = 1.20479E-10;
  gGeoManager->Mixture("air", aAir, zAir, dAirVacuum, 4, wAir, 1);
  TGeoMedium* mair = gGeoManager->Medium("air", 1, 1,
                                         false, /* isvol */
                                         0,     /* ifield */
                                         -1.0,  /* fieldm */
                                         -1.0,  /* tmaxfd */
                                         -1.0,  /* stemax */
                                         -1.0,  /* deemax */
                                         -1.0,  /* epsil */
                                         -1.0 /* stmin */);
  return gGeoManager->MakeBox(name, mair, 2000.0, 2000.0, 3000.0);
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
  o2::mch::createGeometry(*top);
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

  // Temporary
  //gGeoManager->GetTopVolume()->Draw("ogl");
  //gGeoManager->GetTopVolume()->Draw();
  gGeoManager->CheckOverlaps(0.01);
  gGeoManager->PrintOverlaps();
  gGeoManager->Export("MCHgeometry.root");
  /*
  TGLViewer* gl = static_cast<TGLViewer*>(gPad->GetViewer3D("ogl"));
  TGLCamera& c = gl->CurrentCamera();

  // gl->SetStyle(TGLRnrCtx::kWireFrame);
  gl->SetStyle(TGLRnrCtx::kOutline);
  // gl->SetStyle(TGLRnrCtx::kFill);
  */
}

o2::Base::GeometryManager::MatBudget getMatBudget(const o2::Transform3D& t, Vector3D<double>& n, float x, float y, float thickness)
{
  Point3D<double> point;
  t.LocalToMaster(Point3D<double>{ x, y, 0 }, point);
  auto front = Point3D<double>(point + n * thickness / 2.0);
  auto back = Point3D<double>(point - n * thickness / 2.0);
  return o2::Base::GeometryManager::MeanMaterialBudget(front, back);
}

std::ostream& operator<<(std::ostream& os, o2::Base::GeometryManager::MatBudget m)
{
  os << "L=" << m.length << " <Rho>=" << m.meanRho << " <A>=" << m.meanA
     << " <Z>=" << m.meanZ << " <x/x0>=" << m.meanX2X0 << " nCross=" << m.nCross;
  return os;
}

Vector3D<double> getNormalVector(const o2::Transform3D& t)
{
  Point3D<double> pa, pb;
  t.LocalToMaster(Point3D<double>{ 0, 1, 0 }, pb);
  t.LocalToMaster(Point3D<double>{ 1, 0, 0 }, pa);
  Vector3D<double> a{ pa };
  Vector3D<double> b{ pb };
  return a.Cross(b).Unit();
}

TH2* getRadio(int detElemId, float xmin, float ymin, float xmax, float ymax, float xstep, float ystep, float thickness)
{
  if (xmin >= xmax || ymin >= ymax) {
    std::cerr << "incorrect limits\n";
    return nullptr;
  }
  TH2* hmatb = new TH2F("hmatb", "hmatb", (int)((xmax - xmin) / xstep), xmin, xmax, (int)((ymax - ymin) / ystep), ymin, ymax);

  auto t = o2::mch::getTransformation(detElemId, *gGeoManager);

  auto normal = getNormalVector(t);

  std::cout << getMatBudget(t, normal, 30, 15, 5) << "\n";
  std::cout << getMatBudget(t, normal, 30, 15, 10) << "\n";
  std::cout << getMatBudget(t, normal, 30, 15, 15) << "\n";
  std::cout << getMatBudget(t, normal, 30, 15, 35) << "\n";

  for (auto x = xmin; x < xmax; x += xstep) {
    for (auto y = ymin; y < ymax; y += ystep) {
      auto matb = getMatBudget(t, normal, x, y, thickness);
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
