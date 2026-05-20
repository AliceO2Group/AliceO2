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

#include "IOTOFBase/GeometryTGeo.h"
#include "IOTOFSimulation/Layer.h"

#include <TCanvas.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoVolume.h>
#include <TStyle.h>

#include <iostream>

namespace
{
void ensureMedium(const char* name, int id, double a, double z, double density)
{
  if (!gGeoManager->GetMedium(name)) {
    auto* mat = new TGeoMaterial(name, a, z, density);
    new TGeoMedium(name, id, mat);
  }
}

void prepareMinimalMedia()
{
  ensureMedium("VACUUM$", 0, 1., 1., 1.e-16);
  ensureMedium("TF3_AIR$", 1, 14.61, 7.3, 1.20479e-3);
  ensureMedium("TF3_SILICON$", 3, 28.086, 14., 2.33);
}
} // namespace

void drawTOFGeometry(double x2x0 = 0.02,
                     double sensorThickness = 0.005,
                     bool checkOverlaps = true,
                     double overlapToleranceCm = 0.01)
{
  gStyle->SetOptStat(0);

  if (gGeoManager) {
    delete gGeoManager;
  }

  auto* geo = new TGeoManager("IOTOFGeomFromLayer", "Geometry built from Layer.h classes");
  prepareMinimalMedia();

  auto* top = geo->MakeBox("TOP", geo->GetMedium("VACUUM$"), 1200., 1200., 1200.);
  geo->SetTopVolume(top);

  auto* mother = new TGeoVolumeAssembly("IOTOFMacroVol");
  top->AddNode(mother, 1, new TGeoTranslation(0., 0., 0.));

  // Build using the same classes and createLayer() used by detector geometry code.
  o2::iotof::ITOFLayer itof(o2::iotof::GeometryTGeo::getITOFLayerPattern(),
                            21.f, 0.f, 129.f, 0.f, x2x0,
                            o2::iotof::Layer::kBarrelSegmented,
                            24, 5.42, 3.0, 10, sensorThickness);

  o2::iotof::OTOFLayer otof(o2::iotof::GeometryTGeo::getOTOFLayerPattern(),
                            92.f, 0.f, 680.f, 0.f, x2x0,
                            o2::iotof::Layer::kBarrelSegmented,
                            62, 9.74, 5.0, 54, sensorThickness);

  itof.createLayer(mother);
  otof.createLayer(mother);

  geo->CloseGeometry();

  std::cout << "Built geometry from Layer.h classes with x2x0=" << x2x0
            << " and sensorThickness=" << sensorThickness << " cm\n";
  std::cout << "ITOF sensitive volumes: " << o2::iotof::ITOFLayer::mRegister.size() << "\n";
  std::cout << "OTOF sensitive volumes: " << o2::iotof::OTOFLayer::mRegister.size() << "\n";

  if (checkOverlaps) {
    std::cout << "Checking overlaps with tolerance=" << overlapToleranceCm << " cm\n";
    geo->CheckOverlaps(overlapToleranceCm);
    geo->PrintOverlaps();
  }

  top->Draw("ogl");
}
