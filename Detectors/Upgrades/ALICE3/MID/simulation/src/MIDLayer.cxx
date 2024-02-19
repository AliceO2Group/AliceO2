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

#include "MI3Simulation/MIDLayer.h"
#include "MI3Base/GeometryTGeo.h"
#include <TGeoManager.h>
#include <TMath.h>

#include "Framework/Logger.h"

#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TGeoBBox.h>

namespace o2::mi3
{
MIDLayer::MIDLayer(int layerNumber,
                   std::string layerName,
                   float rInn,
                   float length,
                   int nstaves) : mName(layerName),
                                  mRadius(rInn),
                                  mLength(length),
                                  mNumber(layerNumber)
{
  mStaves.reserve(nstaves);
  LOGP(info, "Constructing MIDLayer: {} with inner radius: {}, length: {} cm and {} staves", mName, mRadius, mLength, mNStaves);
  for (int iStave = 0; iStave < nstaves; ++iStave) {
    mStaves.emplace_back(GeometryTGeo::composeSymNameStave(layerNumber, iStave),
                         mRadius,
                         TMath::TwoPi() / (float)nstaves * iStave,
                         mNumber,
                         mLength,
                         50.f,  // hardcoded for now
                         0.5f); // hardcoded for now
  }
}

void MIDLayer::createLayer(TGeoVolume* motherVolume)
{
  LOGP(info, "Creating MIDLayer: {}", mName);
  TGeoTube* layer = new TGeoTube(mName.c_str(), mRadius, mRadius + 10, mLength);
  auto* airMed = gGeoManager->GetMedium("MI3_AIR");
  TGeoVolume* layerVolume = new TGeoVolume(mName.c_str(), layer, airMed);
  layerVolume->SetVisibility(false);
  motherVolume->AddNode(layerVolume, 0);
  for (auto& stave : mStaves) {
    stave.createStave(layerVolume);
  }
}

MIDLayer::Stave::Stave(std::string staveName,
                       float radDistance,
                       float rotAngle,
                       int layer,
                       float staveLength,
                       float staveWidth,
                       float staveThickness) : mName(staveName),
                                               mRadDistance(radDistance),
                                               mRotAngle(rotAngle),
                                               mLength(staveLength),
                                               mWidth(staveWidth),
                                               mThickness(staveThickness),
                                               mLayer(layer)
{
  // Staves are ideal shapes made of air including the modules, for now.
}

void MIDLayer::Stave::createStave(TGeoVolume* motherVolume)
{
  LOGP(info, "\tCreating MIDStave: {} layer: {}", mName, mLayer);
  TGeoBBox* stave = new TGeoBBox(mName.c_str(), mWidth, mThickness, mLength);
  auto* airMed = gGeoManager->GetMedium("MI3_AIR");
  TGeoVolume* staveVolume = new TGeoVolume(mName.c_str(), stave, airMed);
  staveVolume->SetVisibility(true);
  staveVolume->SetLineColor(mLayer ? kRed : kBlue);
  //
  TGeoCombiTrans* staveTrans = new TGeoCombiTrans(mRadDistance * TMath::Cos(mRotAngle),
                                                  mRadDistance * TMath::Sin(mRotAngle),
                                                  0,
                                                  new TGeoRotation("rot", 90 + mRotAngle * TMath::RadToDeg(), 0, 0));
  motherVolume->AddNode(staveVolume, 0, staveTrans);
}

MIDLayer::Stave::Module::Module(std::string moduleName,
                                int nBars,
                                float barSpacing,
                                float barWidth,
                                float barLength,
                                float barThickness)
{
  TGeoMedium* medPoly = gGeoManager->GetMedium("MI3_POLYSTYRENE");
  if (!medPoly) {
    LOGP(fatal, "MID_POLYSTYRENE medium not found");
  }

  // TGeoVolume* module = new TGeoVolumeAssembly(moduleName.c_str());
  // for (int i = 0; i < 5; ++i) {
  //   TGeoVolume* box = gGeoManager->MakeBox(, med, boxWidth / 2, boxHeight / 2, boxDepth / 2);
  //   box->SetLineColor(i + 1); // Set different colors for visualization
  //   top->AddNode(box, i, new TGeoTranslation(i * (boxWidth + separation), 0, 0));
  // }
}
} // namespace o2::mi3