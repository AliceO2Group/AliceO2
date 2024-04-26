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
                                  mNumber(layerNumber),
                                  mNStaves(nstaves)
{
  mStaves.reserve(nstaves);
  LOGP(debug, "Constructing MIDLayer: {} with inner radius: {}, length: {} cm and {} staves", mName, mRadius, mLength, mNStaves);
  for (int iStave = 0; iStave < mNStaves; ++iStave) {
    mStaves.emplace_back(GeometryTGeo::composeSymNameStave(layerNumber, iStave),
                         mRadius,
                         TMath::TwoPi() / (float)nstaves * iStave,
                         mNumber,
                         iStave,
                         mLength,
                         !layerNumber ? 59.8f : 61.75f,
                         0.5f);
  }
}

MIDLayer::Stave::Stave(std::string staveName,
                       float radDistance,
                       float rotAngle,
                       int layer,
                       int number,
                       float staveLength,
                       float staveWidth,
                       float staveThickness,
                       int nModulesZ) : mName(staveName),
                                        mRadDistance(radDistance),
                                        mRotAngle(rotAngle),
                                        mLength(staveLength),
                                        mWidth(staveWidth),
                                        mThickness(staveThickness),
                                        mLayer(layer),
                                        mNumber(number),
                                        mNModulesZ(nModulesZ)
{
  // Staves are ideal shapes made of air including the modules, for now.
  LOGP(debug, "\t\tConstructing MIDStave: {} layer: {} at angle {}", mName, mLayer, mRotAngle * TMath::RadToDeg());
  mModules.reserve(nModulesZ);
  for (int iModule = 0; iModule < mNModulesZ; ++iModule) {
    mModules.emplace_back(GeometryTGeo::composeSymNameModule(mLayer, mNumber, iModule),
                          mLayer,
                          mNumber,
                          iModule,
                          !mLayer ? 23 : 20,
                          -staveLength,
                          !mLayer ? 49.9f : 61.75f);
  }
}

MIDLayer::Stave::Module::Module(std::string moduleName,
                                int layer,
                                int stave,
                                int number,
                                int nBars,
                                float zOffset,
                                float barLength,
                                float barSpacing,
                                float barWidth,
                                float barThickness) : mName(moduleName),
                                                      mNBars(nBars),
                                                      mLayer(layer),
                                                      mStave(stave),
                                                      mNumber(number),
                                                      mZOffset(zOffset),
                                                      mBarSpacing(barSpacing),
                                                      mBarWidth(barWidth),
                                                      mBarLength(barLength),
                                                      mBarThickness(barThickness)
{
  mSensors.reserve(nBars);
  LOGP(debug, "\t\t\tConstructing MIDModule: {}", mName);
  for (int iBar = 0; iBar < mNBars; ++iBar) {
    mSensors.emplace_back(GeometryTGeo::composeSymNameSensor(mLayer, mStave, mNumber, iBar),
                          mLayer,
                          mStave,
                          mNumber,
                          iBar,
                          !mLayer ? -59.8f : -52.f,  // offset
                          !mLayer ? 49.9f : 61.75f); // sensor length
  }
}

MIDLayer::Stave::Module::Sensor::Sensor(std::string sensorName,
                                        int layer,
                                        int stave,
                                        int module,
                                        int number,
                                        float moduleOffset,
                                        float sensorLength,
                                        float sensorWidth,
                                        float sensorThickness,
                                        float sensorSpacing) : mName(sensorName),
                                                               mLayer(layer),
                                                               mStave(stave),
                                                               mModule(module),
                                                               mNumber(number),
                                                               mModuleOffset(moduleOffset),
                                                               mWidth(sensorWidth),
                                                               mLength(sensorLength),
                                                               mThickness(sensorThickness),
                                                               mSpacing(sensorSpacing)
{
  LOGP(debug, "\t\t\t\tConstructing MIDSensor: {}", mName);
}

void MIDLayer::createLayer(TGeoVolume* motherVolume)
{
  LOGP(debug, "Creating MIDLayer: {}", mName);
  TGeoVolumeAssembly* layerVolume = new TGeoVolumeAssembly(mName.c_str());
  motherVolume->AddNode(layerVolume, 0);
  for (auto& stave : mStaves) {
    stave.createStave(layerVolume);
  }
}

void MIDLayer::Stave::createStave(TGeoVolume* motherVolume)
{
  LOGP(debug, "\tCreating MIDStave: {} layer: {}", mName, mLayer);
  TGeoVolumeAssembly* staveVolume = new TGeoVolumeAssembly(mName.c_str());
  // Create the modules
  for (auto& module : mModules) {
    module.createModule(staveVolume);
  }

  TGeoCombiTrans* staveTrans = new TGeoCombiTrans(mRadDistance * TMath::Cos(mRotAngle),
                                                  mRadDistance * TMath::Sin(mRotAngle),
                                                  0,
                                                  new TGeoRotation("rot", 90 + mRotAngle * TMath::RadToDeg(), 0, 0));
  motherVolume->AddNode(staveVolume, 0, staveTrans);
}

void MIDLayer::Stave::Module::createModule(TGeoVolume* motherVolume)
{
  // Module is an air box with floating bars inside for the moment
  auto sumWidth = ((mBarWidth * 2 + mBarSpacing) * mNBars) / 2;
  LOGP(debug, "\t\t\tCreating MIDModule: {} with ", mName);
  TGeoVolumeAssembly* moduleVolume = new TGeoVolumeAssembly(mName.c_str() /*, module, airMed*/);

  // Create the bars
  for (auto& sensor : mSensors) {
    sensor.createSensor(moduleVolume);
  }
  TGeoCombiTrans* modTrans = nullptr;
  if (!mLayer) {
    modTrans = new TGeoCombiTrans(0, 0, mZOffset + mNumber * 2 * mBarLength + mBarLength, nullptr);
  } else {
    modTrans = new TGeoCombiTrans(0, 0, mZOffset + mNumber * 2 * sumWidth + sumWidth, nullptr);
  }
  motherVolume->AddNode(moduleVolume, 0, modTrans);
}

void MIDLayer::Stave::Module::Sensor::createSensor(TGeoVolume* motherVolume)
{
  LOGP(debug, "\t\t\t\tCreating MIDSensor: {}", mName);
  TGeoBBox* sensor = nullptr;
  if (!mLayer) {
    sensor = new TGeoBBox(mName.c_str(), mWidth, mThickness, mLength);
  } else {
    sensor = new TGeoBBox(mName.c_str(), mLength, mThickness, mWidth);
  }
  auto* polyMed = gGeoManager->GetMedium("MI3_POLYSTYRENE");
  TGeoVolume* sensorVolume = new TGeoVolume(mName.c_str(), sensor, polyMed);
  sensorVolume->SetVisibility(true);
  auto totWidth = mWidth + mSpacing / 2;
  TGeoTranslation* sensorTrans = nullptr;
  if (!mLayer) {
    sensorTrans = new TGeoTranslation(mModuleOffset + 2 * totWidth * mNumber + totWidth, 0, 0);
    sensorVolume->SetLineColor(kAzure + 4);
    sensorVolume->SetTransparency(50);
  } else {
    sensorTrans = new TGeoTranslation(0, 0, mModuleOffset + 2 * totWidth * mNumber + totWidth);
    sensorVolume->SetLineColor(kAzure + 4);
    sensorVolume->SetTransparency(50);
  }
  motherVolume->AddNode(sensorVolume, 0, sensorTrans);
}
} // namespace o2::mi3