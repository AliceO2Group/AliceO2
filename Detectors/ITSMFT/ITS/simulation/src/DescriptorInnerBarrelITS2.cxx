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

#include "FairDetector.h"      // for FairDetector
#include <fairlogger/Logger.h> // for LOG, LOG_IF
#include "FairRootManager.h"   // for FairRootManager
#include "FairRun.h"           // for FairRun
#include "FairRuntimeDb.h"     // for FairRuntimeDb
#include "FairVolume.h"        // for FairVolume
#include "FairRootManager.h"

#include "TGeoManager.h"     // for TGeoManager, gGeoManager
#include "TGeoTube.h"        // for TGeoTube
#include "TGeoPcon.h"        // for TGeoPcon
#include "TGeoVolume.h"      // for TGeoVolume, TGeoVolumeAssembly
#include "TString.h"         // for TString, operator+
#include "TVirtualMC.h"      // for gMC, TVirtualMC
#include "TVirtualMCStack.h" // for TVirtualMCStack

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSBase/ITSBaseParam.h"
#include "ITSSimulation/DescriptorInnerBarrelITS2.h"
#include "ITSSimulation/V3Services.h"
#include "ITSSimulation/V3Layer.h"

using namespace o2::its;

/// \cond CLASSIMP
ClassImp(DescriptorInnerBarrelITS2);
/// \endcond

//________________________________________________________________
DescriptorInnerBarrelITS2::DescriptorInnerBarrelITS2() : DescriptorInnerBarrel()
{
  //
  // Default constructor
  //

  mSensorLayerThickness = o2::itsmft::SegmentationAlpide::SensorLayerThickness;
}

//________________________________________________________________
DescriptorInnerBarrelITS2::DescriptorInnerBarrelITS2(int nlayers) : DescriptorInnerBarrel(nlayers)
{
  //
  // Standard constructor
  //

  mSensorLayerThickness = o2::itsmft::SegmentationAlpide::SensorLayerThickness;
}

//________________________________________________________________
void DescriptorInnerBarrelITS2::configure()
{
  // build ITS2 upgrade detector
  mTurboLayer.resize(mNumLayers);
  mLayerPhi0.resize(mNumLayers);
  mLayerRadii.resize(mNumLayers);
  mStavePerLayer.resize(mNumLayers);
  mUnitPerStave.resize(mNumLayers);
  mChipThickness.resize(mNumLayers);
  mDetectorThickness.resize(mNumLayers);
  mStaveTilt.resize(mNumLayers);
  mStaveWidth.resize(mNumLayers);
  mChipTypeID.resize(mNumLayers);
  mBuildLevel.resize(mNumLayers);
  mLayer.resize(mNumLayers);

  // Radii are from last TDR (ALICE-TDR-017.pdf Tab. 1.1)
  std::vector<std::array<double, 6>> IBdat;
  IBdat.emplace_back(std::array<double, 6>{2.24, 2.34, 2.67, 9., 16.42, 12});
  IBdat.emplace_back(std::array<double, 6>{3.01, 3.15, 3.46, 9., 12.18, 16});
  IBdat.emplace_back(std::array<double, 6>{3.78, 3.93, 4.21, 9., 9.55, 20});

  for (auto idLayer{0u}; idLayer < mNumLayers; ++idLayer) {
    mTurboLayer[idLayer] = true;
    mLayerPhi0[idLayer] = IBdat[idLayer][4];
    mLayerRadii[idLayer] = IBdat[idLayer][1];
    mStavePerLayer[idLayer] = IBdat[idLayer][5];
    mUnitPerStave[idLayer] = IBdat[idLayer][3];
    mChipThickness[idLayer] = 50.e-4;
    mStaveWidth[idLayer] = o2::itsmft::SegmentationAlpide::SensorSizeRows;
    mStaveTilt[idLayer] = radii2Turbo(IBdat[idLayer][0], IBdat[idLayer][1], IBdat[idLayer][2], o2::itsmft::SegmentationAlpide::SensorSizeRows);
    mDetectorThickness[idLayer] = mSensorLayerThickness;
    mChipTypeID[idLayer] = 0;
    mBuildLevel[idLayer] = 0;

    LOG(info) << "L# " << idLayer << " Phi:" << mLayerPhi0[idLayer] << " R:" << mLayerRadii[idLayer] << " Nst:" << mStavePerLayer[idLayer] << " Nunit:" << mUnitPerStave[idLayer]
              << " W:" << mStaveWidth[idLayer] << " Tilt:" << mStaveTilt[idLayer] << " Lthick:" << mChipThickness[idLayer] << " Dthick:" << mDetectorThickness[idLayer]
              << " DetID:" << mChipTypeID[idLayer] << " B:" << mBuildLevel[idLayer];
  }

  mWrapperMinRadius = 2.1;
  mWrapperMaxRadius = 16.4;
  mWrapperZSpan = 70.;
}

//________________________________________________________________
V3Layer* DescriptorInnerBarrelITS2::createLayer(int idLayer, TGeoVolume* dest)
{
  if (idLayer >= mNumLayers) {
    LOG(fatal) << "Trying to define layer " << idLayer << " of inner barrel, but only " << mNumLayers << " layers expected!";
    return nullptr;
  }

  if (mTurboLayer[idLayer]) {
    mLayer[idLayer] = new V3Layer(idLayer, true, false);
    mLayer[idLayer]->setStaveWidth(mStaveWidth[idLayer]);
    mLayer[idLayer]->setStaveTilt(mStaveTilt[idLayer]);
  } else {
    mLayer[idLayer] = new V3Layer(idLayer, false);
  }

  mLayer[idLayer]->setPhi0(mLayerPhi0[idLayer]);
  mLayer[idLayer]->setRadius(mLayerRadii[idLayer]);
  mLayer[idLayer]->setNumberOfStaves(mStavePerLayer[idLayer]);
  mLayer[idLayer]->setNumberOfUnits(mUnitPerStave[idLayer]);
  mLayer[idLayer]->setChipType(mChipTypeID[idLayer]);
  mLayer[idLayer]->setBuildLevel(mBuildLevel[idLayer]);

  mLayer[idLayer]->setStaveModel(V3Layer::kIBModel4);

  if (mChipThickness[idLayer] != 0) {
    mLayer[idLayer]->setChipThick(mChipThickness[idLayer]);
  }
  if (mDetectorThickness[idLayer] != 0) {
    mLayer[idLayer]->setSensorThick(mDetectorThickness[idLayer]);
  }

  mLayer[idLayer]->createLayer(dest);

  return mLayer[idLayer]; // is this needed?
}

//________________________________________________________________
void DescriptorInnerBarrelITS2::createServices(TGeoVolume* dest)
{
  //
  // Creates the Inner Barrel Service structures
  //
  // Input:
  //         motherVolume : the volume hosting the services
  //
  // Output:
  //
  // Return:
  //
  // Created:      15 May 2019  Mario Sitta
  //               (partially based on P.Namwongsa implementation in AliRoot)
  // Updated:      19 Jun 2019  Mario Sitta  IB Side A added
  // Updated:      21 Oct 2019  Mario Sitta  CYSS added
  //

  auto& itsBaseParam = ITSBaseParam::Instance();

  std::unique_ptr<V3Services> mServicesGeometry(new V3Services("ITS"));

  if (itsBaseParam.buildEndWheels) {
    // Create the End Wheels on Side A
    TGeoVolume* endWheelsA = mServicesGeometry.get()->createIBEndWheelsSideA();
    dest->AddNode(endWheelsA, 1, nullptr);

    // Create the End Wheels on Side C
    TGeoVolume* endWheelsC = mServicesGeometry.get()->createIBEndWheelsSideC();
    dest->AddNode(endWheelsC, 1, nullptr);
  }
  if (itsBaseParam.buildCYSSAssembly) {
    // Create the CYSS Assembly (i.e. the supporting half cylinder and cone)
    TGeoVolume* cyss = mServicesGeometry.get()->createCYSSAssembly();
    dest->AddNode(cyss, 1, nullptr);
  }
  mServicesGeometry.get()->createIBGammaConvWire(dest);
}

//________________________________________________________________
void DescriptorInnerBarrelITS2::addAlignableVolumesLayer(int idLayer, int wrapperLayerId, TString& parentPath, int& lastUID)
{
  //
  // Add alignable volumes for a Layer and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  // Updated:      06 Jul 2021  Mario Sitta Do not set Layer as alignable volume
  //

  TString wrpV = wrapperLayerId != -1 ? Form("%s%d_1", GeometryTGeo::getITSWrapVolPattern(), wrapperLayerId) : "";
  TString path = Form("%s/%s/%s%d_1", parentPath.Data(), wrpV.Data(), GeometryTGeo::getITSLayerPattern(), idLayer);
  TString sname = GeometryTGeo::composeSymNameLayer(idLayer);

  int nHalfBarrel = mLayer[idLayer]->getNumberOfHalfBarrelsPerParent();
  int start = nHalfBarrel > 0 ? 0 : -1;
  for (int iHalfBarrel{start}; iHalfBarrel < nHalfBarrel; ++iHalfBarrel) {
    addAlignableVolumesHalfBarrel(idLayer, iHalfBarrel, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::addAlignableVolumesHalfBarrel(int idLayer, int iHalfBarrel, TString& parentPath, int& lastUID) const
{
  //
  // Add alignable volumes for a Half barrel and its daughters
  //
  // Created:      28 Jun 2021  Mario Sitta First version (based on similar methods)
  //

  TString path = parentPath;
  if (iHalfBarrel >= 0) {
    path = Form("%s/%s%d_%d", parentPath.Data(), GeometryTGeo::getITSHalfBarrelPattern(), idLayer, iHalfBarrel);
    TString sname = GeometryTGeo::composeSymNameHalfBarrel(idLayer, iHalfBarrel);

    LOG(debug) << "Add " << sname << " <-> " << path;

    if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
      LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
    }
  }

  int nStaves = mLayer[idLayer]->getNumberOfStavesPerParent();
  for (int iStave{0}; iStave < nStaves; ++iStave) {
    addAlignableVolumesStave(idLayer, iHalfBarrel, iStave, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::addAlignableVolumesStave(int idLayer, int iHalfBarrel, int iStave, TString& parentPath, int& lastUID) const
{
  //
  // Add alignable volumes for a Stave and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  // Updated:      29 Jun 2021  Mario Sitta Hal Barrel index added
  //

  TString path = Form("%s/%s%d_%d", parentPath.Data(), GeometryTGeo::getITSStavePattern(), idLayer, iStave);
  TString sname = GeometryTGeo::composeSymNameStave(idLayer, iHalfBarrel, iStave);

  LOG(debug) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
    LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  int nHalfStave = mLayer[idLayer]->getNumberOfHalfStavesPerParent();
  int start = nHalfStave > 0 ? 0 : -1;
  for (int iHalfStave{start}; iHalfStave < nHalfStave; ++iHalfStave) {
    addAlignableVolumesHalfStave(idLayer, iHalfBarrel, iStave, iHalfStave, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::addAlignableVolumesHalfStave(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, TString& parentPath, int& lastUID) const
{
  //
  // Add alignable volumes for a HalfStave (if any) and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  // Updated:      29 Jun 2021  Mario Sitta Hal Barrel index added
  //

  TString path = parentPath;
  if (iHalfStave >= 0) {
    path = Form("%s/%s%d_%d", parentPath.Data(), GeometryTGeo::getITSHalfStavePattern(), idLayer, iHalfStave);
    TString sname = GeometryTGeo::composeSymNameHalfStave(idLayer, iHalfBarrel, iStave, iHalfStave);

    LOG(debug) << "Add " << sname << " <-> " << path;

    if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
      LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
    }
  }

  int nModules = mLayer[idLayer]->getNumberOfModulesPerParent();
  int start = nModules > 0 ? 0 : -1;
  for (int iModule{start}; iModule < nModules; iModule++) {
    addAlignableVolumesModule(idLayer, iHalfBarrel, iStave, iHalfStave, iModule, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::addAlignableVolumesModule(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, int iModule, TString& parentPath, int& lastUID) const
{
  //
  // Add alignable volumes for a Module (if any) and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  // Updated:      29 Jun 2021  Mario Sitta Hal Barrel index added
  //

  TString path = parentPath;
  if (iModule >= 0) {
    path = Form("%s/%s%d_%d", parentPath.Data(), GeometryTGeo::getITSModulePattern(), idLayer, iModule);
    TString sname = GeometryTGeo::composeSymNameModule(idLayer, iHalfBarrel, iStave, iHalfStave, iModule);

    LOG(debug) << "Add " << sname << " <-> " << path;

    if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
      LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
    }
  }

  int nChips = mLayer[idLayer]->getNumberOfChipsPerParent();
  for (int iChip{0}; iChip < nChips; ++iChip) {
    addAlignableVolumesChip(idLayer, iHalfBarrel, iStave, iHalfStave, iModule, iChip, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::addAlignableVolumesChip(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, int iModule, int iChip, TString& parentPath, int& lastUID) const
{
  //
  // Add alignable volumes for a Chip
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  // Updated:      29 Jun 2021  Mario Sitta Hal Barrel index added
  //

  TString path = Form("%s/%s%d_%d", parentPath.Data(), GeometryTGeo::getITSChipPattern(), idLayer, iChip);
  TString sname = GeometryTGeo::composeSymNameChip(idLayer, iHalfBarrel, iStave, iHalfStave, iModule, iChip);
  int modUID = o2::base::GeometryManager::getSensID(o2::detectors::DetID::ITS, lastUID++);

  LOG(debug) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname, path.Data(), modUID)) {
    LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  return;
}