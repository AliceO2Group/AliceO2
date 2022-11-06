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

  fSensorLayerThickness = o2::itsmft::SegmentationAlpide::SensorLayerThickness;
}

//________________________________________________________________
DescriptorInnerBarrelITS2::DescriptorInnerBarrelITS2(int nlayers) : DescriptorInnerBarrel(nlayers)
{
  //
  // Standard constructor
  //

  fSensorLayerThickness = o2::itsmft::SegmentationAlpide::SensorLayerThickness;
}

//________________________________________________________________
void DescriptorInnerBarrelITS2::Configure()
{
  // build ITS2 upgrade detector
  fTurboLayer.resize(fNumLayers);
  fLayerPhi0.resize(fNumLayers);
  fLayerRadii.resize(fNumLayers);
  fLayerZLen.resize(fNumLayers);
  fStavePerLayer.resize(fNumLayers);
  fUnitPerStave.resize(fNumLayers);
  fChipThickness.resize(fNumLayers);
  fDetectorThickness.resize(fNumLayers);
  fStaveTilt.resize(fNumLayers);
  fStaveWidth.resize(fNumLayers);
  fChipTypeID.resize(fNumLayers);
  fBuildLevel.resize(fNumLayers);
  fLayer.resize(fNumLayers);

  // Radii are from last TDR (ALICE-TDR-017.pdf Tab. 1.1)
  std::vector<std::array<double, 6>> IBdat;
  IBdat.emplace_back(std::array<double, 6>{2.24, 2.34, 2.67, 9., 16.42, 12});
  IBdat.emplace_back(std::array<double, 6>{3.01, 3.15, 3.46, 9., 12.18, 16});
  IBdat.emplace_back(std::array<double, 6>{3.78, 3.93, 4.21, 9., 9.55, 20});

  for (auto idLayer{0u}; idLayer < fNumLayers; ++idLayer) {
    fTurboLayer[idLayer] = true;
    fLayerPhi0[idLayer] = IBdat[idLayer][4];
    fLayerRadii[idLayer] = IBdat[idLayer][1];
    fStavePerLayer[idLayer] = IBdat[idLayer][5];
    fUnitPerStave[idLayer] = IBdat[idLayer][3];
    fChipThickness[idLayer] = 50.e-4;
    fStaveWidth[idLayer] = o2::itsmft::SegmentationAlpide::SensorSizeRows;
    fStaveTilt[idLayer] = radii2Turbo(IBdat[idLayer][0], IBdat[idLayer][1], IBdat[idLayer][2], o2::itsmft::SegmentationAlpide::SensorSizeRows);
    fDetectorThickness[idLayer] = fSensorLayerThickness;
    fChipTypeID[idLayer] = 0;
    fBuildLevel[idLayer] = 0;

    LOG(info) << "L# " << idLayer << " Phi:" << fLayerPhi0[idLayer] << " R:" << fLayerRadii[idLayer] << " Nst:" << fStavePerLayer[idLayer] << " Nunit:" << fUnitPerStave[idLayer]
              << " W:" << fStaveWidth[idLayer] << " Tilt:" << fStaveTilt[idLayer] << " Lthick:" << fChipThickness[idLayer] << " Dthick:" << fDetectorThickness[idLayer]
              << " DetID:" << fChipTypeID[idLayer] << " B:" << fBuildLevel[idLayer];
  }

  fWrapperMinRadius = 2.1;
  fWrapperMaxRadius = 16.4;
  fWrapperZSpan = 70.;
}

//________________________________________________________________
V3Layer* DescriptorInnerBarrelITS2::CreateLayer(int idLayer, TGeoVolume* dest)
{
  if (idLayer >= fNumLayers) {
    LOG(fatal) << "Trying to define layer " << idLayer << " of inner barrel, but only " << fNumLayers << " layers expected!";
    return nullptr;
  }

  if (fTurboLayer[idLayer]) {
    fLayer[idLayer] = new V3Layer(idLayer, true, false);
    fLayer[idLayer]->setStaveWidth(fStaveWidth[idLayer]);
    fLayer[idLayer]->setStaveTilt(fStaveTilt[idLayer]);
  } else {
    fLayer[idLayer] = new V3Layer(idLayer, false);
  }

  fLayer[idLayer]->setPhi0(fLayerPhi0[idLayer]);
  fLayer[idLayer]->setRadius(fLayerRadii[idLayer]);
  fLayer[idLayer]->setNumberOfStaves(fStavePerLayer[idLayer]);
  fLayer[idLayer]->setNumberOfUnits(fUnitPerStave[idLayer]);
  fLayer[idLayer]->setChipType(fChipTypeID[idLayer]);
  fLayer[idLayer]->setBuildLevel(fBuildLevel[idLayer]);

  fLayer[idLayer]->setStaveModel(V3Layer::kIBModel4);

  if (fChipThickness[idLayer] != 0) {
    fLayer[idLayer]->setChipThick(fChipThickness[idLayer]);
  }
  if (fDetectorThickness[idLayer] != 0) {
    fLayer[idLayer]->setSensorThick(fDetectorThickness[idLayer]);
  }

  fLayer[idLayer]->createLayer(dest);

  return fLayer[idLayer]; // is this needed?
}

//________________________________________________________________
void DescriptorInnerBarrelITS2::CreateServices(TGeoVolume* dest)
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

  std::unique_ptr<V3Services> mServicesGeometry(new V3Services());

  // Create the End Wheels on Side A
  TGeoVolume* endWheelsA = mServicesGeometry.get()->createIBEndWheelsSideA();
  dest->AddNode(endWheelsA, 1, nullptr);

  // Create the End Wheels on Side C
  TGeoVolume* endWheelsC = mServicesGeometry.get()->createIBEndWheelsSideC();
  dest->AddNode(endWheelsC, 1, nullptr);

  // Create the CYSS Assembly (i.e. the supporting half cylinder and cone)
  TGeoVolume* cyss = mServicesGeometry.get()->createCYSSAssembly();
  dest->AddNode(cyss, 1, nullptr);

  mServicesGeometry.get()->createIBGammaConvWire(dest);
}

//________________________________________________________________
void DescriptorInnerBarrelITS2::AddAlignableVolumesLayer(int idLayer, int wrapperLayerId, TString& parentPath, int& lastUID)
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

  int nHalfBarrel = fLayer[idLayer]->getNumberOfHalfBarrelsPerParent();
  int start = nHalfBarrel > 0 ? 0 : -1;
  for (int iHalfBarrel{start}; iHalfBarrel < nHalfBarrel; ++iHalfBarrel) {
    AddAlignableVolumesHalfBarrel(idLayer, iHalfBarrel, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::AddAlignableVolumesHalfBarrel(int idLayer, int iHalfBarrel, TString& parentPath, int& lastUID) const
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

  int nStaves = fLayer[idLayer]->getNumberOfStavesPerParent();
  for (int iStave{0}; iStave < nStaves; ++iStave) {
    AddAlignableVolumesStave(idLayer, iHalfBarrel, iStave, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::AddAlignableVolumesStave(int idLayer, int iHalfBarrel, int iStave, TString& parentPath, int& lastUID) const
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

  int nHalfStave = fLayer[idLayer]->getNumberOfHalfStavesPerParent();
  int start = nHalfStave > 0 ? 0 : -1;
  for (int iHalfStave{start}; iHalfStave < nHalfStave; ++iHalfStave) {
    AddAlignableVolumesHalfStave(idLayer, iHalfBarrel, iStave, iHalfStave, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::AddAlignableVolumesHalfStave(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, TString& parentPath, int& lastUID) const
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

  int nModules = fLayer[idLayer]->getNumberOfModulesPerParent();
  int start = nModules > 0 ? 0 : -1;
  for (int iModule{start}; iModule < nModules; iModule++) {
    AddAlignableVolumesModule(idLayer, iHalfBarrel, iStave, iHalfStave, iModule, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::AddAlignableVolumesModule(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, int iModule, TString& parentPath, int& lastUID) const
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

  int nChips = fLayer[idLayer]->getNumberOfChipsPerParent();
  for (int iChip{0}; iChip < nChips; ++iChip) {
    AddAlignableVolumesChip(idLayer, iHalfBarrel, iStave, iHalfStave, iModule, iChip, path, lastUID);
  }
}

void DescriptorInnerBarrelITS2::AddAlignableVolumesChip(int idLayer, int iHalfBarrel, int iStave, int iHalfStave, int iModule, int iChip, TString& parentPath, int& lastUID) const
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