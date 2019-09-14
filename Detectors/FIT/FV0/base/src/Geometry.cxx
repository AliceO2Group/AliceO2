// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Geometry.cxx
/// \brief  Implementation of FV0 geometry.
///
/// \author Maciej Slupecki, University of Jyvaskyla, Finland
/// \author Andreas Molander, University of Helsinki, Finland

#include "FV0Base/Geometry.h"

#include <cmath>

#include <FairLogger.h>

#include <TGeoBBox.h>
#include <TGeoCompositeShape.h>
#include <TGeoCone.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>

ClassImp(o2::fv0::Geometry);

using namespace o2::fv0;

Geometry::Geometry(EGeoType initType) : mGeometryType(initType)
{
  initializeGeometry();
}

Geometry::Geometry(const Geometry& geometry) : mGeometryType(geometry.mGeometryType), mLeftTransformation(nullptr), mRightTransformation(nullptr)
{
  this->mEnabledComponents = geometry.mEnabledComponents;
}

Geometry::~Geometry()
{
  if (mRightTransformation) {
    delete mRightTransformation;
  }
  if (mLeftTransformation) {
    delete mLeftTransformation;
  }
}

const int Geometry::getCurrentCellId(const TVirtualMC* fMC) const
{
  int detectorHalfID = -1;
  int sectorID = -1;
  int ringID = -1;

  fMC->CurrentVolOffID(2, detectorHalfID);
  fMC->CurrentVolOffID(1, sectorID);
  fMC->CurrentVolOffID(0, ringID);

  sectorID += detectorHalfID * sNumberOfCellSectors;

  LOG(DEBUG) << "FV0 Geometry::getCurrentCellId(): \n"
             << "Half id:     " << detectorHalfID << "\n"
             << "Half name:   " << fMC->CurrentVolOffName(2) << "\n"
             << "Sector id:   " << sectorID << "\n"
             << "Sector name: " << fMC->CurrentVolOffName(1) << "\n"
             << "Ring id:     " << ringID << "\n"
             << "Ring name:   " << fMC->CurrentVolOffName(0) << "\n"
             << "Cell id :    " << sectorID + 8 * ringID << "\n";

  return sectorID + 8 * ringID;
}

const bool Geometry::enableComponent(const EGeoComponent component, const bool enable)
{
  if (mEnabledComponents.find(component) == mEnabledComponents.end()) {
    LOG(DEBUG) << "FV0 Geometry::enableComponent(): Component not initialized and cannot be enabled/disabled!";
    return false;
  }

  return mEnabledComponents[component] = enable;
}

void Geometry::buildGeometry() const
{
  TGeoVolume* vALIC = gGeoManager->GetVolume("cave");
  if (!vALIC) {
    LOG(FATAL) << "FV0: Could not find the top volume";
  }

  // Top volume of FV0 detector
  TGeoVolumeAssembly* vFV0 = new TGeoVolumeAssembly(sDetectorName.c_str());
  LOG(INFO) << "FV0: Building geometry. FV0 volume name is '" << vFV0->GetName() << "'";

  assembleSensVols(vFV0);
  assembleNonSensVols(vFV0);

  // TODO: What should the copy_no be? Currently assigned to '1'
  vALIC->AddNode(vFV0, 1, new TGeoTranslation(sXGlobal, sYGlobal, sZGlobal));
}

void Geometry::initializeGeometry()
{
  initializeMaps();
  initializeVectors();
  initializeTransformations();
  initializeSensVols();
  initializeNonSensVols();
}

void Geometry::initializeMaps()
{
  const bool isFull = mGeometryType == eFull;
  const bool isRough = isFull || mGeometryType == eRough;
  const bool hasScint = isRough || mGeometryType == eOnlySensitive;
  mEnabledComponents.insert(std::pair<EGeoComponent, bool>(eScintillator, hasScint));
  mEnabledComponents.insert(std::pair<EGeoComponent, bool>(ePlastics, isRough));
  mEnabledComponents.insert(std::pair<EGeoComponent, bool>(eFibers, isFull));
  mEnabledComponents.insert(std::pair<EGeoComponent, bool>(eScrews, isFull));
  mEnabledComponents.insert(std::pair<EGeoComponent, bool>(eRods, isFull));
  mEnabledComponents.insert(std::pair<EGeoComponent, bool>(eContainer, isRough));
}

void Geometry::initializeVectors()
{
  initializeCellRingRadii();
  initializeSectorTransformations();
  initializeFiberVolumeRadii();
  initializeFiberMedium();
  initializeScrewAndRodRadii();
  initializeScrewTypeMedium();
  initializeRodTypeMedium();
  initializeScrewAndRodPositionsAndDimensions();
}

void Geometry::initializeCellRingRadii()
{
  // Index of mRAvgRing is NOT linked directly to any ring number
  mRAvgRing.assign(sCellRingRadii, sCellRingRadii + sNumberOfCellRings + 1);

  // Set real scintillator radii (reduced by paint thickness and separation gap)
  for (int i = 0; i < mRAvgRing.size() - 1; ++i) {
    mRMinScintillator.push_back(mRAvgRing[i] + sDrSeparationScint);
    mRMaxScintillator.push_back(mRAvgRing[i + 1] - sDrSeparationScint);
  }
  // Now indices of mRMinScint and mRMaxScint correspond to the same ring
}

void Geometry::initializeSectorTransformations()
{
  for (int iSector = 0; iSector < sNumberOfCellSectors; ++iSector) {
    // iSector = 0 corresponds to the first sector clockwise from the y-axis
    // iSector = 1 corresponds to the next sector in clockwise direction and so on

    TGeoRotation* trans = createAndRegisterRot(sDetectorName + sSectorName + std::to_string(iSector) + "TRANS");

    if (iSector == 2 || iSector == 3) {
      // "a" and "b" mirrors.
      // The reference to "a" and "b" can be understood with the CAD drawings of the detector.
      trans->ReflectY(true);
    }

    mSectorTrans.push_back(trans);
  }
}

void Geometry::initializeFiberVolumeRadii()
{
  mRMinFiber.push_back(0);
  mRMinFiber.push_back(sDrMinContainerCone + sEpsilon);
  mRMinFiber.push_back(sDrMinContainerFront + sEpsilon);

  mRMaxFiber.push_back(sDrMinContainerCone);
  mRMaxFiber.push_back(sDrMinContainerFront);
  mRMaxFiber.push_back(mRMaxScintillator.back());
}

void Geometry::initializeFiberMedium()
{
  // There are no further checks if the medium is actually found
  mMediumFiber.push_back(gGeoManager->GetMedium("FV0_FiberInner$"));
  mMediumFiber.push_back(gGeoManager->GetMedium("FV0_FiberMiddle$"));
  mMediumFiber.push_back(gGeoManager->GetMedium("FV0_FiberOuter$"));
}

void Geometry::initializeScrewAndRodRadii()
{
  mRScrewAndRod.push_back(mRAvgRing[1]);
  mRScrewAndRod.push_back(mRAvgRing[2]);
  mRScrewAndRod.push_back(mRAvgRing[3]);
  mRScrewAndRod.push_back(mRAvgRing[4]);
  mRScrewAndRod.push_back((mRAvgRing[4] + mRAvgRing[5]) / 2);
  mRScrewAndRod.push_back(mRAvgRing[5]);
}

void Geometry::initializeScrewTypeMedium()
{
  // There are no further checks if the medium is actually found
  TGeoMedium* medium = gGeoManager->GetMedium("FV0_Titanium$");
  for (int i = 0; i < sNumberOfScrewTypes; ++i) {
    mMediumScrewTypes.push_back(medium);
  }
}

void Geometry::initializeRodTypeMedium()
{
  // There are no further checks if the medium is actually found
  TGeoMedium* medium = gGeoManager->GetMedium("FV0_Aluminium$");
  for (int i = 0; i < sNumberOfRodTypes; ++i) {
    mMediumRodTypes.push_back(medium);
  }
}

void Geometry::addScrewProperties(const int screwTypeID, const int iRing, const float phi)
{
  float r = mRScrewAndRod[iRing];
  mScrewTypeIDs.push_back(screwTypeID);
  mScrewPos.push_back(std::vector<float>{cosf(phi * M_PI / 180) * r,
                                         sinf(phi * M_PI / 180) * r,
                                         sZScintillator - sDzScintillator / 2 + sZShiftScrew + sDzMaxScrewTypes[screwTypeID] / 2});
  mDrMinScrews.push_back(sDrMinScrewTypes[screwTypeID]);
  mDrMaxScrews.push_back(sDrMaxScrewTypes[screwTypeID]);
  mDzMaxScrews.push_back(sDzMaxScrewTypes[screwTypeID]);
  mDzMinScrews.push_back(sDzMinScrewTypes[screwTypeID]);
}

void Geometry::addRodProperties(const int rodTypeID, const int iRing)
{
  mRodTypeIDs.push_back(rodTypeID);
  mRodPos.push_back(std::vector<float>{sDxMinRodTypes[rodTypeID] / 2,
                                       mRScrewAndRod[iRing],
                                       sZScintillator - sDzScintillator / 2 + sZShiftRod + sDzMaxRodTypes[rodTypeID] / 2});
  mDxMinRods.push_back(sDxMinRodTypes[rodTypeID]);
  mDzMaxRods.push_back(sDxMaxRodTypes[rodTypeID]);
  mDyMinRods.push_back(sDyMinRodTypes[rodTypeID]);
  mDyMaxRods.push_back(sDyMaxRodTypes[rodTypeID]);
  mDzMaxRods.push_back(sDzMaxRodTypes[rodTypeID]);
  mDzMinRods.push_back(sDzMinRodTypes[rodTypeID]);

  mRodTypeIDs.push_back(rodTypeID);
  mRodPos.push_back(std::vector<float>{sDxMinRodTypes[rodTypeID] / 2,
                                       -mRScrewAndRod[iRing],
                                       sZScintillator - sDzScintillator / 2 + sZShiftRod + sDzMaxRodTypes[rodTypeID] / 2});
  mDxMinRods.push_back(sDxMinRodTypes[rodTypeID]);
  mDzMaxRods.push_back(sDxMaxRodTypes[rodTypeID]);
  mDyMinRods.push_back(sDyMinRodTypes[rodTypeID]);
  mDyMaxRods.push_back(sDyMaxRodTypes[rodTypeID]);
  mDzMaxRods.push_back(sDzMaxRodTypes[rodTypeID]);
  mDzMinRods.push_back(sDzMinRodTypes[rodTypeID]);
}

void Geometry::initializeScrewAndRodPositionsAndDimensions()
{
  for (int iRing = 0; iRing < mRScrewAndRod.size(); ++iRing) {
    switch (iRing) {
      case 0:
        addRodProperties(0, iRing);
        for (float phi = 45; phi >= -45; phi -= 45) {
          addScrewProperties(0, iRing, phi);
        }
        break;
      case 1:
        addRodProperties(0, iRing);
        for (float phi = 45; phi >= -45; phi -= 45) {
          addScrewProperties(1, iRing, phi);
        }
        break;
      case 2:
        addRodProperties(1, iRing);
        for (float phi = 67.5; phi >= -67.5; phi -= 22.5) {
          addScrewProperties(2, iRing, phi);
        }
        break;
      case 3:
        addRodProperties(2, iRing);
        for (float phi = 67.5; phi >= -67.5; phi -= 22.5) {
          addScrewProperties(3, iRing, phi);
        }
        break;
      case 4:
        addRodProperties(3, iRing);
        for (float phi = 45; phi >= -45; phi -= 45) {
          addScrewProperties(4, iRing, phi);
        }
        break;
      case 5:
        addRodProperties(3, iRing);
        for (float phi = 67.5; phi >= -67.5; phi -= 22.5) {
          addScrewProperties(5, iRing, phi);
        }
        break;
      default:
        break;
    }
  }
}

void Geometry::initializeTransformations()
{
  TGeoTranslation leftTranslation(-sDxHalvesSeparation / 2, 0, 0);
  TGeoRotation leftRotation;
  leftRotation.ReflectX(true);
  TGeoHMatrix leftTotalTransformation = leftTranslation * leftRotation;

  mLeftTransformation = new TGeoHMatrix(leftTotalTransformation);
  mRightTransformation = new TGeoTranslation(sDxHalvesSeparation / 2, sDyHalvesSeparation, sDzHalvesSeparation);
}

void Geometry::initializeSensVols()
{
  initializeScintCells();
}

void Geometry::initializeNonSensVols()
{
  initializeScrewHoles();
  initializeRodHoles();
  initializePlasticCells();
  initializeFibers();
  initializeScrews();
  initializeRods();
  initializeMetalContainer();
}

void Geometry::initializeScrewHoles()
{
  std::string boolFormula = "";

  for (int i = 0; i < mScrewPos.size(); ++i) {
    std::string holeShapeName = sDetectorName + sScrewName + "HOLE" + std::to_string(i);
    std::string holeTransName = sDetectorName + sScrewName + "HOLETRANS" + std::to_string(i);

    createScrewShape(holeShapeName, mScrewTypeIDs[i], sEpsilon, sEpsilon);
    createAndRegisterTrans(holeTransName, mScrewPos[i][0] + sXShiftScrews, mScrewPos[i][1], mScrewPos[i][2]);

    boolFormula += ((i != 0) ? "+" : "") + holeShapeName + ":" + holeTransName;
  }

  new TGeoCompositeShape(sScrewHolesCSName.c_str(), boolFormula.c_str());
}

void Geometry::initializeRodHoles()
{
  std::string boolFormula = "";

  for (int i = 0; i < mRodPos.size(); ++i) {
    std::string holeShapeName = sDetectorName + sRodName + "HOLE" + std::to_string(i);
    std::string holeTransName = sDetectorName + sRodName + "HOLETRANS" + std::to_string(i);

    createRodShape(holeShapeName, mRodTypeIDs[i], sEpsilon, sEpsilon);
    createAndRegisterTrans(holeTransName, mRodPos[i][0] + sXShiftScrews, mRodPos[i][1], mRodPos[i][2]);

    boolFormula += ((i != 0) ? "+" : "") + holeShapeName + ":" + holeTransName;
  }

  new TGeoCompositeShape(sRodHolesCSName.c_str(), boolFormula.c_str());
}

void Geometry::initializeCells(const std::string& cellType, const float zThickness, const TGeoMedium* medium,
                               const bool isSensitive)
{
  // Creating the two types of cells, "a" and "b", for each ring.
  // All sectors can be assembled with these cells.
  //
  // The reference to "a" and "b" can be understood with the CAD drawings of the detector.

  const float dxHoleCut = sDxHoleExtensionScintillator; // width of extension of hole 1, 2 and 7 in the "a" cell
  const float xHole = sDrSeparationScint + dxHoleCut;   // x-placement of holes 1, 2 and 7 in the "a" cell

  // Sector separation gap shape
  const std::string secSepShapeName = "FV0_" + cellType + "SectorSeparation";
  new TGeoBBox(secSepShapeName.c_str(), mRMaxScintillator.back() + sEpsilon, sDrSeparationScint, zThickness / 2);

  // Sector separation gap rotations
  const std::string secSepRot45Name = "FV0_" + cellType + "SecSepRot45";
  const std::string secSepRot90Name = "FV0_" + cellType + "SecSepRot90";

  createAndRegisterRot(secSepRot45Name, 45, 0, 0);
  createAndRegisterRot(secSepRot90Name, 90, 0, 0);

  // Hole shapes
  const std::string holeSmallName = "FV0_" + cellType + "HoleSmall";
  const std::string holeLargeName = "FV0_" + cellType + "HoleLarge";
  const std::string holeSmallCutName = "FV0_" + cellType + "HoleSmallCut";
  const std::string holeLargeCutName = "FV0_" + cellType + "HoleLargeCut";

  new TGeoTube(holeSmallName.c_str(), 0, sDrHoleSmallScintillator, zThickness / 2);
  new TGeoTube(holeLargeName.c_str(), 0, sDrHoleLargeScintillator, zThickness / 2);
  new TGeoBBox(holeSmallCutName.c_str(), dxHoleCut, sDrHoleSmallScintillator, zThickness / 2);
  new TGeoBBox(holeLargeCutName.c_str(), dxHoleCut, sDrHoleLargeScintillator, zThickness / 2);

  for (int ir = 0; ir < sNumberOfCellRings; ++ir) {
    // Radii without separation
    const float rMin = mRAvgRing[ir];
    const float rMax = mRAvgRing[ir + 1];
    const float rMid = rMin + (rMax - rMin) / 2;

    // "a"-type cell
    //
    // Initial placement:
    //
    // y
    // ^
    // |  1******
    // |  ************5
    // |  7*****************
    // |  *********************3
    // |  *******************
    // |  2**************8
    // |      6********
    // |        **4
    // |
    // |
    // |  O
    // ------------------------> x
    //
    // * = cell volume
    // numbers = hole numbers (as numbered in the code below)
    // O = beam pipe

    const std::string aCellName = createVolumeName(cellType + sCellName + "a", ir);

    // Base shape
    const std::string aCellShapeName = aCellName + "Shape";

    // The cells in the innermost ring have a slightly shifted inner radius origin.
    if (ir == 0) {
      // The innermost "a"-type cell
      const std::string a1CellShapeFullName = aCellShapeName + "Full";
      const std::string a1CellShapeHoleCutName = aCellShapeName + "HoleCut";
      const std::string a1CellShapeHoleCutTransName = a1CellShapeHoleCutName + "Trans";

      new TGeoTubeSeg(a1CellShapeFullName.c_str(), 0, mRMaxScintillator[ir], zThickness / 2 - sEpsilon, 45, 90);
      new TGeoTube(a1CellShapeHoleCutName.c_str(), 0, mRMinScintillator[ir], zThickness);

      createAndRegisterTrans(a1CellShapeHoleCutTransName, sXShiftInnerRadiusScintillator, 0, 0);

      const std::string a1BoolFormula = a1CellShapeFullName + "-" + a1CellShapeHoleCutName + ":" + a1CellShapeHoleCutTransName;
      new TGeoCompositeShape(aCellShapeName.c_str(), a1BoolFormula.c_str());
    } else {
      // The rest of the "a"-type cells
      new TGeoTubeSeg(aCellShapeName.c_str(), mRMinScintillator[ir], mRMaxScintillator[ir], zThickness / 2, 45, 90);
    }

    // Translations for screw holes (inner = rmin, half-length = rmid, outer = rmax)
    //
    // 1 = outer left
    // 2 = inner left
    // 3 = outer right
    // 4 = inner right
    // 5 = outer middle
    // 6 = inner middle
    // 7 = half-length left
    // 8 = half-length right
    //
    // holes 1, 2 and 7 are slightly shifted along the rim of the cell

    const std::string aHole1TransName = aCellName + "Hole1Trans";
    const std::string aHole2TransName = aCellName + "Hole2Trans";
    const std::string aHole3TransName = aCellName + "Hole3Trans";
    const std::string aHole4TransName = aCellName + "Hole4Trans";
    const std::string aHole5TransName = aCellName + "Hole5Trans";
    const std::string aHole6TransName = aCellName + "Hole6Trans";
    const std::string aHole7TransName = aCellName + "Hole7Trans";
    const std::string aHole8TransName = aCellName + "Hole8Trans";
    const std::string aHole1CutTransName = aCellName + "Hole1CutTrans";
    const std::string aHole2CutTransName = aCellName + "Hole2CutTrans";
    const std::string aHole7CutTransName = aCellName + "Hole7CutTrans";

    createAndRegisterTrans(aHole1TransName, xHole, cos(asin(xHole / rMax)) * rMax, 0);
    createAndRegisterTrans(aHole2TransName, xHole, cos(asin(xHole / rMin)) * rMin, 0);
    createAndRegisterTrans(aHole3TransName, sin(45 * M_PI / 180) * rMax, cos(45 * M_PI / 180) * rMax, 0);
    createAndRegisterTrans(aHole4TransName, sin(45 * M_PI / 180) * rMin, cos(45 * M_PI / 180) * rMin, 0);
    createAndRegisterTrans(aHole5TransName, sin(22.5 * M_PI / 180) * rMax, cos(22.5 * M_PI / 180) * rMax, 0);
    createAndRegisterTrans(aHole6TransName, sin(22.5 * M_PI / 180) * rMin, cos(22.5 * M_PI / 180) * rMin, 0);
    createAndRegisterTrans(aHole7TransName, xHole, cos(asin(xHole / rMid)) * rMid, 0);
    createAndRegisterTrans(aHole8TransName, sin(45 * M_PI / 180) * rMid, cos(45 * M_PI / 180) * rMid, 0);
    createAndRegisterTrans(aHole1CutTransName, 0, cos(asin(xHole / rMax)) * rMax, 0);
    createAndRegisterTrans(aHole2CutTransName, 0, cos(asin(xHole / rMin)) * rMin, 0);
    createAndRegisterTrans(aHole7CutTransName, 0, cos(asin(xHole / rMid)) * rMid, 0);

    // Composite shape
    std::string aBoolFormula = aCellShapeName;

    // sector separation
    aBoolFormula += "-" + secSepShapeName + ":" + secSepRot45Name;
    aBoolFormula += "-" + secSepShapeName + ":" + secSepRot90Name;

    // outer holes
    aBoolFormula += "-" + ((ir < 2) ? holeSmallName : holeLargeName) + ":" + aHole1TransName;
    aBoolFormula += "-" + ((ir < 2) ? holeSmallCutName : holeLargeCutName) + ":" + aHole1CutTransName;
    aBoolFormula += "-" + ((ir < 2) ? holeSmallName : holeLargeName) + ":" + aHole3TransName;

    // inner holes
    if (ir > 0) {
      const std::string screwHoleName = (ir < 3) ? holeSmallName : holeLargeName;
      const std::string screwHoleCutName = (ir < 3) ? holeSmallCutName : holeLargeCutName;

      aBoolFormula += "-" + screwHoleName + ":" + aHole2TransName;
      aBoolFormula += "-" + screwHoleCutName + ":" + aHole2CutTransName;
      aBoolFormula += "-" + screwHoleName + ":" + aHole4TransName;
    }

    // outer middle hole
    if (ir > 1) {
      aBoolFormula += "-" + holeLargeName + ":" + aHole5TransName;
    }

    // inner middle hole
    if (ir > 2) {
      aBoolFormula += "-" + holeLargeName + ":" + aHole6TransName;
    }

    // half-length holes
    if (ir == 4) {
      aBoolFormula += "-" + holeLargeName + ":" + aHole7TransName;
      aBoolFormula += "-" + holeLargeCutName + ":" + aHole7CutTransName;
      aBoolFormula += "-" + holeLargeName + ":" + aHole8TransName;
    }

    const std::string aCellCSName = aCellName + "CS";
    const TGeoCompositeShape* aCellCs = new TGeoCompositeShape(aCellCSName.c_str(), aBoolFormula.c_str());

    // Cell volume
    const TGeoVolume* aCell = new TGeoVolume(aCellName.c_str(), aCellCs, medium);

    // "b"-type cell
    //
    // Initial placement:
    //
    // y
    // ^
    // |                    1
    // |                 *****
    // |               *********
    // |            7*************
    // |          ****************5
    // |        2*******************
    // |           6*****************
    // |            *****************
    // |  O          4*******8******3
    // |
    // ------------------------------> x
    //
    // * = cell volume
    // numbers = number of holes (as numbered in the code below)
    // O = beam pipe

    const std::string bCellName = createVolumeName(cellType + sCellName + "b", ir);

    // Base shape
    const std::string bCellShapeName = bCellName + "Shape";

    // The cells in the innermost ring are slightly different than the rest
    if (ir == 0) {
      // The innermost "b"-type cell
      const std::string b1CellShapeFullName = bCellShapeName + "Full";
      const std::string b1CellShapeHoleCutName = bCellShapeName + "Cut";
      const std::string b1CellShapeHoleCutTransName = b1CellShapeHoleCutName + "Trans";

      new TGeoTubeSeg(b1CellShapeFullName.c_str(), 0, mRMaxScintillator[ir], zThickness / 2 - sEpsilon, 0, 45);
      new TGeoTube(b1CellShapeHoleCutName.c_str(), 0, mRMinScintillator[ir], zThickness);

      createAndRegisterTrans(b1CellShapeHoleCutTransName, sXShiftInnerRadiusScintillator, 0, 0);

      const std::string b1BoolFormula = b1CellShapeFullName + "-" + b1CellShapeHoleCutName + ":" + b1CellShapeHoleCutTransName;
      new TGeoCompositeShape(bCellShapeName.c_str(), b1BoolFormula.c_str());
    } else {
      // The rest of the "b"-type cells
      new TGeoTubeSeg(bCellShapeName.c_str(), mRMinScintillator[ir], mRMaxScintillator[ir], zThickness / 2, 0, 45);
    }

    // Translations for holes
    //
    // 1 = outer left
    // 2 = inner left
    // 3 = outer right
    // 4 = inner right
    // 5 = outer middle
    // 6 = inner middle
    // 7 = half-lenght left
    // 8 = half-length right

    const std::string bHole1TransName = bCellName + "Hole1Trans";
    const std::string bHole2TransName = bCellName + "Hole2Trans";
    const std::string bHole3TransName = bCellName + "Hole3Trans";
    const std::string bHole4TransName = bCellName + "Hole4Trans";
    const std::string bHole5TransName = bCellName + "Hole5Trans";
    const std::string bHole6TransName = bCellName + "Hole6Trans";
    const std::string bHole7TransName = bCellName + "Hole7Trans";
    const std::string bHole8TransName = bCellName + "Hole8Trans";

    createAndRegisterTrans(bHole1TransName, sin(45 * M_PI / 180) * rMax, cos(45 * M_PI / 180) * rMax, 0);
    createAndRegisterTrans(bHole2TransName, sin(45 * M_PI / 180) * rMin, cos(45 * M_PI / 180) * rMin, 0);
    createAndRegisterTrans(bHole3TransName, rMax, 0, 0);
    createAndRegisterTrans(bHole4TransName, rMin, 0, 0);
    createAndRegisterTrans(bHole5TransName, cos(22.5 * M_PI / 180) * rMax, sin(22.5 * M_PI / 180) * rMax, 0);
    createAndRegisterTrans(bHole6TransName, cos(22.5 * M_PI / 180) * rMin, sin(22.5 * M_PI / 180) * rMin, 0);
    createAndRegisterTrans(bHole7TransName, sin(45 * M_PI / 180) * rMid, cos(45 * M_PI / 180) * rMid, 0);
    createAndRegisterTrans(bHole8TransName, rMid, 0, 0);

    // Composite shape
    std::string bBoolFormula = bCellShapeName;

    // sector separation
    bBoolFormula += "-" + secSepShapeName;
    bBoolFormula += "-" + secSepShapeName + ":" + secSepRot45Name;

    // outer holes
    bBoolFormula += "-" + ((ir < 2) ? holeSmallName : holeLargeName) + ":" + bHole1TransName;
    bBoolFormula += "-" + ((ir < 2) ? holeSmallName : holeLargeName) + ":" + bHole3TransName;

    // inner holes
    if (ir > 0) {
      const std::string holeName = (ir < 3) ? holeSmallName : holeLargeName;

      bBoolFormula += "-" + holeName + ":" + bHole2TransName;
      bBoolFormula += "-" + holeName + ":" + bHole4TransName;
    }

    // outer middle hole
    if (ir > 1) {
      bBoolFormula += "-" + holeLargeName + ":" + bHole5TransName;
    }

    // inner middle hole
    if (ir > 2) {
      bBoolFormula += "-" + holeLargeName + ":" + bHole6TransName;
    }

    // half-lenght holes
    if (ir == 4) {
      bBoolFormula += "-" + holeLargeName + ":" + bHole7TransName;
      bBoolFormula += "-" + holeLargeName + ":" + bHole8TransName;
    }

    const std::string bCellCSName = bCellName + "CS";
    const TGeoCompositeShape* bCellCs = new TGeoCompositeShape(bCellCSName.c_str(), bBoolFormula.c_str());

    // Cell volume
    const TGeoVolume* bCell = new TGeoVolume(bCellName.c_str(), bCellCs, medium);

    if (isSensitive) {
      mSensitiveVolumeNames.push_back(aCell->GetName());
      mSensitiveVolumeNames.push_back(bCell->GetName());
    }
  }
}

void Geometry::initializeScintCells()
{
  const TGeoMedium* medium = gGeoManager->GetMedium("FV0_Scintillator$");
  initializeCells(sScintillatorName, sDzScintillator, medium, true);
}

void Geometry::initializePlasticCells()
{
  const TGeoMedium* medium = gGeoManager->GetMedium("FV0_Plastic$");
  initializeCells(sPlasticName, sDzPlastic, medium, false);
}

void Geometry::initializeFibers()
{
  // depth of the fiber volumes
  const float dzFibers = sDzContainer - sDzContainerBack - sDzContainerFront - sDzScintillator - sDzPlastic - 2 * sEpsilon;

  const std::string fiberName = "FV0_Fibers"; // No volume with this name

  const std::string fiberSepCutName = fiberName + "SepCut";
  const std::string fiberConeCutName = fiberName + "ConeCut";
  const std::string fiberHoleCutName = fiberName + "HoleCut";

  const std::string fiberTransName = fiberName + "Trans";
  const std::string fiberConeCutTransName = fiberConeCutName + "Trans";
  const std::string fiberHoleCutTransName = fiberHoleCutName + "Trans";

  new TGeoBBox(fiberSepCutName.c_str(), sDrSeparationScint, mRMaxFiber.back() + sEpsilon, dzFibers / 2 + sEpsilon);
  new TGeoConeSeg(fiberConeCutName.c_str(), sDzContainerCone / 2 + sEpsilon, 0,
                  sDrMinContainerCone + sXYThicknessContainerCone + sEpsilon, 0, sDrMinContainerFront + sEpsilon, -90, 90);
  new TGeoTube(fiberHoleCutName.c_str(), 0, mRMinScintillator.front(), dzFibers / 2 + sEpsilon);

  createAndRegisterTrans(fiberTransName, sXScintillator, 0, sZFiber);
  createAndRegisterTrans(fiberConeCutTransName, sXScintillator, 0, sZCone);
  createAndRegisterTrans(fiberHoleCutTransName, sXScintillator + sXShiftInnerRadiusScintillator, 0, sZFiber);

  for (int i = 0; i < mRMinFiber.size(); ++i) {
    const std::string fiberShapeName = fiberName + std::to_string(i + 1);
    new TGeoTubeSeg(fiberShapeName.c_str(), mRMinFiber[i], mRMaxFiber[i] - sEpsilon, dzFibers / 2, -90, 90);

    // Composite shape
    std::string boolFormula = "";
    boolFormula += fiberShapeName + ":" + fiberTransName;
    boolFormula += "-" + fiberSepCutName + ":" + fiberTransName;
    boolFormula += "-" + fiberConeCutName + ":" + fiberConeCutTransName;

    if (i == 0) {
      // Cut out the hole in the innermost fiber volume
      boolFormula += "-" + fiberHoleCutName + ":" + fiberHoleCutTransName;
    }

    // Remove holes for screws and rods
    boolFormula += "-" + sScrewHolesCSName;
    boolFormula += "-" + sRodHolesCSName;

    const TGeoCompositeShape* fiberCS = new TGeoCompositeShape((fiberShapeName + "CS").c_str(), boolFormula.c_str());

    // If modifying materials, make sure the appropriate initialization is done in initializeXxxMedium() methods
    new TGeoVolume(createVolumeName(sFiberName, i).c_str(), fiberCS, mMediumFiber[i]);
  }
}

void Geometry::initializeScrews()
{
  for (int i = 0; i < sNumberOfScrewTypes; ++i) {
    const std::string screwName = createVolumeName(sScrewName, i);
    const TGeoShape* screwShape = createScrewShape(screwName + "Shape", i, 0, 0, 0);

    // If modifying materials, make sure the appropriate initialization is done in initializeXxxMedium() methods
    new TGeoVolume(screwName.c_str(), screwShape, mMediumScrewTypes[i]);
  }
}

void Geometry::initializeRods()
{
  for (int i = 0; i < sNumberOfRodTypes; ++i) {
    const std::string rodName = createVolumeName(sRodName, i);
    const TGeoShape* rodShape = createRodShape(rodName + "Shape", i, -sEpsilon, -sEpsilon);

    // If modifying materials, make sure the appropriate initialization is done in initializeXxxMedium() methods
    new TGeoVolume(rodName.c_str(), rodShape, mMediumRodTypes[i]);
  }
}

void Geometry::initializeMetalContainer()
{
  // The metal container is constructed starting from the backplate. The backplate is positioned first, relative to
  // the scintillator cells. The rest of the container parts are positioned relative to the backplate.
  //
  // Caution: some position variables are in global coords, and some are relative to some other part of the container

  // Backplate

  const std::string backPlateName = "FV0_BackPlate";                        // the full backplate
  const std::string backPlateStandName = backPlateName + "Stand";           // the stand part of the backplate
  const std::string backPlateHoleName = backPlateName + "Hole";             // the hole in the middle of the backplate
  const std::string backPlateHoleCutName = backPlateHoleName + "Cut";       // extension of the hole
  const std::string backPlateStandTransName = backPlateStandName + "Trans"; // shift of the backplate stand
  const std::string backPlateHoleTransName = backPlateHoleName + "Trans";   // shift of the backplate inner radius

  new TGeoTubeSeg(backPlateName.c_str(), 0, sDrMaxContainerBack, sDzContainerBack / 2, -90, 90);
  new TGeoBBox(backPlateStandName.c_str(), sDxContainerStand / 2, (sDrMaxContainerBack + sDyContainerStand) / 2,
               sDzContainerBack / 2);
  new TGeoTubeSeg(backPlateHoleName.c_str(), 0, sDrContainerHole, sDzContainerBack / 2, -90, 90);
  new TGeoBBox(backPlateHoleCutName.c_str(), -sXShiftContainerHole, sDrContainerHole, sDzContainerBack);

  createAndRegisterTrans(backPlateStandTransName, sDxContainerStand / 2,
                         -(sDrMaxContainerBack + sDyContainerStand) / 2, 0);
  createAndRegisterTrans(backPlateHoleTransName, sXShiftContainerHole, 0, 0);

  // Backplate composite shape
  std::string backPlateBoolFormula = "";
  backPlateBoolFormula += backPlateName;
  backPlateBoolFormula += "+" + backPlateStandName + ":" + backPlateStandTransName;
  backPlateBoolFormula += "-" + backPlateHoleName + ":" + backPlateHoleTransName;
  backPlateBoolFormula += "-" + backPlateHoleCutName;

  const std::string backPlateCSName = backPlateName + "CS";
  const std::string backPlateCSTransName = backPlateCSName + "Trans";

  new TGeoCompositeShape(backPlateCSName.c_str(), backPlateBoolFormula.c_str());
  createAndRegisterTrans(backPlateCSTransName, 0, 0, sZContainerBack);

  // Frontplate

  // the z-position o the frontplate
  const float zPosFrontPlate = sZContainerFront;
  // the height of the total stand overlapping with the rest of the plate
  const float dyFrontPlateStand = sDyContainerStand + (sDrMaxContainerFront - sDrMinContainerFront) / 2;
  // the y-position of the total stand
  const float yPosFrontPlateStand = -sDrMaxContainerFront - sDyContainerStand + dyFrontPlateStand / 2;

  const std::string frontPlateName = "FV0_FrontPlate";
  const std::string frontPlateStandName = frontPlateName + "Stand";
  const std::string frontPlateTransName = frontPlateName + "Trans";
  const std::string frontPlateStandTransName = frontPlateStandName + "Trans";

  new TGeoTubeSeg(frontPlateName.c_str(), sDrMinContainerFront, sDrMaxContainerFront, sDzContainerFront / 2, -90, 90);
  new TGeoBBox(frontPlateStandName.c_str(), sDxContainerStand / 2, dyFrontPlateStand / 2, sDzContainerBack / 2);

  createAndRegisterTrans(frontPlateTransName, 0, 0, zPosFrontPlate);
  createAndRegisterTrans(frontPlateStandTransName, sDxContainerStand / 2, yPosFrontPlateStand, 0);

  // Frontplate cone composite shape
  std::string frontPlateBoolFormula = "";
  frontPlateBoolFormula += frontPlateName;
  frontPlateBoolFormula += "+" + frontPlateStandName + ":" + frontPlateStandTransName;

  const std::string frontPlateCSName = frontPlateName + "CS";

  new TGeoCompositeShape(frontPlateCSName.c_str(), frontPlateBoolFormula.c_str());

  // Frontplate cone

  // radial thickness of frontplate cone in the xy-plane
  const float thicknessFrontPlateCone = sXYThicknessContainerCone;
  // z-position of the frontplate cone relative to the frontplate
  const float zPosCone = sDzContainerFront / 2 - sDzContainerCone / 2;

  const std::string frontPlateConeName = "FV0_FrontPlateCone";                // no volume with this name
  const std::string frontPlateConeShieldName = frontPlateConeName + "Shield"; // the "sides" of the cone
  const std::string frontPlateConeShieldTransName = frontPlateConeShieldName + "Trans";

  new TGeoConeSeg(frontPlateConeShieldName.c_str(), sDzContainerCone / 2, sDrMinContainerCone,
                  sDrMinContainerCone + thicknessFrontPlateCone, sDrMinContainerFront - thicknessFrontPlateCone,
                  sDrMinContainerFront,
                  -90, 90);
  createAndRegisterTrans(frontPlateConeShieldTransName, 0, 0, zPosCone);

  // Frontplate cone "bottom"

  // z-position of the cone bottom relative to the frontplate
  const float zPosConePlate = sDzContainerFront / 2 - sDzContainerCone + thicknessFrontPlateCone / 2;
  // the bottom of the cone
  const std::string frontPlateConePlateName = frontPlateConeName + "Plate";

  new TGeoTubeSeg(frontPlateConePlateName.c_str(), 0, sDrMinContainerCone + thicknessFrontPlateCone,
                  thicknessFrontPlateCone / 2, -90, 90);

  // Frontplate cone bottom composite shape
  std::string frontPlateConePlateCSBoolFormula;
  frontPlateConePlateCSBoolFormula += frontPlateConePlateName;
  frontPlateConePlateCSBoolFormula += "-" + backPlateHoleName + ":" + backPlateHoleTransName;

  const std::string frontPlateConePlateCSName = frontPlateConePlateName + "CS";
  const std::string frontPlateConePlateCSTransName = frontPlateConePlateCSName + "Trans";
  new TGeoCompositeShape(frontPlateConePlateCSName.c_str(), frontPlateConePlateCSBoolFormula.c_str());
  createAndRegisterTrans(frontPlateConePlateCSTransName, 0, 0, zPosConePlate);

  // Frontplate cone composite shape
  std::string frontPlateConeCSBoolFormula = "";
  frontPlateConeCSBoolFormula += frontPlateConeShieldName + ":" + frontPlateConeShieldTransName;
  frontPlateConeCSBoolFormula += "+" + frontPlateConePlateCSName + ":" + frontPlateConePlateCSTransName;

  const std::string frontPlateConeCSName = frontPlateConeName + "CS";
  new TGeoCompositeShape(frontPlateConeCSName.c_str(), frontPlateConeCSBoolFormula.c_str());

  // Shields
  const float dzShieldGap = 0.7;                         // z-distance between the shields and the front- and backplate outer edges (in z-direction)
  const float dzShield = sDzContainer - 2 * dzShieldGap; // depth of the shields

  // Outer shield
  const float zPosOuterShield = (sZContainerBack + sZContainerFront) / 2; // z-position of the outer shield

  const std::string outerShieldName = "FV0_OuterShield";
  const std::string outerShieldTransName = outerShieldName + "Trans";

  new TGeoTubeSeg(outerShieldName.c_str(), sDrMinContainerOuterShield, sDrMaxContainerOuterShield, dzShield / 2, -90,
                  90);
  createAndRegisterTrans(outerShieldTransName, 0, 0, zPosOuterShield);

  // Inner shield
  const float dzInnerShield = sDzContainer - sDzContainerCone - dzShieldGap;                              // depth of the inner shield
  const float zPosInnerShield = sZContainerBack - sDzContainerBack / 2 + dzShieldGap + dzInnerShield / 2; // z-position of the inner shield relative to the backplate

  const std::string innerShieldName = "FV0_InnerShield";
  const std::string innerShieldCutName = innerShieldName + "Cut";

  new TGeoTubeSeg(innerShieldName.c_str(), sDrMinContainerInnerShield, sDrMaxContainerInnerShield, dzInnerShield / 2, -90, 90);
  new TGeoBBox(innerShieldCutName.c_str(), fabs(sXShiftContainerHole), sDrMaxContainerInnerShield, dzInnerShield / 2);

  // Inner shield composite shape
  std::string innerShieldCSBoolFormula;
  innerShieldCSBoolFormula = innerShieldName;
  innerShieldCSBoolFormula += "-" + innerShieldCutName;

  const std::string innerShieldCSName = innerShieldName + "CS";
  const std::string innerShieldCSTransName = innerShieldCSName + "Trans";
  new TGeoCompositeShape(innerShieldCSName.c_str(), innerShieldCSBoolFormula.c_str());
  createAndRegisterTrans(innerShieldCSTransName, sXShiftContainerHole, 0, zPosInnerShield);

  // Cover
  const float dzCover = sDzContainer;                       // Depth of the covers
  const float zPosCoverConeCut = zPosFrontPlate + zPosCone; // Set the cone cut relative to the frontplate so that the exact position of the aluminium cone part can be used.

  const std::string coverName = "FV0_Cover";
  const std::string coverConeCutName = coverName + "ConeCut";
  const std::string coverHoleCutName = coverName + "HoleCut";

  new TGeoBBox(coverName.c_str(), sDxContainerCover / 2, sDrMaxContainerOuterShield, dzCover / 2);
  new TGeoCone(coverConeCutName.c_str(), sDzContainerCone / 2, 0, sDrMinContainerCone + thicknessFrontPlateCone, 0,
               sDrMinContainerFront);
  new TGeoTubeSeg(coverHoleCutName.c_str(), 0, sDrMinContainerInnerShield, dzCover / 2, 0, 360);

  const std::string coverTransName = coverName + "Trans";
  const std::string coverConeCutTransName = coverConeCutName + "Trans";
  const std::string coverHoleCutTransName = coverHoleCutName + "Trans";

  createAndRegisterTrans(coverTransName, sDxContainerCover / 2, 0, zPosOuterShield);
  createAndRegisterTrans(coverConeCutTransName, 0, 0, zPosCoverConeCut);
  createAndRegisterTrans(coverHoleCutTransName.c_str(), sXShiftContainerHole, 0, zPosOuterShield);

  // Cover composite shape
  std::string coverCSBoolFormula = "";
  coverCSBoolFormula += coverName + ":" + coverTransName;
  coverCSBoolFormula += "-" + coverConeCutName + ":" + coverConeCutTransName;
  coverCSBoolFormula += "-" + coverHoleCutName + ":" + coverHoleCutTransName;

  const std::string coverCSName = coverName + "CS";
  new TGeoCompositeShape(coverCSName.c_str(), coverCSBoolFormula.c_str());

  // Stand bottom
  const float dzStandBottom = sDzContainer - sDzContainerBack - sDzContainerFront;
  const float dyStandBottomGap = 0.5; // This bottom part is not vertically aligned with the "front and backplate stands"
  const float dxStandBottomHole = 9.4;
  const float dzStandBottomHole = 20.4;
  const float dxStandBottomHoleSpacing = 3.1;

  const std::string standName = "FV0_StandBottom";
  const std::string standHoleName = standName + "Hole";

  new TGeoBBox(standName.c_str(), sDxContainerStandBottom / 2, sDyContainerStandBottom / 2, dzStandBottom / 2);
  new TGeoBBox(standHoleName.c_str(), dxStandBottomHole / 2, sDyContainerStandBottom / 2 + sEpsilon,
               dzStandBottomHole / 2);

  const std::string standHoleTrans1Name = standHoleName + "Trans1";
  const std::string standHoleTrans2Name = standHoleName + "Trans2";
  const std::string standHoleTrans3Name = standHoleName + "Trans3";

  createAndRegisterTrans(standHoleTrans1Name, -dxStandBottomHoleSpacing - dxStandBottomHole, 0, 0);
  createAndRegisterTrans(standHoleTrans2Name, 0, 0, 0);
  createAndRegisterTrans(standHoleTrans3Name, dxStandBottomHoleSpacing + dxStandBottomHole, 0, 0);

  // Stand bottom composite shape
  const std::string standCSName = standName + "CS";

  std::string standBoolFormula = "";
  standBoolFormula += standName;
  standBoolFormula += "-" + standHoleName + ":" + standHoleTrans1Name;
  standBoolFormula += "-" + standHoleName + ":" + standHoleTrans2Name;
  standBoolFormula += "-" + standHoleName + ":" + standHoleTrans3Name;

  new TGeoCompositeShape(standCSName.c_str(), standBoolFormula.c_str());

  const std::string standCSTransName = standCSName + "Trans";

  createAndRegisterTrans(standCSTransName.c_str(),
                         sDxContainerStand - sDxContainerStandBottom / 2,
                         -(sDrMaxContainerBack + sDyContainerStand) + sDyContainerStandBottom / 2 + dyStandBottomGap,
                         sZContainerMid);

  // Composite shape
  std::string boolFormula = "";
  boolFormula += backPlateCSName + ":" + backPlateCSTransName;
  boolFormula += "+" + frontPlateCSName + ":" + frontPlateTransName;
  boolFormula += "+" + frontPlateConeCSName + ":" + frontPlateTransName;
  boolFormula += "+" + outerShieldName + ":" + outerShieldTransName;
  boolFormula += "+" + innerShieldCSName + ":" + innerShieldCSTransName;
  boolFormula += "+" + coverCSName;
  boolFormula += "+" + standCSName + ":" + standCSTransName;
  boolFormula += "-" + sScrewHolesCSName; // Remove holes for screws
  boolFormula += "-" + sRodHolesCSName;   // Remove holes for rods

  const std::string aluContCSName = "FV0_AluContCS";
  const TGeoCompositeShape* aluContCS = new TGeoCompositeShape(aluContCSName.c_str(), boolFormula.c_str());

  // Volume
  const std::string aluContName = createVolumeName(sContainerName);
  const TGeoMedium* medium = gGeoManager->GetMedium("FV0_Aluminium$");
  new TGeoVolume(aluContName.c_str(), aluContCS, medium);
}

void Geometry::assembleSensVols(TGeoVolume* vFV0) const
{
  if (mEnabledComponents.at(eScintillator)) {
    assembleScintSectors(vFV0);
  }
}

void Geometry::assembleNonSensVols(TGeoVolume* vFV0) const
{
  if (mEnabledComponents.at(ePlastics)) {
    assemblePlasticSectors(vFV0);
  }
  if (mEnabledComponents.at(eFibers)) {
    assembleFibers(vFV0);
  }
  if (mEnabledComponents.at(eScrews)) {
    assembleScrews(vFV0);
  }
  if (mEnabledComponents.at(eRods)) {
    assembleRods(vFV0);
  }
  if (mEnabledComponents.at(eContainer)) {
    assembleMetalContainer(vFV0);
  }
}

void Geometry::assembleScintSectors(TGeoVolume* vFV0) const
{
  TGeoVolumeAssembly* sectors = buildSectorAssembly(sScintillatorName);
  vFV0->AddNode(sectors, 0, mRightTransformation);
  vFV0->AddNode(sectors, 1, mLeftTransformation);
}

void Geometry::assemblePlasticSectors(TGeoVolume* vFV0) const
{
  TGeoVolumeAssembly* sectors = buildSectorAssembly(sPlasticName);

  // Move the plastic cells next to the scintillator cells
  TGeoHMatrix* leftTrans = new TGeoHMatrix(*mLeftTransformation);
  leftTrans->SetDz(leftTrans->GetTranslation()[2] + sZPlastic);

  TGeoMatrix* rightTrans = new TGeoHMatrix(*mRightTransformation);
  rightTrans->SetDz(rightTrans->GetTranslation()[2] + sZPlastic);

  vFV0->AddNode(sectors, 0, rightTrans);
  vFV0->AddNode(sectors, 1, leftTrans);
}

void Geometry::assembleFibers(TGeoVolume* vFV0) const
{
  TGeoVolumeAssembly* fibers = new TGeoVolumeAssembly(createVolumeName("FIBERS").c_str());

  for (int i = 0; i < mRMinFiber.size(); ++i) {
    TGeoVolume* fiber = gGeoManager->GetVolume(createVolumeName(sFiberName, i).c_str());
    if (!fiber) {
      LOG(WARNING) << "FV0 Geometry::assembleFibers(): Fiber volume no. " << i << " not found.";
    }
    fibers->AddNode(fiber, i);
  }

  vFV0->AddNode(fibers, 1, mLeftTransformation);
  vFV0->AddNode(fibers, 2, mRightTransformation);
}

void Geometry::assembleScrews(TGeoVolume* vFV0) const
{
  TGeoVolumeAssembly* screws = new TGeoVolumeAssembly(createVolumeName("SCREWS").c_str());

  // If modifying something here, make sure screw initialization is OK
  for (int i = 0; i < mScrewPos.size(); ++i) {
    TGeoVolume* screw = gGeoManager->GetVolume(createVolumeName(sScrewName, mScrewTypeIDs[i]).c_str());
    if (!screw) {
      LOG(WARNING) << "FV0 Geometry::assembleScrews(): Screw no. " << i << " not found";
    } else {
      screws->AddNode(screw, i, new TGeoTranslation(mScrewPos[i][0] + sXShiftScrews, mScrewPos[i][1], mScrewPos[i][2]));
    }
  }

  vFV0->AddNode(screws, 1, mLeftTransformation);
  vFV0->AddNode(screws, 2, mRightTransformation);
}

void Geometry::assembleRods(TGeoVolume* vFV0) const
{
  TGeoVolumeAssembly* rods = new TGeoVolumeAssembly(createVolumeName("RODS").c_str());

  // If modifying something here, make sure rod initialization is OK
  for (int i = 0; i < mRodPos.size(); ++i) {
    TGeoVolume* rod = gGeoManager->GetVolume(createVolumeName(sRodName, mRodTypeIDs[i]).c_str());

    if (!rod) {
      LOG(INFO) << "FV0 Geometry::assembleRods(): Rod no. " << i << " not found";
    } else {
      rods->AddNode(rod, i, new TGeoTranslation(mRodPos[i][0] + sXShiftScrews, mRodPos[i][1], mRodPos[i][2]));
    }
  }

  vFV0->AddNode(rods, 1, mLeftTransformation);
  vFV0->AddNode(rods, 2, mRightTransformation);
}

void Geometry::assembleMetalContainer(TGeoVolume* volV0) const
{
  TGeoVolume* container = gGeoManager->GetVolume(createVolumeName(sContainerName).c_str());
  if (!container) {
    LOG(WARNING) << "FV0: Could not find container volume";
  } else {
    volV0->AddNode(container, 1, mLeftTransformation);
    volV0->AddNode(container, 2, mRightTransformation);
  }
}

TGeoVolumeAssembly* Geometry::buildSectorAssembly(const std::string& cellName) const
{
  TGeoVolumeAssembly* assembly = new TGeoVolumeAssembly(createVolumeName(cellName).c_str());

  for (int iSector = 0; iSector < mSectorTrans.size(); ++iSector) {
    TGeoVolumeAssembly* sector = buildSector(cellName, iSector);
    assembly->AddNode(sector, iSector, mSectorTrans[iSector]);
  }

  return assembly;
}

TGeoVolumeAssembly* Geometry::buildSector(const std::string& cellType, const int iSector) const
{
  TGeoVolumeAssembly* sector = new TGeoVolumeAssembly(createVolumeName(cellType + sSectorName, iSector).c_str());

  for (int i = 0; i < sNumberOfCellRings; ++i) {
    TGeoVolume* cell = gGeoManager->GetVolume(createVolumeName(cellType + sCellName + sCellTypes[iSector], i).c_str());

    if (!cell) {
      LOG(WARNING) << "FV0 Geometry::buildSector(): Couldn't find cell volume no. " << i;
    } else {
      sector->AddNode(cell, i, new TGeoTranslation(sXScintillator, 0, 0));
    }
  }

  return sector;
}

TGeoShape* Geometry::createScrewShape(const std::string& shapeName, const int screwTypeID, const float xEpsilon,
                                      const float yEpsilon, const float zEpsilon) const
{
  const float xyEpsilon = (fabs(xEpsilon) > fabs(yEpsilon)) ? xEpsilon : yEpsilon;
  const float dzMax = sDzMaxScrewTypes[screwTypeID] / 2 + zEpsilon;
  const float dzMin = sDzMinScrewTypes[screwTypeID] / 2 + zEpsilon;

  const std::string thinPartName = shapeName + "Thin";
  const std::string thickPartName = shapeName + "Thick";
  const std::string thickPartTransName = thickPartName + "Trans";

  new TGeoTube(thinPartName.c_str(), 0, sDrMinScrewTypes[screwTypeID] + xyEpsilon, dzMax);
  new TGeoTube(thickPartName.c_str(), 0, sDrMaxScrewTypes[screwTypeID] + xyEpsilon, dzMin);
  createAndRegisterTrans(thickPartTransName, 0, 0, -dzMax - sZShiftScrew + sDzScintillator + sDzPlastic + dzMin);

  std::string boolFormula = thinPartName;
  boolFormula += "+" + thickPartName + ":" + thickPartTransName;

  TGeoCompositeShape* screwShape = new TGeoCompositeShape(shapeName.c_str(), boolFormula.c_str());
  return screwShape;
}

TGeoShape* Geometry::createRodShape(const std::string& shapeName, const int rodTypeID, const float xEpsilon,
                                    const float yEpsilon, const float zEpsilon) const
{
  const float dxMin = sDxMinRodTypes[rodTypeID] / 2 + xEpsilon;
  const float dxMax = sDxMaxRodTypes[rodTypeID] / 2 + xEpsilon;
  const float dyMin = sDyMinRodTypes[rodTypeID] / 2 + yEpsilon;
  const float dyMax = sDyMaxRodTypes[rodTypeID] / 2 + yEpsilon;
  const float dzMax = sDzMaxRodTypes[rodTypeID] / 2 + zEpsilon;
  const float dzMin = sDzMinRodTypes[rodTypeID] / 2 + zEpsilon;

  const std::string thinPartName = shapeName + "Thin";
  const std::string thickPartName = shapeName + "Thick";
  const std::string thickPartTransName = thickPartName + "Trans";

  new TGeoBBox(thinPartName.c_str(), dxMin, dyMin, dzMax);
  new TGeoBBox(thickPartName.c_str(), dxMax, dyMax, dzMin);
  createAndRegisterTrans(thickPartTransName, dxMax - dxMin, 0,
                         -dzMax - sZShiftRod + sDzScintillator + sDzPlastic + dzMin);

  std::string boolFormula = thinPartName;
  boolFormula += "+" + thickPartName + ":" + thickPartTransName;

  TGeoCompositeShape* rodShape = new TGeoCompositeShape(shapeName.c_str(), boolFormula.c_str());
  return rodShape;
}

TGeoTranslation* Geometry::createAndRegisterTrans(const std::string& name, const double dx, const double dy,
                                                  const double dz) const
{
  TGeoTranslation* trans = new TGeoTranslation(name.c_str(), dx, dy, dz);
  trans->RegisterYourself();
  return trans;
}

TGeoRotation* Geometry::createAndRegisterRot(const std::string& name, const double phi, const double theta,
                                             const double psi) const
{
  TGeoRotation* rot = new TGeoRotation(name.c_str(), phi, theta, psi);
  rot->RegisterYourself();
  return rot;
}

const std::string Geometry::createVolumeName(const std::string& volumeType, const int number) const
{
  return sDetectorName + volumeType + ((number >= 0) ? std::to_string(number) : "");
}
