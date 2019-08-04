// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Ladder.cxx
/// \brief Ladder builder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"

#include "FairLogger.h"

#include "ITSMFTSimulation/AlpideChip.h"
#include "ITSMFTBase/SegmentationAlpide.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Flex.h"
#include "MFTBase/Ladder.h"
#include "MFTBase/Geometry.h"

using namespace o2::itsmft;
using namespace o2::mft;
using AlpideChip = o2::itsmft::AlpideChip;

ClassImp(o2::mft::Ladder);

// Units are cm
const Double_t Ladder::sLadderDeltaY = SegmentationAlpide::SensorSizeRows + 2. * Geometry::sSensorTopOffset;
const Double_t Ladder::sLadderDeltaZ =
  Geometry::sFlexThickness + Geometry::sChipThickness; // TODO: Adjust that value when adding glue layer

/// \brief Default constructor

//_____________________________________________________________________________
Ladder::Ladder() : TNamed(), mSegmentation(nullptr), mFlex(nullptr), mLadderVolume(nullptr) {}

/// \brief Constructor

//_____________________________________________________________________________
Ladder::Ladder(LadderSegmentation* segmentation)
  : TNamed(segmentation->GetName(), segmentation->GetName()), mSegmentation(segmentation), mFlex(nullptr)
{

  LOG(DEBUG1) << "Ladder " << Form("creating : %s", GetName()) << FairLogger::endl;
  mLadderVolume = new TGeoVolumeAssembly(GetName());
}

//_____________________________________________________________________________
Ladder::~Ladder() { delete mFlex; }

/// \brief Build the ladder

//_____________________________________________________________________________
TGeoVolume* Ladder::createVolume()
{

  Int_t nChips = mSegmentation->getNSensors();

  // Create the flex
  mFlex = new Flex(mSegmentation);
  Double_t flexLength = nChips * (SegmentationAlpide::SensorSizeCols + Geometry::sSensorInterspace) +
                        Geometry::sLadderOffsetToEnd + Geometry::sSensorSideOffset;
  Double_t shiftY =
    4 * Geometry::sSensorTopOffset + SegmentationAlpide::SensorSizeRows - Geometry::sFlexHeight / 2; // to be verified!!
  TGeoVolumeAssembly* flexVol = mFlex->makeFlex(mSegmentation->getNSensors(), flexLength);
  mLadderVolume->AddNode(flexVol, 1, new TGeoTranslation(flexLength / 2 + Geometry::sSensorSideOffset / 2, shiftY, Geometry::sFlexThickness / 2 - 2 * (Geometry::sKaptonOnCarbonThickness + Geometry::sKaptonGlueThickness)));

  // Create the CMOS Sensors
  createSensors();

  return mLadderVolume;
}

/// \brief Build the sensors

//_____________________________________________________________________________
void Ladder::createSensors()
{

  Geometry* mftGeom = Geometry::instance();

  // Create Shapes

  // sensor = sensitive volume
  TString namePrefixS = "MFTSensor";

  // chip = sensor + readout
  TString namePrefixC =
    Form("MFT_C_%d_%d_%d", mftGeom->getHalfID(mSegmentation->GetUniqueID()),
         mftGeom->getDiskID(mSegmentation->GetUniqueID()), mftGeom->getLadderID(mSegmentation->GetUniqueID()));

  // the MFT glue
  TString namePrefixG =
    Form("MFT_G_%d_%d_%d", mftGeom->getHalfID(mSegmentation->GetUniqueID()),
         mftGeom->getDiskID(mSegmentation->GetUniqueID()), mftGeom->getLadderID(mSegmentation->GetUniqueID()));

  TGeoMedium* kMedGlue = gGeoManager->GetMedium("MFT_SE4445$");

  TGeoVolume* glue = gGeoManager->MakeBox(
    namePrefixG.Data(), kMedGlue, (SegmentationAlpide::SensorSizeCols - Geometry::sGlueEdge) / 2.,
    (SegmentationAlpide::SensorSizeRows - Geometry::sGlueEdge) / 2., Geometry::sGlueThickness / 2.);
  glue->SetVisibility(kTRUE);
  glue->SetLineColor(kRed - 10);
  glue->SetLineWidth(1);
  glue->SetFillColor(glue->GetLineColor());
  glue->SetFillStyle(4000); // 0% transparent

  // common with ITS
  TGeoVolume* chipVol = AlpideChip::createChip(Geometry::sChipThickness / 2., Geometry::sSensorThickness / 2.,
                                               namePrefixC, namePrefixS, kFALSE);

  // chipVol->Print();

  for (int ichip = 0; ichip < mSegmentation->getNSensors(); ichip++) {

    ChipSegmentation* chipSeg = mSegmentation->getSensor(ichip);
    TGeoCombiTrans* chipPos = chipSeg->getTransformation();
    TGeoCombiTrans* chipPosGlue = chipSeg->getTransformation();

    // Position of the center on the chip in the chip coordinate system
    Double_t pos[3] = {SegmentationAlpide::SensorSizeCols / 2., SegmentationAlpide::SensorSizeRows / 2.,
                       Geometry::sChipThickness / 2. - Geometry::sGlueThickness - 2 * (Geometry::sKaptonOnCarbonThickness + Geometry::sKaptonGlueThickness)};

    Double_t posglue[3] = {SegmentationAlpide::SensorSizeCols / 2., SegmentationAlpide::SensorSizeRows / 2.,
                           Geometry::sGlueThickness / 2 - Geometry::sChipThickness - 2 * (Geometry::sKaptonOnCarbonThickness + Geometry::sKaptonGlueThickness)};

    Double_t master[3];
    Double_t masterglue[3];
    chipPos->LocalToMaster(pos, master);
    chipPosGlue->LocalToMaster(posglue, masterglue);

    TGeoBBox* shape = (TGeoBBox*)mLadderVolume->GetShape();
    master[0] -= shape->GetDX();
    master[1] -= shape->GetDY();
    master[2] -= shape->GetDZ();

    masterglue[0] -= shape->GetDX();
    masterglue[1] -= shape->GetDY();
    masterglue[2] -= shape->GetDZ();

    LOG(DEBUG1) << "CreateSensors " << Form("adding chip %s_%d ", namePrefixS.Data(), ichip) << FairLogger::endl;
    // chipPos->Print();

    TGeoTranslation* trans = new TGeoTranslation(master[0], master[1], master[2]);
    TGeoHMatrix* final = new TGeoHMatrix((*trans) * (Geometry::sTransMFT2ITS).Inverse());
    mLadderVolume->AddNode(chipVol, ichip, final);
    mLadderVolume->AddNode(glue, ichip, new TGeoTranslation(masterglue[0], masterglue[1], masterglue[2]));
  }
}
