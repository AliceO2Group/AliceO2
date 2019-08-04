// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file LadderSegmentation.cxx
/// \brief Description of the virtual segmentation of a ladder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace o2::mft;

ClassImp(LadderSegmentation);

/// Default constructor

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation() : VSegmentation(), mChips(nullptr) {}

/// Constructor
/// \param [in] uniqueID UInt_t: Unique ID of the Ladder to build

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation(UInt_t uniqueID) : VSegmentation(), mChips(nullptr)
{

  SetUniqueID(uniqueID);

  Geometry* mftGeom = Geometry::instance();

  SetName(Form("%s_%d_%d_%d", GeometryTGeo::getMFTLadderPattern(), mftGeom->getHalfID(GetUniqueID()),
               mftGeom->getDiskID(GetUniqueID()), mftGeom->getLadderID(GetUniqueID())));

  // constructor
}

/// Copy Constructor

//_____________________________________________________________________________
LadderSegmentation::LadderSegmentation(const LadderSegmentation& ladder)
  : VSegmentation(ladder), mNSensors(ladder.mNSensors)
{
  // copy constructor

  if (ladder.mChips)
    mChips = new TClonesArray(*(ladder.mChips));
  else
    mChips = new TClonesArray("o2::mft::ChipSegmentation", mNSensors);

  mChips->SetOwner(kTRUE);
}

/// Creates the Sensors Segmentation array on the Ladder

//_____________________________________________________________________________
void LadderSegmentation::createSensors(TXMLEngine* xml, XMLNodePointer_t node)
{

  if (!mChips) {
    mChips = new TClonesArray("o2::mft::ChipSegmentation", mNSensors);
    mChips->SetOwner(kTRUE);
  }

  Int_t ichip;
  Double_t pos[3];
  Double_t ang[3] = {0., 0., 0.};

  Geometry* mftGeom = Geometry::instance();

  TString nodeName = xml->GetNodeName(node);
  if (!nodeName.CompareTo("chip")) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(node);
    while (attr != nullptr) {
      TString attrName = xml->GetAttrName(attr);
      TString attrVal = xml->GetAttrValue(attr);
      if (!attrName.CompareTo("ichip")) {
        ichip = attrVal.Atoi();
        if (ichip >= getNSensors() || ichip < 0) {
          LOG(FATAL) << "Wrong chip number : " << ichip << FairLogger::endl;
        }
      } else if (!attrName.CompareTo("xpos")) {
        pos[0] = attrVal.Atof();
      } else if (!attrName.CompareTo("ypos")) {
        pos[1] = attrVal.Atof();
      } else if (!attrName.CompareTo("zpos")) {
        pos[2] = attrVal.Atof();
      } else if (!attrName.CompareTo("phi")) {
        ang[0] = attrVal.Atof();
      } else if (!attrName.CompareTo("theta")) {
        ang[1] = attrVal.Atof();
      } else if (!attrName.CompareTo("psi")) {
        ang[2] = attrVal.Atof();
      } else {
        LOG(ERROR) << "Unknwon Attribute name " << xml->GetAttrName(attr) << FairLogger::endl;
      }
      attr = xml->GetNextAttr(attr);
    }

    UInt_t chipUniqueID =
      mftGeom->getObjectID(Geometry::SensorType,
                           mftGeom->getHalfID(GetUniqueID()),
                           mftGeom->getDiskID(GetUniqueID()),
                           mftGeom->getPlaneID(GetUniqueID()),
                           mftGeom->getLadderID(GetUniqueID()),
                           ichip);

    auto* chip = new ChipSegmentation(chipUniqueID);
    //    pos[0] = mftGeom->getSensorID(GetUniqueID())*
    //    (SegmentationAlpide::SensorSizeCols + Geometry::sSensorInterspace) + Geometry::sSensorSideOffset;
    //    pos[1] = Geometry::sSensorTopOffset;
    //    pos[2] = Geometry::sFlexThickness;
    chip->setPosition(pos);
    chip->setRotationAngles(ang);

    new ((*mChips)[ichip]) ChipSegmentation(*chip);
    delete chip;
  }

  // display all child nodes
  XMLNodePointer_t child = xml->GetChild(node);
  while (child != nullptr) {
    createSensors(xml, child);
    child = xml->GetNext(child);
  }
}

/// Returns pointer to a sensor segmentation
/// \param [in] sensorID Int_t: ID of the sensor on the ladder

//_____________________________________________________________________________
ChipSegmentation* LadderSegmentation::getSensor(Int_t sensorID) const
{

  if (sensorID < 0 || sensorID >= mNSensors)
    return nullptr;

  ChipSegmentation* chip = (ChipSegmentation*)mChips->At(sensorID);

  return chip;
}

/// Print out Ladder information (position, orientation, # of sensors)
/// \param [in] opt "s" or "sensor" -> The individual sensor information will be printed out as well

//_____________________________________________________________________________
void LadderSegmentation::print(Option_t* opt)
{

  getTransformation()->Print();
  if (opt && (strstr(opt, "sensor") || strstr(opt, "s"))) {
    for (int i = 0; i < getNSensors(); i++)
      getSensor(i)->Print("");
  }
}
