// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChipSegmentation.cxx
/// \brief Description of the virtual segmentation of the chips
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "ITSMFTBase/SegmentationAlpide.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace o2::MFT;
using namespace o2::ITSMFT;

ClassImp(ChipSegmentation);

/// Default constructor

//_____________________________________________________________________________
ChipSegmentation::ChipSegmentation():
  VSegmentation()
{
}

/// Constructor
//_____________________________________________________________________________
ChipSegmentation::ChipSegmentation(UInt_t uniqueID):
  VSegmentation()
{
  // constructor
  Geometry * mftGeom = Geometry::instance();

  SetUniqueID(uniqueID);

  SetName(Form("%s_%d_%d_%d_%d",GeometryTGeo::getMFTChipPattern(),
               mftGeom->getHalfID(GetUniqueID()),
               mftGeom->getDiskID(GetUniqueID()),
               mftGeom->getLadderID(GetUniqueID()),
               mftGeom->getSensorID(GetUniqueID()) ));

  Double_t pos[3];
  pos[0] = mftGeom->getSensorID(GetUniqueID())*
    (SegmentationAlpide::SensorSizeCols + Geometry::sSensorInterspace) + Geometry::sSensorSideOffset;
  pos[1] = Geometry::sSensorTopOffset;
  pos[2] = Geometry::sFlexThickness;
  setPosition(pos);
  
}

/// \brief Print out Sensor information (Name, ID, position, orientation)
//_____________________________________________________________________________
void ChipSegmentation::print(Option_t* /*option*/){
  
  getTransformation()->Print();

}
