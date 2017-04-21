/// \file ChipSegmentation.cxx
/// \brief Description of the virtual segmentation of the chips
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace o2::MFT;

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

  SetName(Form("%s_%d_%d_%d_%d",GeometryTGeo::getSensorName(),
               mftGeom->getHalfMFTID(GetUniqueID()),
               mftGeom->getHalfDiskID(GetUniqueID()),
               mftGeom->getLadderID(GetUniqueID()),
               mftGeom->getSensorID(GetUniqueID()) ));

  Double_t pos[3];
  pos[0] = mftGeom->getSensorID(GetUniqueID())*(Geometry::sSensorLength + Geometry::sSensorInterspace) + Geometry::sSensorSideOffset;
  pos[1] = Geometry::sSensorTopOffset;
  pos[2] = Geometry::sFlexThickness;
  setPosition(pos);
  
}

/// Returns the pixel ID corresponding to a hit at (x,y) in the Sensor  frame
///
/// \param [in] xHit Double_t : x Position of the Hit
/// \param [in] yHit Double_t : y Position of the Hit
///
/// \param [out] xPixel Int_t : x position of the pixel hit on the sensor matrix
/// \param [out] yPixel Int_t : y position of the pixel hit on the sensor matrix
/// \retval <kTRUE> if hit into the active part of the sensor
/// \retval <kFALSE> if hit outside the active part
//

//_____________________________________________________________________________
Bool_t ChipSegmentation::hitToPixelID(Double_t xHit, Double_t yHit, Int_t &xPixel, Int_t &yPixel) {
  // TODO Need to work on the Misalignment
  
  Double_t xHitLocal = xHit-Geometry::sSensorMargin;
  Double_t yHitLocal = yHit-(Geometry::sSensorMargin + Geometry::sSensorHeight - Geometry::sSensorActiveHeight);

  if (xHitLocal<0. || xHitLocal>Geometry::sSensorActiveWidth) return kFALSE;
  if (yHitLocal<0. || yHitLocal>Geometry::sSensorActiveHeight) return kFALSE;

  xPixel = Int_t( xHitLocal / Geometry::sXPixelPitch );
  yPixel = Int_t( yHitLocal / Geometry::sYPixelPitch );

  return kTRUE;

}

/// \brief Print out Sensor information (Name, ID, position, orientation)

//_____________________________________________________________________________
void ChipSegmentation::print(Option_t* /*option*/){
  
  getTransformation()->Print();

}
