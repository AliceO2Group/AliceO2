// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HalfDiskSegmentation.cxx
/// \brief Class for the description of the structure of a half-disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TClonesArray.h"

#include "FairLogger.h"

#include "ITSMFTBase/SegmentationAlpide.h"
#include "MFTBase/Constants.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace o2::MFT;
using namespace o2::ITSMFT;

ClassImp(HalfDiskSegmentation);

/// Default constructor

//_____________________________________________________________________________
HalfDiskSegmentation::HalfDiskSegmentation():
  VSegmentation(),
  mNLadders(0),
  mLadders(nullptr)
{


}

/// Constructor
/// \param [in] uniqueID UInt_t: Unique ID of the Half-Disk to build

//_____________________________________________________________________________
HalfDiskSegmentation::HalfDiskSegmentation(UInt_t uniqueID):
  VSegmentation(),
  mNLadders(0),
  mLadders(nullptr)
{

  // constructor
  SetUniqueID(uniqueID);

  LOG(DEBUG1) << "Start creating half-disk UniqueID = " << GetUniqueID() << FairLogger::endl;
  
  Geometry * mftGeom = Geometry::instance();
  
  SetName(Form("%s_%d_%d",GeometryTGeo::getMFTDiskPattern(),mftGeom->getHalfID(GetUniqueID()), mftGeom->getDiskID(GetUniqueID()) ));
  
  mLadders  = new TClonesArray("o2::MFT::LadderSegmentation");
  mLadders -> SetOwner(kTRUE);
    
}

/// Copy Constructor

//_____________________________________________________________________________
HalfDiskSegmentation::HalfDiskSegmentation(const HalfDiskSegmentation& input):
  VSegmentation(input),
  mNLadders(input.mNLadders)
{
  
  // copy constructor
  if(input.mLadders) mLadders  = new TClonesArray(*(input.mLadders));
  else mLadders = new TClonesArray("o2::MFT::LadderSegmentation");
  mLadders -> SetOwner(kTRUE);

}


//_____________________________________________________________________________
HalfDiskSegmentation::~HalfDiskSegmentation() 
{

  Clear("");

}

/// Clear the TClonesArray holding the ladder segmentations

//_____________________________________________________________________________
void HalfDiskSegmentation::Clear(const Option_t* /*opt*/) 
{

  if (mLadders) mLadders->Delete();
  delete mLadders; 
  mLadders = nullptr;

}

/// Creates the Ladders on this half-Disk based on the information contained in the XML file

//_____________________________________________________________________________
void HalfDiskSegmentation::createLadders(TXMLEngine* xml, XMLNodePointer_t node)
{
  Int_t iladder;
  Int_t nsensor;
  Double_t pos[3];
  Double_t ang[3]={0.,0.,0.};

  Geometry * mftGeom = Geometry::instance();
    
  TString nodeName = xml->GetNodeName(node);
  if (!nodeName.CompareTo("ladder")) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(node);
    while (attr != nullptr) {
      TString attrName = xml->GetAttrName(attr);
      TString attrVal  = xml->GetAttrValue(attr);
      if(!attrName.CompareTo("iladder")) {
        iladder = attrVal.Atoi();
        if (iladder >= getNLadders() || iladder < 0) {
          LOG(FATAL) << "Wrong ladder number : " << iladder << FairLogger::endl;
        }
      } else
        if(!attrName.CompareTo("nsensor")) {
          nsensor = attrVal.Atoi();
        } else
          if(!attrName.CompareTo("xpos")) {
            pos[0] = attrVal.Atof();
          } else
            if(!attrName.CompareTo("ypos")) {
              pos[1] = attrVal.Atof();
            } else
              if(!attrName.CompareTo("zpos")) {
                pos[2] = attrVal.Atof();
              } else
                if(!attrName.CompareTo("phi")) {
                  ang[0] = attrVal.Atof();
                } else
                  if(!attrName.CompareTo("theta")) {
                    ang[1] = attrVal.Atof();
                  } else
                    if(!attrName.CompareTo("psi")) {
                      ang[2] = attrVal.Atof();
                    } else{
                      LOG(ERROR) << "Unknwon Attribute name " << xml->GetAttrName(attr) << FairLogger::endl;
                    }      
      attr = xml->GetNextAttr(attr);
    }
    
    Int_t plane = -1;
    Int_t ladderID=iladder;
    if( iladder < getNLadders()/2) {
      plane = 0;
    } else {
      plane = 1;
      //ladderID -= getNLadders()/2;
    }
    
    UInt_t ladderUniqueID = mftGeom->getObjectID(Geometry::LadderType,mftGeom->getHalfID(GetUniqueID()),mftGeom->getDiskID(GetUniqueID()),plane,ladderID);

    auto * ladder = new LadderSegmentation(ladderUniqueID);
    ladder->setNSensors(nsensor);
    ladder->setPosition(pos);
    ladder->setRotationAngles(ang);

    /// @todo : In the XML geometry file, the position of the top-left corner of the chip closest to the pipe is given in the Halfdisk coordinate system.
    /// Need to put in the XML file the position of the ladder coordinate center
    // Find the position of the corner of the flex which is the ladder corrdinate system center.
    
    pos[0] = -Geometry::sSensorSideOffset;
    pos[1] = -Geometry::sSensorTopOffset - SegmentationAlpide::SensorSizeRows;
    pos[2] = -Geometry::sFlexThickness - Geometry::sChipThickness;
    Double_t master[3];
    ladder->getTransformation()->LocalToMaster(pos, master);
    ladder->setPosition(master);
    
    ladder->createSensors();

    new ((*mLadders)[iladder]) LadderSegmentation(*ladder);
    delete ladder;

    //getLadder(iladder)->Print();
    
  }
  
  // display all child nodes
  XMLNodePointer_t child = xml->GetChild(node);
  while (child!=nullptr) {
    createLadders(xml, child);
    child = xml->GetNext(child);
  }

}

/// Returns the number of sensors on the Half-Disk

//_____________________________________________________________________________
Int_t HalfDiskSegmentation::getNChips() {

  Int_t nChips = 0;

  for (Int_t iLadder=0; iLadder<mLadders->GetEntries(); iLadder++) {

    LadderSegmentation *ladder = (LadderSegmentation*) mLadders->At(iLadder);
    nChips += ladder->getNSensors();

  }

  return nChips;

}

/// Print out Half-Disk information
/// \param [in] opt "l" or "ladder" -> The ladder information will be printed out as well

//_____________________________________________________________________________
void HalfDiskSegmentation::print(Option_t* opt){

  getTransformation()->Print();
  if(opt && (strstr(opt,"ladder")||strstr(opt,"l"))){
    for (int i=0; i<getNLadders(); i++)  getLadder(i)->Print(opt);    
  }

}
