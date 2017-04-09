/// \file HalfDiskSegmentation.cxx
/// \brief Class for the description of the structure of a half-disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TClonesArray.h"

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace o2::MFT;

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
  
  Geometry * mftGeom = Geometry::Instance();
  
  SetName(Form("%s_%d_%d",GeometryTGeo::GetHalfDiskName(),mftGeom->GetHalfMFTID(GetUniqueID()), mftGeom->GetHalfDiskID(GetUniqueID()) ));
  
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
void HalfDiskSegmentation::CreateLadders(TXMLEngine* xml, XMLNodePointer_t node)
{
  Int_t iladder;
  Int_t nsensor;
  Double_t pos[3];
  Double_t ang[3]={0.,0.,0.};

  Geometry * mftGeom = Geometry::Instance();
    
  TString nodeName = xml->GetNodeName(node);
  if (!nodeName.CompareTo("ladder")) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(node);
    while (attr != nullptr) {
      TString attrName = xml->GetAttrName(attr);
      TString attrVal  = xml->GetAttrValue(attr);
      if(!attrName.CompareTo("iladder")) {
        iladder = attrVal.Atoi();
        if (iladder >= GetNLadders() || iladder < 0) {
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
    if( iladder < GetNLadders()/2) {
      plane = 0;
    } else {
      plane = 1;
      //ladderID -= GetNLadders()/2;
    }
    
    //if ((plane==0 && pos[2]<0.) || (plane==1 && pos[2]>0.))
    //AliFatal(Form(" Wrong Z Position or ladder number ???  :  z= %f ladder id = %d",pos[2],ladderID));

    UInt_t ladderUniqueID = mftGeom->GetObjectID(Geometry::LadderType,mftGeom->GetHalfMFTID(GetUniqueID()),mftGeom->GetHalfDiskID(GetUniqueID()),ladderID);

    //UInt_t ladderUniqueID = (Geometry::LadderType<<13) +  (((GetUniqueID()>>9) & 0xF)<<9) + (plane<<8) + (ladderID<<3);
    
    auto * ladder = new LadderSegmentation(ladderUniqueID);
    ladder->SetNSensors(nsensor);
    ladder->SetPosition(pos);
    ladder->SetRotationAngles(ang);

    /// @todo : In the XML geometry file, the position of the top-left corner of the chip closest to the pipe is given in the Halfdisk coordinate system.
    /// Need to put in the XML file the position of the ladder coordinate center
    // Find the position of the corner of the flex which is the ladder corrdinate system center.
    
    pos[0] = -Geometry::sSensorSideOffset;
    pos[1] = -Geometry::sSensorTopOffset - Geometry::sSensorHeight;
    pos[2] = -Geometry::sFlexThickness - Geometry::sSensorThickness;
    Double_t master[3];
    ladder->GetTransformation()->LocalToMaster(pos, master);
    ladder->SetPosition(master);
    //AliDebug(2,Form("Creating Ladder %2d with %d Sensors at the position (%.2f,%.2f,%.2f) with angles (%.2f,%.2f,%.2f) and ID = %d",iladder,nsensor,master[0],master[1],master[2],ang[0],ang[1],ang[2], ladderUniqueID ) );
    
    ladder->CreateSensors();

    new ((*mLadders)[iladder]) LadderSegmentation(*ladder);
    delete ladder;

    //GetLadder(iladder)->Print();
    
  }
  
  // display all child nodes
  XMLNodePointer_t child = xml->GetChild(node);
  while (child!=nullptr) {
    CreateLadders(xml, child);
    child = xml->GetNext(child);
  }

}

/// Returns the number of sensors on the Half-Disk

//_____________________________________________________________________________
Int_t HalfDiskSegmentation::GetNChips() {

  Int_t nChips = 0;

  for (Int_t iLadder=0; iLadder<mLadders->GetEntries(); iLadder++) {

    LadderSegmentation *ladder = (LadderSegmentation*) mLadders->At(iLadder);
    nChips += ladder -> GetNSensors();

  }

  return nChips;

}

/// Print out Half-Disk information
/// \param [in] opt "l" or "ladder" -> The ladder information will be printed out as well

//_____________________________________________________________________________
void HalfDiskSegmentation::Print(Option_t* opt){

  //AliInfo(Form("Half-Disk %s (Unique ID = %d)",GetName(),GetUniqueID()));
  GetTransformation()->Print();
  //AliInfo(Form("N Ladders = %d",mNLadders));
  if(opt && (strstr(opt,"ladder")||strstr(opt,"l"))){
    for (int i=0; i<GetNLadders(); i++)  GetLadder(i)->Print(opt);
    
  }
}
