/// \file HalfSegmentation.cxx
/// \brief Segmentation class for each half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TClonesArray.h"

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(HalfSegmentation);
/// \endcond

/// Default constructor

//_____________________________________________________________________________
HalfSegmentation::HalfSegmentation():
  VSegmentation(),
  mHalfDisks(NULL)
{ 


}

/// Copy constructor

//_____________________________________________________________________________
HalfSegmentation::HalfSegmentation(const HalfSegmentation& source):
VSegmentation(source),
mHalfDisks(NULL)
{
  
  if (source.mHalfDisks) mHalfDisks = new TClonesArray(*(source.mHalfDisks));

	
}

/// Constructor
/// \param nameGeomFile Char_t * : name of the XML geometry file.
/// By default it is : $ALICE_ROOT/ITSMFT/MFT/data/AliMFTGeometry.xml
/// \param id Short_t : ID Of the Half-MFT to build (0=Bottom; 1=Top)

//_____________________________________________________________________________
HalfSegmentation::HalfSegmentation(const Char_t *nameGeomFile, const Short_t id):
VSegmentation(),
mHalfDisks(NULL)
{

  Geometry * mftGeom = Geometry::Instance();
  
  UInt_t halfUniqueID = mftGeom->GetObjectID(Geometry::kHalfType, id);
  SetUniqueID(halfUniqueID);
  SetName(Form("%s_%d",GeometryTGeo::GetHalfDetName(),id));
    
  mHalfDisks = new TClonesArray("AliceO2::MFT::HalfDiskSegmentation", Constants::kNDisks);
  mHalfDisks -> SetOwner(kTRUE);

  // Create XML engine
  auto* geomFile = new TXMLEngine;
  
  // take access to main node
  XMLDocPointer_t  xmldoc   = geomFile->ParseFile(nameGeomFile);
  if (xmldoc==0) {
    delete geomFile;
    LOG(FATAL) << "Could not parse Geometry XML File named " << nameGeomFile << FairLogger::endl;
  }
  XMLNodePointer_t mainnode = geomFile->DocGetRootElement(xmldoc);
  
  // Find  Half-MFT node in the XML file
  XMLNodePointer_t halfnode ;
  FindHalf(geomFile, mainnode, halfnode);

  // Create Half Disks belonging to that Half-MFT
  CreateHalfDisks(geomFile, halfnode);
  
  // Release memory
  geomFile->FreeDoc(xmldoc);
  delete geomFile;

}

//_____________________________________________________________________________
HalfSegmentation::~HalfSegmentation() {

  if (mHalfDisks) mHalfDisks->Delete();
  delete mHalfDisks; 
  
}

///Clear the TClonesArray holding the HalfDiskSegmentation objects

//_____________________________________________________________________________
void HalfSegmentation::Clear(const Option_t* /*opt*/) {

  if (mHalfDisks) mHalfDisks->Delete();
  delete mHalfDisks; 
  mHalfDisks = NULL;
  
}

///Create the Half-Disks 

//_____________________________________________________________________________
void HalfSegmentation::CreateHalfDisks(TXMLEngine* xml, XMLNodePointer_t node)
{
  // this function display all accessible information about xml node and its children
  Int_t idisk;
  Int_t nladder;
  Double_t pos[3]={0., 0., 0.};
  Double_t ang[3]={0., 0., 0.};

  Geometry * mftGeom = Geometry::Instance();
    
  TString nodeName = xml->GetNodeName(node);
  if (!nodeName.CompareTo("disk")) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(node);
    while (attr != 0) {
      TString attrName = xml->GetAttrName(attr);
      TString attrVal  = xml->GetAttrValue(attr);
      if(!attrName.CompareTo("idisk")) {
        idisk = attrVal.Atoi();
        if (idisk >= Constants::kNDisks || idisk < 0) {
          LOG(FATAL) << "Wrong disk number : " << idisk << FairLogger::endl;
        }
      } else
	if(!attrName.CompareTo("nladder")) {
	  nladder = attrVal.Atoi();
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
    
    //AliDebug(1,Form("Creating Half-Disk %d with %d Ladders at the position (%.2f,%.2f,%.2f) with angles  (%.2f,%.2f,%.2f)",idisk,nladder,pos[0],pos[1],pos[2],ang[0],ang[1],ang[2]));
    
    UInt_t diskUniqueID = mftGeom->GetObjectID(Geometry::kHalfDiskType,mftGeom->GetHalfMFTID(GetUniqueID()),idisk );
    
    auto *halfDisk = new HalfDiskSegmentation(diskUniqueID);
    halfDisk->SetPosition(pos);
    halfDisk->SetRotationAngles(ang);
    halfDisk->SetNLadders(nladder);
    halfDisk->CreateLadders(xml, node);
    if(halfDisk->GetNLaddersBuild() != halfDisk->GetNLadders()) {
      LOG(FATAL) << "Number of ladder build " << halfDisk->GetNLaddersBuild() << " does not correspond to the number declared " << halfDisk->GetNLadders() << " Check XML file" << FairLogger::endl;
    }
    new ((*mHalfDisks)[idisk]) HalfDiskSegmentation(*halfDisk);
    delete halfDisk;
    //GetHalfDisk(idisk)->Print("ls");

  }

  // display all child nodes
  XMLNodePointer_t child = xml->GetChild(node);
  while (child!=0) {
    CreateHalfDisks(xml, child);
    child = xml->GetNext(child);
  }

}

/// Find Half-Disk in the XML file (private)

//_____________________________________________________________________________
void HalfSegmentation::FindHalf(TXMLEngine* xml, XMLNodePointer_t node, XMLNodePointer_t &retnode){
  // Find in the XML Geometry File the node corresponding to the Half-MFT being build
  // Set Position and Orientation of the Half-MFT
  Int_t isTop;
  Int_t ndisk;
  Double_t pos[3] = {0., 0., 0.};
  Double_t ang[3] = {0., 0., 0.};

  TString nodeName = xml->GetNodeName(node);
  if (!nodeName.CompareTo("half")) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(node);
    while (attr!=0) {
      TString attrName = xml->GetAttrName(attr);
      TString attrVal  = xml->GetAttrValue(attr);
      if(!attrName.CompareTo("top")){
        isTop = attrVal.Atoi();
        if (isTop>1 || isTop<0) {
          LOG(FATAL) << "Wrong Half MFT number : " << isTop << FairLogger::endl;
        }
      } else
        if(!attrName.CompareTo("ndisk")){
          ndisk = attrVal.Atoi();
          if (ndisk>5 || ndisk<0) {
            LOG(ERROR) << "Wrong number of disk : " << ndisk << FairLogger::endl;
          }
          
        } else
          if(!attrName.CompareTo("xpos")){
            pos[0] = attrVal.Atof();
          } else
            if(!attrName.CompareTo("ypos")){
              pos[1] = attrVal.Atof();
            } else
              if(!attrName.CompareTo("zpos")){
                pos[2] = attrVal.Atof();
              } else
                if(!attrName.CompareTo("phi")){
                  ang[0] = attrVal.Atof();
                } else
                  if(!attrName.CompareTo("theta")){
                    ang[1] = attrVal.Atof();
                  } else
                    if(!attrName.CompareTo("psi")){
                      ang[2] = attrVal.Atof();
                    } else{
		      LOG(ERROR) << "Unknwon Attribute name " << xml->GetAttrName(attr) << FairLogger::endl;
              }
      
      attr = xml->GetNextAttr(attr);
    }
    
    Geometry * mftGeom = Geometry::Instance();
    if(isTop == mftGeom->GetHalfMFTID(GetUniqueID())) {
      //AliDebug(1,Form("Setting up %s Half-MFT  %d Disk(s) at the position (%.2f,%.2f,%.2f) with angles (%.2f,%.2f,%.2f)",(isTop?"Top":"Bottom"),ndisk,pos[0],pos[1],pos[2],ang[0],ang[1],ang[2]));
      SetPosition(pos);
      SetRotationAngles(ang);
      retnode = node;
      return;
    }
    
  }
  
  // display all child nodes
  XMLNodePointer_t child = xml->GetChild(node);
  while (child!=0) {
    FindHalf(xml, child, retnode);
    child = xml->GetNext(child);
  }
}

