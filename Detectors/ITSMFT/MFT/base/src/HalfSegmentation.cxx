// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

using namespace o2::MFT;

ClassImp(HalfSegmentation);

/// Default constructor

//_____________________________________________________________________________
HalfSegmentation::HalfSegmentation():
  VSegmentation(),
  mHalfDisks(nullptr)
{ 


}

/// Copy constructor

//_____________________________________________________________________________
HalfSegmentation::HalfSegmentation(const HalfSegmentation& source):
VSegmentation(source),
mHalfDisks(nullptr)
{
  
  if (source.mHalfDisks) mHalfDisks = new TClonesArray(*(source.mHalfDisks));

        
}

/// Constructor
/// \param nameGeomFile Char_t * : name of the XML geometry file.
/// By default it is : $(VMCWORKDIR)/Detectors/Geometry/MFT/data/Geometry.xml
/// \param id Short_t : ID Of the Half-MFT to build (0=Bottom; 1=Top)

//_____________________________________________________________________________
HalfSegmentation::HalfSegmentation(const Char_t *nameGeomFile, const Short_t id):
VSegmentation(),
mHalfDisks(nullptr)
{

  Geometry * mftGeom = Geometry::instance();
  
  UInt_t halfUniqueID = mftGeom->getObjectID(Geometry::HalfType, id);
  SetUniqueID(halfUniqueID);
  SetName(Form("%s_%d",GeometryTGeo::getMFTHalfPattern(),id));
    
  mHalfDisks = new TClonesArray("o2::MFT::HalfDiskSegmentation", Constants::sNDisks);
  mHalfDisks -> SetOwner(kTRUE);

  // Create XML engine
  auto* geomFile = new TXMLEngine;
  
  // take access to main node
  XMLDocPointer_t  xmldoc   = geomFile->ParseFile(nameGeomFile);
  if (xmldoc==nullptr) {
    delete geomFile;
    LOG(FATAL) << "Could not parse Geometry XML File named " << nameGeomFile << FairLogger::endl;
  }
  XMLNodePointer_t mainnode = geomFile->DocGetRootElement(xmldoc);
  
  // Find  Half-MFT node in the XML file
  XMLNodePointer_t halfnode ;
  findHalf(geomFile, mainnode, halfnode);

  // Create Half Disks belonging to that Half-MFT
  createHalfDisks(geomFile, halfnode);
  
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
  mHalfDisks = nullptr;
  
}

///Create the Half-Disks 

//_____________________________________________________________________________
void HalfSegmentation::createHalfDisks(TXMLEngine* xml, XMLNodePointer_t node)
{
  // this function display all accessible information about xml node and its children
  Int_t idisk;
  Int_t nladder;
  Double_t pos[3]={0., 0., 0.};
  Double_t ang[3]={0., 0., 0.};

  Geometry * mftGeom = Geometry::instance();
    
  TString nodeName = xml->GetNodeName(node);
  if (!nodeName.CompareTo("disk")) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(node);
    while (attr != nullptr) {
      TString attrName = xml->GetAttrName(attr);
      TString attrVal  = xml->GetAttrValue(attr);
      if(!attrName.CompareTo("idisk")) {
        idisk = attrVal.Atoi();
        if (idisk >= Constants::sNDisks || idisk < 0) {
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
    
    UInt_t diskUniqueID = mftGeom->getObjectID(Geometry::HalfDiskType,mftGeom->getHalfID(GetUniqueID()),idisk);
    
    auto *halfDisk = new HalfDiskSegmentation(diskUniqueID);
    halfDisk->setPosition(pos);
    halfDisk->setRotationAngles(ang);
    halfDisk->setNLadders(nladder);
    halfDisk->createLadders(xml, node);
    if(halfDisk->getNLaddersBuild() != halfDisk->getNLadders()) {
      LOG(FATAL) << "Number of ladder build " << halfDisk->getNLaddersBuild() << " does not correspond to the number declared " << halfDisk->getNLadders() << " Check XML file" << FairLogger::endl;
    }
    new ((*mHalfDisks)[idisk]) HalfDiskSegmentation(*halfDisk);
    delete halfDisk;

  }

  // display all child nodes
  XMLNodePointer_t child = xml->GetChild(node);
  while (child!=nullptr) {
    createHalfDisks(xml, child);
    child = xml->GetNext(child);
  }

}

/// Find Half-Disk in the XML file (private)

//_____________________________________________________________________________
void HalfSegmentation::findHalf(TXMLEngine* xml, XMLNodePointer_t node, XMLNodePointer_t &retnode){
  // Find in the XML Geometry File the node corresponding to the Half-MFT being build
  // Set Position and Orientation of the Half-MFT
  Int_t isTop;
  Int_t ndisk;
  Double_t pos[3] = {0., 0., 0.};
  Double_t ang[3] = {0., 0., 0.};

  TString nodeName = xml->GetNodeName(node);
  if (!nodeName.CompareTo("half")) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(node);
    while (attr!=nullptr) {
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
    
    Geometry * mftGeom = Geometry::instance();
    if(isTop == mftGeom->getHalfID(GetUniqueID())) {
      setPosition(pos);
      setRotationAngles(ang);
      retnode = node;
      return;
    }
    
  }
  
  // display all child nodes
  XMLNodePointer_t child = xml->GetChild(node);
  while (child!=nullptr) {
    findHalf(xml, child, retnode);
    child = xml->GetNext(child);
  }
}

