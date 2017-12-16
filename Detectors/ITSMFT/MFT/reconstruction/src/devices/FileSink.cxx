// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <memory> // for unique_ptr

#include "TMessage.h"
#include "TFile.h"
#include "TTree.h"
#include "TFolder.h"
#include "TClonesArray.h"

#include "FairLogger.h"
#include "FairEventHeader.h"
#include "options/FairMQProgOptions.h"

#include "ITSMFTReconstruction/Cluster.h"

#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/devices/FileSink.h"

using namespace o2::MFT;
using namespace std;

namespace o2 { namespace MFT {

// special class to expose protected TMessage constructor
//_____________________________________________________________________________
class SinkTMessage : public TMessage
{
  public:
  SinkTMessage(void* buf, Int_t len)
    : TMessage(buf, len)
  {
    LOG(INFO) << "FileSink::TMessage >>>>> create message of length " << len << "";
    ResetBit(kIsOwner);
  }
};

}
}

//_____________________________________________________________________________
FileSink::FileSink()
  : FairMQDevice()
  , mInputChannelName("data-in")
  , mAckChannelName("")
  , mFileName()
  , mTreeName()
 
  , mBranchNames()
  , mClassNames()
  , mFileOption()
  , mFlowMode(false)
  , mWrite(false)

  , mOutFile(nullptr)
  , mTree(nullptr)
  , mNObjects(0)
  , mOutputObjects(new TObject*[1000])
  , mFolder(nullptr)
{

}

//_____________________________________________________________________________
FileSink::~FileSink()
{ 

  if (mTree)
    {
      LOG(INFO) << "FileSink::~FileSink >>>>> write tree" << "";
      mTree->Write();
      delete mTree;
    }
  
  if (mOutFile)
    {
      if (mOutFile->IsOpen()) {
        LOG(INFO) << "FileSink::~FileSink >>>>> close output file" << "";       
        mOutFile->Close();
      }
      delete mOutFile;
    }

}

//_____________________________________________________________________________
void FileSink::Init()
{

  mFileName = GetConfig()->GetValue<std::string>("file-name");
  mClassNames = GetConfig()->GetValue<std::vector<std::string>>("class-name");
  mBranchNames = GetConfig()->GetValue<std::vector<std::string>>("branch-name");
  mInputChannelName = GetConfig()->GetValue<std::string>("in-channel");
  mAckChannelName = GetConfig()->GetValue<std::string>("ack-channel");

  LOG(INFO) << "FileSink::Init >>>>> SHOULD CREATE THE FILE AND TREE";
  
  mFileOption = "RECREATE";
  mTreeName = "o2sim";  
  
  mOutFile = TFile::Open(mFileName.c_str(),mFileOption.c_str());
  
  mTree = new TTree(mTreeName.c_str(), "/o2out");

  mFolder = new TFolder("cbmout", "Main Output Folder");
  TFolder* foldEventHeader = mFolder->AddFolder("EvtHeader","EvtHeader");
  TFolder* foldMFT         = mFolder->AddFolder("MFT","MFT");
  
  auto* BranchNameList = new TList();
  
  for ( mNObjects = 0 ; mNObjects < mBranchNames.size() ; mNObjects++ ) {

    LOG(INFO) << R"(FileSink::Init >>>>> Creating output branch ")" << mClassNames[mNObjects] << R"(" with name ")" << mBranchNames[mNObjects] << R"(")";

    if (mClassNames[mNObjects].find("TClonesArray(") == 0) {

      mClassNames[mNObjects] = mClassNames[mNObjects].substr(13,mClassNames[mNObjects].length()-12-2);

      LOG(INFO) << "FileSink::Init >>>>> Create a TClonesArray of this class: " << mClassNames[mNObjects].c_str() << "";

      mOutputObjects[mNObjects] = new TClonesArray(mClassNames[mNObjects].c_str());

      LOG(INFO) << "FileSink::Init >>>>> Create a branch " << mBranchNames[mNObjects].c_str() << "";

      mTree->Branch(mBranchNames[mNObjects].c_str(),"TClonesArray", &mOutputObjects[mNObjects]);
      foldMFT->Add(mOutputObjects[mNObjects]);
      BranchNameList->AddLast(new TObjString(mBranchNames[mNObjects].c_str()));

    } else if ( mClassNames[mNObjects].find("o2::MFT::EventHeader") == 0 ) {

      LOG(INFO) << "FileSink::Init >>>>> Create the branch EventHeader" << "";

      mOutputObjects            [mNObjects] = new EventHeader();
      mTree->Branch(mBranchNames[mNObjects].c_str(),"o2::MFT::EventHeader", &mOutputObjects[mNObjects]);
      foldEventHeader->Add(mOutputObjects[mNObjects]);
      BranchNameList->AddLast(new TObjString(mBranchNames[mNObjects].c_str()));

    } else {

      LOG(ERROR) << R"(!!! Unknown output object ")" << mClassNames[mNObjects] << R"(" !!!)";

    }
  }  

  mFolder->Write();
  BranchNameList->Write("BranchList", TObject::kSingleKey);
  BranchNameList->Delete();
  delete BranchNameList;

  OnData(mInputChannelName, &FileSink::storeData);

}

//_____________________________________________________________________________
bool FileSink::storeData(FairMQParts& parts, int index)
{

  TObject* tempObjects[10];

  LOG(INFO) << "FileSink::storeData >>>>> receive " << parts.Size() << " parts" << "";
      
  for (int ipart = 0; ipart < parts.Size(); ipart++) { 
    
    SinkTMessage tm(parts.At(ipart)->GetData(), parts.At(ipart)->GetSize());
    tempObjects[ipart] = (TObject*)tm.ReadObject(tm.GetClass());

    for (unsigned int ibr = 0; ibr < mBranchNames.size(); ibr++) { 
          
      LOG(INFO) << "FileSink::storeData >>>>> branch " << ibr << "   " << mBranchNames[ibr].c_str() << " part " << ipart << " " << tempObjects[ipart]->GetName() << "";

      // !!! force ???
      //if (kFALSE || (strcmp(tempObjects[ipart]->GetName(),fBranchNames[ibr].c_str()) == 0)) { 

      if ((strcmp(tempObjects[ipart]->GetName(),mBranchNames[ibr].c_str()) == 0) || (strncmp(mBranchNames[ibr].c_str(),"MFT",3) == 0 && strncmp(tempObjects[ipart]->GetName(),"AliceO2",7) == 0)) {

        mOutputObjects[ibr] = tempObjects[ipart];

        LOG(INFO) << "FileSink::storeData >>>>> branch selected for output " << ibr << "   " << mBranchNames[ibr].c_str() << " part " << ipart << " " << tempObjects[ipart]->GetName() << "";

        //fOutputObjects[ibr]->Dump();
        mTree->SetBranchAddress(mBranchNames[ibr].c_str(),&mOutputObjects[ibr]);
        
      }
    }
  }
  //fTree->Print();
  mTree->Fill();
      
  if (strcmp(mAckChannelName.data(),"") != 0) {
    LOG(INFO) << "FileSink::storeData >>>>> Send acknowldege" << "";
    unique_ptr<FairMQMessage> msg(NewMessage());
    Send(msg, mAckChannelName);
  }
      
  return true;

}

