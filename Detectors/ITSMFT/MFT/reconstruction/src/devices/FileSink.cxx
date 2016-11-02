#include <memory> // for unique_ptr

#include "TMessage.h"
#include "TFile.h"
#include "TTree.h"
#include "TFolder.h"
#include "TClonesArray.h"

#include "FairEventHeader.h"

#include "MFTReconstruction/devices/FileSink.h"
#include "MFTReconstruction/Hit.h"

using namespace AliceO2::MFT;

// special class to expose protected TMessage constructor
//_____________________________________________________________________________
class TMessage2 : public TMessage
{
  public:
  TMessage2(void* buf, Int_t len)
    : TMessage(buf, len)
  {
    LOG(INFO) << "FileSink::TMessage2 >>>>> create message of length " << len << "";
    ResetBit(kIsOwner);
  }
};

//_____________________________________________________________________________
FileSink::FileSink()
  : FairMQDevice()
  , fInputChannelName("data-in")
  , fAckChannelName("")
  , fFileName()
  , fTreeName()
 
  , fBranchNames()
  , fClassNames()
  , fFileOption()
  , fFlowMode(false)
  , fWrite(false)

  , fOutFile(NULL)
  , fTree(NULL)
  , fNObjects(0)
  , fOutputObjects(new TObject*[1000])
  , fFolder(NULL)
{

}

//_____________________________________________________________________________
FileSink::~FileSink()
{ 

  if (fTree)
    {
      LOG(INFO) << "FileSink::~FileSink >>>>> write tree" << "";
      fTree->Write();
      delete fTree;
    }
  
  if (fOutFile)
    {
      if (fOutFile->IsOpen()) {
	LOG(INFO) << "FileSink::~FileSink >>>>> close output file" << "";	
	fOutFile->Close();
      }
      delete fOutFile;
    }

}

//_____________________________________________________________________________
void FileSink::Init()
{

  LOG(INFO) << "SHOULD CREATE THE FILE AND TREE";
  
  fFileOption = "RECREATE";
  fTreeName = "o2sim";  
  
  fOutFile = TFile::Open(fFileName.c_str(),fFileOption.c_str());
  
  fTree = new TTree(fTreeName.c_str(), "/o2out");

  fFolder = new TFolder("cbmout", "Main Output Folder");
  TFolder* foldEventHeader = fFolder->AddFolder("EvtHeader","EvtHeader");
  TFolder* foldMFT         = fFolder->AddFolder("MFT","MFT");
  
  TList* BranchNameList = new TList();
  
  for ( fNObjects = 0 ; fNObjects < fBranchNames.size() ; fNObjects++ ) {

    LOG(INFO) << "FileSink::Init >>>>> Creating output branch \"" << fClassNames[fNObjects] << "\" with name \"" << fBranchNames[fNObjects] << "\"";

    if (fClassNames[fNObjects].find("TClonesArray(") == 0) {

      fClassNames[fNObjects] = fClassNames[fNObjects].substr(13,fClassNames[fNObjects].length()-12-2);
      LOG(INFO) << "FileSink::Init >>>>> Create a TClonesArray of this class: " << fClassNames[fNObjects].c_str() << "";
      fOutputObjects[fNObjects] = new TClonesArray(fClassNames[fNObjects].c_str());
      fOutputObjects[fNObjects]->Print();
      LOG(INFO) << "FileSink::Init >>>>> Create a branch " << fBranchNames[fNObjects].c_str() << "";
      fTree->Branch(fBranchNames[fNObjects].c_str(),"TClonesArray", &fOutputObjects[fNObjects]);
      foldMFT->Add(fOutputObjects[fNObjects]);
      BranchNameList->AddLast(new TObjString(fBranchNames[fNObjects].c_str()));

    } else if ( fClassNames[fNObjects].find("FairEventHeader") == 0 ) {

      LOG(INFO) << "FileSink::Init >>>>> Create the branch FairEventHeader" << "";
      fOutputObjects            [fNObjects] = new    FairEventHeader();
      fTree->Branch(fBranchNames[fNObjects].c_str(),"FairEventHeader", &fOutputObjects[fNObjects]);
      foldEventHeader->Add(fOutputObjects[fNObjects]);
      BranchNameList->AddLast(new TObjString(fBranchNames[fNObjects].c_str()));

    } else {

      LOG(ERROR) << "!!! Unknown output object \"" << fClassNames[fNObjects] << "\" !!!";

    }
  }  

  fFolder->Write();
  BranchNameList->Write("BranchList", TObject::kSingleKey);
  BranchNameList->Delete();
  delete BranchNameList;

}

//_____________________________________________________________________________
void FileSink::Run()
{

  FairMQParts parts;
  TObject* tempObjects[10];

  LOG(INFO) << "FileSink::Run >>>>>" << "";

  while (CheckCurrentState(RUNNING)) {
      
    LOG(INFO) << "FileSink::Run >>>>> RUNNING" << "";

    if (Receive(parts, fInputChannelName) >= 0) {

      LOG(INFO) << "FileSink::Run >>>>> receive " << parts.Size() << " parts" << "";
      
      for (int ipart = 0; ipart < parts.Size(); ipart++) { 
	
	TMessage2 tm(parts.At(ipart)->GetData(), parts.At(ipart)->GetSize());
	tempObjects[ipart] = (TObject*)tm.ReadObject(tm.GetClass());
	for (unsigned int ibr = 0; ibr < fBranchNames.size(); ibr++) { 
	  
          LOG(INFO) << "FileSink::Run >>>>> branch " << ibr << "   " << fBranchNames[ibr].c_str() << "";
          LOG(INFO) << "FileSink::Run >>>>> branch " << fBranchNames[ibr].c_str() << " " << tempObjects[ipart]->GetName() << "";
	  // !!! force !!!
	  if (kTRUE || (strcmp(tempObjects[ipart]->GetName(),fBranchNames[ibr].c_str()) == 0)) { 
	    fOutputObjects[ibr] = tempObjects[ipart];
	    LOG(INFO) << "FileSink""Run >>>>> out object branch " << ibr << " part " << ipart << "";	    
	    //fOutputObjects[ibr]->Dump();
	    fTree->SetBranchAddress(fBranchNames[ibr].c_str(),&fOutputObjects[ibr]);
	  }
	}
      }
      //fTree->Print();
      fTree->Fill();
      
      if (strcmp(fAckChannelName.data(),"") != 0) {
	std::unique_ptr<FairMQMessage> msg(NewMessage());
	Send(msg, fAckChannelName);
      }
      
    }
  }

}

//_____________________________________________________________________________
void FileSink::SetProperty(const int key, const std::string& value)
{
    switch (key)
    {
        case OutputFileName :
          SetOutputFileName(value);
            break;

        default:
            FairMQDevice::SetProperty(key, value);
            break;
    }
}

