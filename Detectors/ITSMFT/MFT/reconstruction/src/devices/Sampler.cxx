
#include "TClonesArray.h"
#include "TMessage.h"

#include "FairMQLogger.h"
#include "FairMQMessage.h"

#include "MFTReconstruction/devices/Sampler.h"

using namespace AliceO2::MFT;

//_____________________________________________________________________________
Sampler::Sampler()
  : FairMQDevice()
  , fOutputChannelName("data-out")
  , fAckChannelName("")
  , fRunAna(NULL)
  , fSource(NULL)
  , fInputObjects()
  , fNObjects(0)
  , fMaxIndex(-1)
  , fBranchNames()
  , fFileNames()

{

}

//_____________________________________________________________________________
Sampler::~Sampler()
{

}

//_____________________________________________________________________________
void Sampler::Run()
{

  LOG(INFO) << "SamplerRun::Run >>>>> run" << "";
  
  TMessage* message[1000];
  FairMQParts parts;
  
  int eventCounter = 0;

  while (CheckCurrentState(RUNNING)) {

    if (eventCounter == 100) break;

    Int_t readEventReturn = fSource->ReadEvent(eventCounter);

    if (readEventReturn != 0) break;

    for (int iobj = 0; iobj < fNObjects; iobj++) {

      LOG(INFO) << "Sampler::Run >>>>> object " << iobj << " event " << eventCounter << "";
      //fInputObjects[iobj]->Dump();
      message[iobj] = new TMessage(kMESS_OBJECT);
      message[iobj]->WriteObject(fInputObjects[iobj]);
      parts.AddPart(NewMessage(message[iobj]->Buffer(),message[iobj]->BufferSize(),[](void* /*data*/, void* hint) { delete (TMessage*)hint;},message[iobj]));

    }
    
    Send(parts, fOutputChannelName);

    eventCounter++;

  }

  LOG(INFO) << "Sampler::Run >>>>> going out of RUNNING state";

}

//_____________________________________________________________________________
void Sampler::InitTask()
{

  LOG(INFO) << "Sampler::InitTask >>>>>" << "";
  
  fRunAna = new FairRunAna();
  
  if (fFileNames.size() > 0) {

    fSource = new FairFileSource(fFileNames.at(0).c_str());
    for (unsigned int ifile = 1; ifile < fFileNames.size(); ifile++ ) 
      ((FairFileSource*)fSource)->AddFile(fFileNames.at(ifile));

  }

  fSource->Init();

  LOG(INFO) << "Sampler::InitTask >>>>> going to request " << fBranchNames.size() << "  branches:";

  for (unsigned int ibrn = 0; ibrn < fBranchNames.size(); ibrn++ ) {

    LOG(INFO) << "Sampler::InitTask >>>>> requesting branch \"" << fBranchNames[ibrn] << "\"";

    int branchStat = fSource->ActivateObject((TObject**)&fInputObjects[fNObjects],fBranchNames[ibrn].c_str()); // should check the status...

    if (fInputObjects[fNObjects]) {

      LOG(INFO) << "Sampler::InitTask >>>>> activated object " << fInputObjects[fNObjects] << " with name " << fBranchNames[ibrn] << " (" << branchStat << "), it got name: " << fInputObjects[fNObjects]->GetName() << "";

      if (strcmp(fInputObjects[fNObjects]->GetName(),fBranchNames[ibrn].c_str()))
        if (strcmp(fInputObjects[fNObjects]->ClassName(),"TClonesArray") == 0) 
          ((TClonesArray*)fInputObjects[fNObjects])->SetName(fBranchNames[ibrn].c_str());
      fNObjects++;

    }

  }

  LOG(INFO) << "Sampler::InitTask >>>>> nof objects = " << fNObjects << "";
  
  fMaxIndex = fSource->CheckMaxEventNo();

  LOG(INFO) << "Sampler::InitTask >>>>> input source has " << fMaxIndex << " events.";

}

