#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "TClonesArray.h"
#include "TMessage.h"

#include "FairMQLogger.h"
#include "FairMQMessage.h"
#include "FairMQProgOptions.h"

#include "MFTReconstruction/devices/Sampler.h"

using namespace AliceO2::MFT;
using namespace std;

//_____________________________________________________________________________
// helper function to clean up the object holding the data after it is transported.
void free_tmessage2(void* /*data*/, void *hint)
{
    delete (TMessage*)hint;
}

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
  , fEventCounter(0)
  , fBranchNames()
  , fFileNames()

{

}

//_____________________________________________________________________________
Sampler::~Sampler()
{

}

//_____________________________________________________________________________
void Sampler::InitTask()
{

  LOG(INFO) << "Sampler::InitTask >>>>>" << "";
  
  fFileNames = fConfig->GetValue<std::vector<std::string>>("file-name");
  fMaxIndex = fConfig->GetValue<int64_t>("max-index");
  fBranchNames = fConfig->GetValue<std::vector<std::string>>("branch-name");
  fOutputChannelName = fConfig->GetValue<std::string>("out-channel");
  fAckChannelName = fConfig->GetValue<std::string>("ack-channel");

  fRunAna = new FairRunAna();
  
  if (fSource == NULL) {
    fSource = new FairFileSource(fFileNames.at(0).c_str());
    for (unsigned int ifile = 1 ; ifile < fFileNames.size() ; ifile++) 
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
  
  if ( fMaxIndex < 0 )
    fMaxIndex = fSource->CheckMaxEventNo();

  LOG(INFO) << "Sampler::InitTask >>>>> input source has " << fMaxIndex << " event(s).";

}

//_____________________________________________________________________________
bool Sampler::ConditionalRun()
{

  LOG(INFO) << "Sampler::ConditionalRun >>>>> run" << "";
  
  if ( fEventCounter == fMaxIndex ) return false;
  
  Int_t readEventReturn = fSource->ReadEvent(fEventCounter);

  if (readEventReturn != 0) return false;

  TMessage* message[1000];
  FairMQParts parts;
  
  for (int iobj = 0; iobj < fNObjects; iobj++) {

    LOG(INFO) << "Sampler::ConditionalRun >>>>> object " << iobj << " event " << fEventCounter << "";
    //fInputObjects[iobj]->Dump();
    message[iobj] = new TMessage(kMESS_OBJECT);
    message[iobj]->WriteObject(fInputObjects[iobj]);
    parts.AddPart(NewMessage(message[iobj]->Buffer(),
                             message[iobj]->BufferSize(),
                             [](void* /*data*/, void* hint) { delete (TMessage*)hint;},
                             message[iobj]));
    
  }
    
  Send(parts, fOutputChannelName);

  fEventCounter++;

  return true;

}

//_____________________________________________________________________________
void Sampler::PreRun()
{
  LOG(INFO) << "Sampler::PreRun >>>>> started!";

  fAckListener = new boost::thread(boost::bind(&Sampler::ListenForAcks, this));

}

//_____________________________________________________________________________
void Sampler::PostRun() 
{
  
  if ( strcmp(fAckChannelName.data(),"") != 0 ) {
    try
      {
        fAckListener->join();
      }
    catch(boost::thread_resource_error& e)
      {
        LOG(ERROR) << e.what();
        exit(EXIT_FAILURE);
      }
  }
  
  LOG(INFO) << "Sampler::PostRun >>>>> finished!";

}

//_____________________________________________________________________________
void Sampler::ListenForAcks()
{

  if (strcmp(fAckChannelName.data(),"") != 0) {
    for (Long64_t eventNr = 0; eventNr < fMaxIndex ; ++eventNr) {
      unique_ptr<FairMQMessage> ack(NewMessage());
      if (Receive(ack,fAckChannelName.data())) {
	// do not need to do anything
      }

      if (!CheckCurrentState(RUNNING)) {
	break;
      }
    }

    LOG(INFO) << "Sampler::ListenForAcks >>>>> Acknowledged " << fMaxIndex << " messages.";
  }

}
