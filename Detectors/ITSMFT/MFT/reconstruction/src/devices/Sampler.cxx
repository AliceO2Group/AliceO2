// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "TClonesArray.h"
#include "TMessage.h"

#include <FairMQLogger.h>
#include <FairMQMessage.h>
#include <options/FairMQProgOptions.h>

#include "MFTReconstruction/devices/Sampler.h"

using namespace o2::MFT;
using namespace std;

//_____________________________________________________________________________
Sampler::Sampler()
  : FairMQDevice()
  , mOutputChannelName("data-out")
  , mAckChannelName("")
  , mRunAna(nullptr)
  , mSource(nullptr)
  , mInputObjects()
  , mNObjects(0)
  , mMaxIndex(-1)
  , mEventCounter(0)
  , mBranchNames()
  , mFileNames()

{

}

//_____________________________________________________________________________
Sampler::~Sampler()
= default;

//_____________________________________________________________________________
void Sampler::InitTask()
{

  LOG(INFO) << "Sampler::InitTask >>>>>" << "";
  
  mFileNames = GetConfig()->GetValue<std::vector<std::string>>("file-name");
  mMaxIndex = GetConfig()->GetValue<int64_t>("max-index");
  mBranchNames = GetConfig()->GetValue<std::vector<std::string>>("branch-name");
  mOutputChannelName = GetConfig()->GetValue<std::string>("out-channel");
  mAckChannelName = GetConfig()->GetValue<std::string>("ack-channel");

  mRunAna = new FairRunAna();
  
  if (mSource == nullptr) {
    mSource = new FairFileSource(mFileNames.at(0).c_str());
    for (unsigned int ifile = 1 ; ifile < mFileNames.size() ; ifile++) 
      ((FairFileSource*)mSource)->AddFile(mFileNames.at(ifile));
  }

  mSource->Init();

  LOG(INFO) << "Sampler::InitTask >>>>> going to request " << mBranchNames.size() << "  branches:";

  for (unsigned int ibrn = 0; ibrn < mBranchNames.size(); ibrn++ ) {

    LOG(INFO) << R"(Sampler::InitTask >>>>> requesting branch ")" << mBranchNames[ibrn] << R"(")";

    int branchStat = mSource->ActivateObject((TObject**)&mInputObjects[mNObjects],mBranchNames[ibrn].c_str()); // should check the status...

    if (mInputObjects[mNObjects]) {

      LOG(INFO) << "Sampler::InitTask >>>>> activated object " << mInputObjects[mNObjects] << " with name " << mBranchNames[ibrn] << " (" << branchStat << "), it got name: " << mInputObjects[mNObjects]->GetName() << "";

      if (strcmp(mInputObjects[mNObjects]->GetName(),mBranchNames[ibrn].c_str()))
        if (strcmp(mInputObjects[mNObjects]->ClassName(),"TClonesArray") == 0) 
          ((TClonesArray*)mInputObjects[mNObjects])->SetName(mBranchNames[ibrn].c_str());
      mNObjects++;

    }

  }

  LOG(INFO) << "Sampler::InitTask >>>>> nof objects = " << mNObjects << "";
  
  if ( mMaxIndex < 0 )
    mMaxIndex = mSource->CheckMaxEventNo();

  LOG(INFO) << "Sampler::InitTask >>>>> input source has " << mMaxIndex << " event(s).";

}

//_____________________________________________________________________________
bool Sampler::ConditionalRun()
{

  LOG(INFO) << "Sampler::ConditionalRun >>>>> run" << "";
  
  if ( mEventCounter == mMaxIndex ) return false;
  
  Int_t readEventReturn = mSource->ReadEvent(mEventCounter);

  if (readEventReturn != 0) return false;

  TMessage* message[1000];
  FairMQParts parts;
  
  for (int iobj = 0; iobj < mNObjects; iobj++) {

    LOG(INFO) << "Sampler::ConditionalRun >>>>> object " << iobj << " event " << mEventCounter << "";
    //fInputObjects[iobj]->Dump();
    message[iobj] = new TMessage(kMESS_OBJECT);
    message[iobj]->WriteObject(mInputObjects[iobj]);
    parts.AddPart(NewMessage(message[iobj]->Buffer(),
                             message[iobj]->BufferSize(),
                             [](void* /*data*/, void* hint) { delete (TMessage*)hint;},
                             message[iobj]));
    
  }
    
  Send(parts, mOutputChannelName);

  mEventCounter++;

  return true;

}

//_____________________________________________________________________________
void Sampler::PreRun()
{
  LOG(INFO) << "Sampler::PreRun >>>>> started!";

  mAckListener = new boost::thread(boost::bind(&Sampler::listenForAcks, this));

}

//_____________________________________________________________________________
void Sampler::PostRun() 
{
  
  if ( strcmp(mAckChannelName.data(),"") != 0 ) {
    try
      {
        mAckListener->join();
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
void Sampler::listenForAcks()
{

  if (strcmp(mAckChannelName.data(),"") != 0) {
    for (Long64_t eventNr = 0; eventNr < mMaxIndex ; ++eventNr) {
      unique_ptr<FairMQMessage> ack(NewMessage());
      if (Receive(ack,mAckChannelName.data())) {
        // do not need to do anything
      }

      if (!CheckCurrentState(RUNNING)) {
        break;
      }
    }

    LOG(INFO) << "Sampler::listenForAcks >>>>> Acknowledged " << mMaxIndex << " messages.";
  }

}
