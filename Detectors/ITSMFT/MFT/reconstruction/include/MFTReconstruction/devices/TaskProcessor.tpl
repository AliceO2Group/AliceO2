// special class to expose protected TMessage constructor
//_____________________________________________________________________________
class TMessage2 : public TMessage
{
  public:
  TMessage2(void* buf, Int_t len)
    : TMessage(buf, len)
  {
    LOG(INFO) << "TaskProcessor::TMessage2 >>>>> create message of length " << len << "";
    ResetBit(kIsOwner);
  }
};

//_____________________________________________________________________________
template <typename T>
TaskProcessor<T>::TaskProcessor()
  : FairMQDevice()
  , mInputChannelName("data-in")
  , mOutputChannelName("data-out")
  , mParamChannelName("param")
  , mEventHeader(NULL)
  , mInput(NULL)
  , mOutput(NULL)
  , mNewRunId(1)
  , mCurrentRunId(-1)
  , mDataToKeep("")
  , mReceivedMsgs(0)
  , mSentMsgs(0)
  , mFairTask(NULL)
{

}

//_____________________________________________________________________________
template <typename T>
TaskProcessor<T>::~TaskProcessor()
{

  if(mInput)
    {	  
      delete mInput;
      mInput=nullptr;
    }
  
  if(mOutput)
    {
      delete mOutput;
      mOutput=nullptr;
    }

  delete mFairTask;

}

//_____________________________________________________________________________
template <typename T>
void TaskProcessor<T>::Init()
{

  mFairTask = new T();
  
  mInputChannelName  = GetConfig()->template GetValue<std::string>("in-channel");
  mOutputChannelName = GetConfig()->template GetValue<std::string>("out-channel");

  mOutput = new TList();
  mInput = new TList();

  LOG(INFO) << "TaskProcessor::Init >>>>> execute OnData callback on channel " << mInputChannelName.c_str() << "";	

  OnData(mInputChannelName, &TaskProcessor<T>::ProcessData);

}

//_____________________________________________________________________________
template <typename T>
bool TaskProcessor<T>::ProcessData(FairMQParts& parts, int index)
{

  TObject* objectToKeep = NULL;
  
  LOG(INFO) << "TaskProcessor::ProcessData >>>>> message received with " << parts.Size() << " parts.";

  mReceivedMsgs++;

  TObject* tempObjects[10];
  for (int ipart = 0 ; ipart < parts.Size() ; ipart++) {

    TMessage2 tm(parts.At(ipart)->GetData(), parts.At(ipart)->GetSize());
    tempObjects[ipart] = (TObject*)tm.ReadObject(tm.GetClass());

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> got TObject with name \"" << tempObjects[ipart]->GetName() << "\".";

    if (strcmp(tempObjects[ipart]->GetName(),"EventHeader.") == 0 ) {

      mEventHeader = (EventHeader*)tempObjects[ipart];
      mNewRunId = mEventHeader->GetRunId();

      mInput->Add(tempObjects[ipart]);

      LOG(INFO) << "TaskProcessor::ProcessData >>>>> read event header with run = " << mNewRunId << "";

    } else {

      mInput->Add(tempObjects[ipart]);

    }

  }
  
  if (mEventHeader != NULL)	
    mNewRunId = mEventHeader->GetRunId();
  
  LOG(INFO)<<"TaskProcessor::ProcessData >>>>> got event header with run = " << mNewRunId;
  
  if(mNewRunId != mCurrentRunId) {

    mCurrentRunId=mNewRunId;
    mFairTask->InitMQ(nullptr);

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> Parameters updated, back to ProcessData(" << parts.Size() << " parts!)";

  }
    
  // Execute hit finder task
  mOutput->Clear();
  //LOG(INFO) << " The blocking line... analyzing event " << fEventHeader->GetMCEntryNumber();
  mFairTask->ExecMQ(mInput,mOutput);
  
  if (!mDataToKeep.empty()) {

    objectToKeep = mInput->FindObject(mDataToKeep.c_str());
    if (objectToKeep) mOutput->Add(objectToKeep);

  }
  
  TMessage* messageFEH;
  TMessage* messageTCA[10];
  FairMQParts partsOut;
  
  if (mEventHeader != NULL) {

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> create message from EventHeader"  << "";

    messageFEH = new TMessage(kMESS_OBJECT);
    messageFEH->WriteObject(mEventHeader);
    partsOut.AddPart(NewMessage(messageFEH->Buffer(),
                                messageFEH->BufferSize(),
                                [](void* data, void* hint) { delete (TMessage*)hint;},messageFEH));

  }

  for (int iobj = 0 ; iobj < mOutput->GetEntries() ; iobj++) {

    messageTCA[iobj] = new TMessage(kMESS_OBJECT);
    messageTCA[iobj]->WriteObject(mOutput->At(iobj));

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> out object " << iobj << "";

    //fOutput->At(iobj)->Dump();
    partsOut.AddPart(NewMessage(messageTCA[iobj]->Buffer(),
                                messageTCA[iobj]->BufferSize(),
                                [](void* data, void* hint) { delete (TMessage*)hint;},messageTCA[iobj]));
  }

  Send(partsOut, mOutputChannelName);
  mSentMsgs++;

  mInput->Clear();

  return true;

}

//_____________________________________________________________________________
template <typename T>
void TaskProcessor<T>::PostRun()
{

  MQLOG(INFO) << "TaskProcessor::PostRun >>>>> Received " << mReceivedMsgs << " and sent " << mSentMsgs << " messages!";

}


