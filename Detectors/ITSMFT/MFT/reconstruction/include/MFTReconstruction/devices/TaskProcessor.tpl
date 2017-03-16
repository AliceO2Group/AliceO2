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
  , fInputChannelName("data-in")
  , fOutputChannelName("data-out")
  , fParamChannelName("param")
  , fEventHeader(NULL)
  , fInput(NULL)
  , fOutput(NULL)
  , fNewRunId(1)
  , fCurrentRunId(-1)
  , fDataToKeep("")
  , fReceivedMsgs(0)
  , fSentMsgs(0)
  , fFairTask(NULL)
{

}

//_____________________________________________________________________________
template <typename T>
TaskProcessor<T>::~TaskProcessor()
{

  if(fInput)
    {	  
      delete fInput;
      fInput=nullptr;
    }
  
  if(fOutput)
    {
      delete fOutput;
      fOutput=nullptr;
    }

  delete fFairTask;

}

//_____________________________________________________________________________
template <typename T>
void TaskProcessor<T>::Init()
{

  fFairTask = new T();
  
  fInputChannelName  = GetConfig()->template GetValue<std::string>("in-channel");
  fOutputChannelName = GetConfig()->template GetValue<std::string>("out-channel");

  fOutput = new TList();
  fInput = new TList();

  LOG(INFO) << "TaskProcessor::Init >>>>> execute OnData callback on channel " << fInputChannelName.c_str() << "";	

  OnData(fInputChannelName, &TaskProcessor<T>::ProcessData);

}

//_____________________________________________________________________________
template <typename T>
bool TaskProcessor<T>::ProcessData(FairMQParts& parts, int index)
{

  TObject* objectToKeep = NULL;
  
  LOG(INFO) << "TaskProcessor::ProcessData >>>>> message received with " << parts.Size() << " parts.";

  fReceivedMsgs++;

  TObject* tempObjects[10];
  for (int ipart = 0 ; ipart < parts.Size() ; ipart++) {

    TMessage2 tm(parts.At(ipart)->GetData(), parts.At(ipart)->GetSize());
    tempObjects[ipart] = (TObject*)tm.ReadObject(tm.GetClass());

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> got TObject with name \"" << tempObjects[ipart]->GetName() << "\".";

    if (strcmp(tempObjects[ipart]->GetName(),"EventHeader.") == 0 ) {

      fEventHeader = (EventHeader*)tempObjects[ipart];
      fNewRunId = fEventHeader->GetRunId();

      fInput->Add(tempObjects[ipart]);

      LOG(INFO) << "TaskProcessor::ProcessData >>>>> read event header with run = " << fNewRunId << "";

    } else {

      fInput->Add(tempObjects[ipart]);

    }

  }
  
  if (fEventHeader != NULL)	
    fNewRunId = fEventHeader->GetRunId();
  
  LOG(INFO)<<"TaskProcessor::ProcessData >>>>> got event header with run = " << fNewRunId;
  
  if(fNewRunId != fCurrentRunId) {

    fCurrentRunId=fNewRunId;
    fFairTask->InitMQ(nullptr);

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> Parameters updated, back to ProcessData(" << parts.Size() << " parts!)";

  }
    
  // Execute hit finder task
  fOutput->Clear();
  //LOG(INFO) << " The blocking line... analyzing event " << fEventHeader->GetMCEntryNumber();
  fFairTask->ExecMQ(fInput,fOutput);
  
  if (!fDataToKeep.empty()) {

    objectToKeep = fInput->FindObject(fDataToKeep.c_str());
    if (objectToKeep) fOutput->Add(objectToKeep);

  }
  
  TMessage* messageFEH;
  TMessage* messageTCA[10];
  FairMQParts partsOut;
  
  if (fEventHeader != NULL) {

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> create message from EventHeader"  << "";

    messageFEH = new TMessage(kMESS_OBJECT);
    messageFEH->WriteObject(fEventHeader);
    partsOut.AddPart(NewMessage(messageFEH->Buffer(),
                                messageFEH->BufferSize(),
                                [](void* data, void* hint) { delete (TMessage*)hint;},messageFEH));

  }

  for (int iobj = 0 ; iobj < fOutput->GetEntries() ; iobj++) {

    messageTCA[iobj] = new TMessage(kMESS_OBJECT);
    messageTCA[iobj]->WriteObject(fOutput->At(iobj));

    LOG(INFO) << "TaskProcessor::ProcessData >>>>> out object " << iobj << "";

    //fOutput->At(iobj)->Dump();
    partsOut.AddPart(NewMessage(messageTCA[iobj]->Buffer(),
                                messageTCA[iobj]->BufferSize(),
                                [](void* data, void* hint) { delete (TMessage*)hint;},messageTCA[iobj]));
  }

  Send(partsOut, fOutputChannelName);
  fSentMsgs++;

  fInput->Clear();

  return true;

}

//_____________________________________________________________________________
template <typename T>
void TaskProcessor<T>::PostRun()
{

  MQLOG(INFO) << "TaskProcessor::PostRun >>>>> Received " << fReceivedMsgs << " and sent " << fSentMsgs << " messages!";

}


