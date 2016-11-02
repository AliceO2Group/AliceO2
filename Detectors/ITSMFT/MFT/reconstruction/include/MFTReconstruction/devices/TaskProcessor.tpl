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
  , fEventHeader(NULL)
  , fInput(NULL)
  , fOutput(NULL)
  , fNewRunId(1)
  , fCurrentRunId(-1)
  , fDataToKeep("")
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

  //fHitFinder->InitMQ(fRootParFileName,fAsciiParFileName);
  fFairTask = new T();
  
  fOutput = new TList();
  fInput = new TList();

}

//_____________________________________________________________________________
template <typename T>
void TaskProcessor<T>::Run()
{

  LOG(INFO) << "TaskProcessor::Run >>>>>" << "";

  int receivedMsgs = 0;  
  int sentMsgs = 0;
  TObject* objectToKeep = NULL;
  FairMQParts parts;
      
  while (CheckCurrentState(RUNNING)) {

    LOG(INFO) << "TaskProcessor::Run >>>>> RUNNING" << "";

    if (Receive(parts,fInputChannelName) >= 0) {
        
      LOG(INFO) << "TaskProcessor::Run >>>>> message received with " << parts.Size() << " parts.";
      receivedMsgs++;
      TObject* tempObjects[10];
      for (int ipart = 0 ; ipart < parts.Size(); ipart++) { 
            
        TMessage2 tm(parts.At(ipart)->GetData(), parts.At(ipart)->GetSize());
        tempObjects[ipart] = (TObject*)tm.ReadObject(tm.GetClass());
        LOG(INFO) << "TaskProcessor::Run >>>>> got TObject with name \"" << tempObjects[ipart]->GetName() << "\".";
        if (strcmp(tempObjects[ipart]->GetName(),"EventHeader.") == 0) {
	  fEventHeader = (FairEventHeader*)tempObjects[ipart]; 
          fNewRunId = fEventHeader->GetRunId();
	  LOG(INFO)<<"TaskProcessor::Run >>>>> got event header with run = " << fNewRunId;
        } else {
          fInput->Add(tempObjects[ipart]);
        }

      }

      if(fNewRunId != fCurrentRunId) {            
        fCurrentRunId = fNewRunId;
        fFairTask->InitMQ(nullptr);
      }
            
      fOutput->Clear();
      //LOG(INFO) << " The blocking line... analyzing event " << fEventHeader->GetMCEntryNumber();
      fFairTask->ExecMQ(fInput,fOutput);

      if (!fDataToKeep.empty()) {
        objectToKeep = fInput->FindObject(fDataToKeep.c_str());
        if (objectToKeep) fOutput->Add(objectToKeep);
      }

      TMessage* messageFEH;     // FileEventHeader
      TMessage* messageTCA[10]; // TClonesArray
      FairMQParts partsOut;
      
      // no event header!
      //messageFEH = new TMessage(kMESS_OBJECT);
      //messageFEH->WriteObject(fEventHeader);
      //partsOut.AddPart(NewMessage(messageFEH->Buffer(),messageFEH->BufferSize(),[](void* /*data*/, void* hint) { delete (TMessage*)hint;},messageFEH));
      
      for (int iobj = 0; iobj < fOutput->GetEntries(); iobj++) {
        messageTCA[iobj] = new TMessage(kMESS_OBJECT);
        messageTCA[iobj]->WriteObject(fOutput->At(iobj));
        LOG(INFO) << "TaskProcessor::Run >>>>> out object " << iobj << "";
        //fOutput->At(iobj)->Dump();
        partsOut.AddPart(NewMessage(messageTCA[iobj]->Buffer(),messageTCA[iobj]->BufferSize(),[](void* /*data*/, void* hint) { delete (TMessage*)hint;},messageTCA[iobj]));
       }

       Send(partsOut, fOutputChannelName);
       sentMsgs++;

    }  
   
    fInput->Clear();

  }

  MQLOG(INFO) << "Received " << receivedMsgs << " and sent " << sentMsgs << " messages!";

}


