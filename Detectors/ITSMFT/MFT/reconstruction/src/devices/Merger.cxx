#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"

#include "TMessage.h"

#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/devices/Merger.h"

using namespace AliceO2::MFT;

using namespace std;

// special class to expose protected TMessage constructor
//_____________________________________________________________________________
class MergerTMessage : public TMessage
{
  public:
  MergerTMessage(void* buf, Int_t len)
    : TMessage(buf, len)
  {
    LOG(INFO) << "Merger::TMessage >>>>> create message of length " << len << "";
    ResetBit(kIsOwner);
  }
};

//_____________________________________________________________________________
Merger::Merger()
  : FairMQDevice()
  , fEventHeader(NULL)
  , fNofParts(2)
  , fNofPartsPerEventMap()
  , fObjectMap()
  , fInputChannelName("data-in")
  , fOutputChannelName("data-out")
  , fEvRIPair()
  , fEvRIPartTrio()
  , fRet()
  , fNofReceivedMessages(0)
  , fNofSentMessages(0)
{

}

//_____________________________________________________________________________
Merger::~Merger()
{ 

}

//_____________________________________________________________________________
void Merger::Init()
{

  OnData(fInputChannelName, &Merger::MergeData);

}

//_____________________________________________________________________________
bool Merger::MergeData(FairMQParts& parts, int index)
{

  bool printInfo = true;

  //bool dataDuplicationFlag = false;
  int nofReceivedParts = 0; // if set to -1, the data seems to be duplicated

  fNofReceivedMessages++;
  TObject* tempObject;
  TClonesArray* tempArrays[10];
  int nofArrays = 0;

  LOG(INFO) << "Merger::MergeData >>>>> receive " << parts.Size() << " parts" << "";

  for (int ipart = 0 ; ipart < parts.Size() ; ipart++) {

    MergerTMessage tm(parts.At(ipart)->GetData(), parts.At(ipart)->GetSize());
    tempObject = (TObject*)tm.ReadObject(tm.GetClass());

    LOG(INFO) << "Merger::MergeData >>>>> part " << ipart << " with name " << tempObject->GetName() << "";

    if (strcmp(tempObject->GetName(),"EventHeader.") == 0) {
      
      fEventHeader = (EventHeader*)tempObject;
      
      LOG(INFO) << "Merger::MergeData >>>>> got EventHeader part [" << fEventHeader->GetRunId() << "][" << fEventHeader->GetMCEntryNumber() << "][" << fEventHeader->GetPartNo() << "]";
      
      // setting how many parts were received...
      fEvRIPair    .first  = fEventHeader->GetMCEntryNumber();
      fEvRIPair    .second = fEventHeader->GetRunId();
      fEvRIPartTrio.first  = fEvRIPair;
      fEvRIPartTrio.second = fEventHeader->GetPartNo();
      
      MultiMapDef::iterator it3;
      it3 = fObjectMap.find(fEvRIPartTrio);
      if (it3 != fObjectMap.end()) {
	
	LOG(INFO) << "Merger::MergeData >>>>> shouldn't happen, already got objects for part " << fEvRIPartTrio.second 
		  << ", event " << fEvRIPair.first << ", run " << fEvRIPair.second << ". Skipping this message!!!";
	
	nofReceivedParts = -1;
	break; // break the for(ipart) loop, as nothing else is left to do
	
      }
                  
      std::map<std::pair<int,int>,int>::iterator it2;
      it2 = fNofPartsPerEventMap.find(fEvRIPair);
      if (it2 == fNofPartsPerEventMap.end()) {
	
	LOG(INFO) << "Merger::MergeData >>>>> First part of event " << fEvRIPair.first;
	
	fNofPartsPerEventMap[fEvRIPair] = 1;
	nofReceivedParts = 1;
	
      } else {
	
	LOG(INFO) << "Merger::MergeData >>>>> Second part of event " << fEvRIPair.first;
	
	it2->second += 1;
	nofReceivedParts = it2->second;
	
      }
      
      LOG(INFO) << "Merger::MergeData >>>>> got " << nofReceivedParts << " parts of event " << fEvRIPair.first;

    } else { 
      
      tempArrays[nofArrays] = (TClonesArray*)tempObject;
      nofArrays++;

      LOG(INFO) << "Merger::MergeData >>>>> got " << nofArrays << " arrays and " << nofReceivedParts << " parts of event";

    }

  } // end the for(ipart) loop, should have received TCAs in tempArrays and EventHeader

  if (nofReceivedParts == -1) return true;
  
  if (nofReceivedParts != fNofParts) { // got all the parts of the event, have to combine and send message, consisting of objects from tempArrays  
    
    LOG(INFO) << "Merger::MergeData >>>>> not all parts are yet here (" << nofReceivedParts << " of " << fNofParts << ") adding to (size = " << fObjectMap.size() << ")";
    
    LOG(INFO) << "Merger::MergeData >>>>> + " << fEventHeader->GetName() << "[" << fEvRIPartTrio.first.second << "][" << fEvRIPartTrio.first.first << "][" << fEvRIPartTrio.second << "]";
    
    fObjectMap.insert(std::pair<std::pair<std::pair<int,int>,int>,TObject*>(fEvRIPartTrio,(TObject*)fEventHeader));
    
    for (int iarray = 0 ; iarray < nofArrays ; iarray++) {
      
      LOG(INFO) << "Merger::MergeData >>>>> + " << tempArrays[iarray]->GetName() << "[" << fEvRIPartTrio.first.second << "][" << fEvRIPartTrio.first.first << "][" << fEvRIPartTrio.second << "]";
      
      fObjectMap.insert(std::pair<std::pair<std::pair<int,int>,int>,TObject*>(fEvRIPartTrio,(TObject*)tempArrays[iarray]));
      
    }
    
    LOG(INFO) << "Merger::MergeData >>>>> now we have fObjectMap (size = " << fObjectMap.size() << ")";

    if (printInfo) 
      LOG(INFO) << "Merger::MergeData::printInfo >>>>> [" << fEventHeader->GetRunId() << "][" << fEventHeader->GetMCEntryNumber() << "][" << fEventHeader->GetPartNo() << "] Received: " << fNofReceivedMessages << " // Buffered: " << fObjectMap.size() << " // Sent: " << fNofSentMessages << " <<";

  } else { 
	
    int currentEventPart = fEventHeader->GetPartNo();
    for (int iarray = 0 ; iarray < nofArrays; iarray++) {
      
      LOG(INFO) << "Merger::MergeData::printInfo >>>>> before adding, TCA \"" << tempArrays[iarray]->GetName() << "\" has " << tempArrays[iarray]->GetEntries() << " entries.";

      TClonesArray* arrayToAdd;
      
      for (int ieventpart = 0; ieventpart < fNofParts; ieventpart++) {
	
	if ( ieventpart == currentEventPart ) continue;
	
	fEvRIPartTrio.second = ieventpart;
	fRet = fObjectMap.equal_range(fEvRIPartTrio);
	
	for (MultiMapDef::iterator it = fRet.first; it != fRet.second; ++it) {
	  
	  if (strcmp(tempArrays[iarray]->GetName(),it->second->GetName()) == 0) {
	    
	    arrayToAdd = (TClonesArray*)it->second;
	    tempArrays[iarray]->AbsorbObjects(arrayToAdd);
	    LOG(INFO) << "Merger::MergeData::printInfo >>>>> found one!, TCA has now " << tempArrays[iarray]->GetEntries() << " entries.";
	    
	  }
	  
	}
	
      }
      
    }
    
    for (int ieventpart = 0; ieventpart < fNofParts; ieventpart++) {
      
      if ( ieventpart == currentEventPart ) continue;
      fEvRIPartTrio.second = ieventpart;
      fRet = fObjectMap.equal_range(fEvRIPartTrio);
      fObjectMap.erase(fRet.first,fRet.second);
      
    }
    
    TMessage* messageFEH;
    TMessage* messageTCA[10];
    FairMQParts partsOut;
    
    messageFEH = new TMessage(kMESS_OBJECT);
    messageFEH->WriteObject(fEventHeader);
    partsOut.AddPart(NewMessage(messageFEH->Buffer(), 
				messageFEH->BufferSize(), 
				[](void* /*data*/, void* hint) { delete (TMessage*)hint;},messageFEH));

    for (int iarray = 0; iarray < nofArrays; iarray++) {
      
      messageTCA[iarray] = new TMessage(kMESS_OBJECT);
      messageTCA[iarray]->WriteObject(tempArrays[iarray]);
      partsOut.AddPart(NewMessage(messageTCA[iarray]->Buffer(), 
				  messageTCA[iarray]->BufferSize(), 
				  [](void* /*data*/, void* hint) { delete (TMessage*)hint;},messageTCA[iarray]));
      
    }
    
    Send(partsOut, "data-out");
    fNofSentMessages++;
    
    if (printInfo)
      LOG(INFO) << "Merger::MergeData::printInfo >>>>> after Send() [" << fEventHeader->GetRunId() << "][" << fEventHeader->GetMCEntryNumber() << "][" << fEventHeader->GetPartNo() << "] Received: " << fNofReceivedMessages << " // Buffered: " << fObjectMap.size() << " // Sent: " << fNofSentMessages << " <<";
    
  }
  
  return true;

}
