#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"

#include "TMessage.h"

#include "MFTBase/EventHeader.h"
#include "MFTReconstruction/devices/Merger.h"

using namespace AliceO2::MFT;

using namespace std;

// special class to expose protected TMessage constructor
//_____________________________________________________________________________
class TMessage2 : public TMessage
{
  public:
  TMessage2(void* buf, Int_t len)
    : TMessage(buf, len)
  {
    LOG(INFO) << "Merger::TMessage2 >>>>> create message of length " << len << "";
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
{

}

//_____________________________________________________________________________
Merger::~Merger()
{ 

}

//_____________________________________________________________________________
void Merger::Init()
{

}

//_____________________________________________________________________________
void Merger::Run()
{

  typedef std::multimap<std::pair<std::pair<int,int>,int>,TObject*> MultiMapDef;

  int fNofReceivedMessages = 0;
  int fNofSentMessages = 0;
  bool printInfo = true;

  std::pair<int, int> evRIPair;
  std::pair<std::pair<int,int>,int> evRIPartTrio;
  std::pair<MultiMapDef::iterator, MultiMapDef::iterator> ret;
  //  bool dataDuplicationFlag = false;
  int nofReceivedParts = 0; // if set to -1, the data seems to be duplicated

  while (CheckCurrentState(RUNNING)) {

    FairMQParts parts;
    
    if (Receive(parts, "data-in") >= 0) {

      fNofReceivedMessages++;
      //dataDuplicationFlag = false;
      nofReceivedParts = 0;
      TObject*      tempObject;
      TClonesArray* tempArrays[10];
      int nofArrays = 0;

      LOG(INFO) << "Merger::Run >>>>> receive " << parts.Size() << " parts" << "";

      for (int ipart = 0 ; ipart < parts.Size() ; ipart++) {

	TMessage2 tm(parts.At(ipart)->GetData(), parts.At(ipart)->GetSize());
	tempObject = (TObject*)tm.ReadObject(tm.GetClass());

	if (strcmp(tempObject->GetName(),"EventHeader.") == 0) {

	  fEventHeader = (EventHeader*)tempObject;

	  LOG(INFO) << "Merger::Run >>>>> got EventHeader part [" << fEventHeader->GetRunId() << "][" << fEventHeader->GetMCEntryNumber() << "][" << fEventHeader->GetPartNo() << "]";
                  
	  // setting how many parts were received...
	  evRIPair    .first  = fEventHeader->GetMCEntryNumber();
	  evRIPair    .second = fEventHeader->GetRunId();
	  evRIPartTrio.first  = evRIPair;
	  evRIPartTrio.second = fEventHeader->GetPartNo();
          
	  MultiMapDef::iterator it3;
	  it3 = fObjectMap.find(evRIPartTrio);
	  if (it3 != fObjectMap.end()) {

	    LOG(INFO) << "Merger::Run >>>>> shouldn't happen, already got objects for part " << evRIPartTrio.second 
		      << ", event " << evRIPair.first << ", run " << evRIPair.second << ". Skipping this message!!!";

	    nofReceivedParts = -1;
	    break; // break the for(ipart) loop, as nothing else is left to do

	  }
                  
	  std::map<std::pair<int,int>,int>::iterator it2;
	  it2 = fNofPartsPerEventMap.find(evRIPair);
	  if (it2 == fNofPartsPerEventMap.end()) {

	    LOG(INFO) << "Merger::Run >>>>> First part of event " << evRIPair.first;

	    fNofPartsPerEventMap[evRIPair] = 1;
	    nofReceivedParts = 1;

	  } else {

	    it2->second+=1;
	    nofReceivedParts = it2->second;

	  }

	  LOG(INFO) << "Merger::Run >>>>> got " << nofReceivedParts << " parts of event " << evRIPair.first;

	} else { 

	  tempArrays[nofArrays] = (TClonesArray*)tempObject;
	  nofArrays++;

	}

      } // end the for(ipart) loop, should have received TCAs in tempArrays and EventHeader

      if (nofReceivedParts == -1) continue;
      
      if (nofReceivedParts != fNofParts) { // got all the parts of the event, have to combine and send message, consisting of objects from tempArrays  

	LOG(INFO) << "Merger::Run >>>>> not all parts are yet here (" << nofReceivedParts << " of " << fNofParts << ") adding to (size = " << fObjectMap.size() << ")";

	LOG(INFO) << "Merger::Run >>>>> + " << fEventHeader->GetName() << "[" << evRIPartTrio.first.second << "][" << evRIPartTrio.first.first << "][" << evRIPartTrio.second << "]";

	fObjectMap.insert(std::pair<std::pair<std::pair<int,int>,int>,TObject*>(evRIPartTrio,(TObject*)fEventHeader));

	for (int iarray = 0 ; iarray < nofArrays ; iarray++) {
	  
	  LOG(INFO) << "Merger::Run >>>>> + " << tempArrays[iarray]->GetName() << "[" << evRIPartTrio.first.second << "][" << evRIPartTrio.first.first << "][" << evRIPartTrio.second << "]";

	  fObjectMap.insert(std::pair<std::pair<std::pair<int,int>,int>,TObject*>(evRIPartTrio,(TObject*)tempArrays[iarray]));

	}

	LOG(INFO) << "Merger::Run >>>>> now we have fObjectMap (size = " << fObjectMap.size() << ")";

	if (printInfo) 
	  LOG(INFO) << "Merger::Run::printInfo >>>>> [" << fEventHeader->GetRunId() << "][" << fEventHeader->GetMCEntryNumber() << "][" << fEventHeader->GetPartNo() << "] Received: " << fNofReceivedMessages << " // Buffered: " << fObjectMap.size() << " // Sent: " << fNofSentMessages << " <<";

      } else { 
	
	int currentEventPart = fEventHeader->GetPartNo();
	for (int iarray = 0 ; iarray < nofArrays; iarray++) {

	  LOG(TRACE) << "BEFORE ADDING, TCA \"" << tempArrays[iarray]->GetName() << "\" has " << tempArrays[iarray]->GetEntries() << " entries.";
	  TClonesArray* arrayToAdd;
          
	  for (int ieventpart = 0; ieventpart < fNofParts; ieventpart++) {

	    if ( ieventpart == currentEventPart ) continue;

	    evRIPartTrio.second = ieventpart;
	    ret = fObjectMap.equal_range(evRIPartTrio);

	    for (MultiMapDef::iterator it = ret.first; it != ret.second; ++it) {

	      if (strcmp(tempArrays[iarray]->GetName(),it->second->GetName()) == 0) {

		arrayToAdd = (TClonesArray*)it->second;
		tempArrays[iarray]->AbsorbObjects(arrayToAdd);
		LOG(TRACE) << "FOUND ONE!, TCA has now " << tempArrays[iarray]->GetEntries() << " entries.";

	      }

	    }

	  }

	}
	
	for (int ieventpart = 0; ieventpart < fNofParts; ieventpart++) {

	  if ( ieventpart == currentEventPart ) continue;
	  evRIPartTrio.second = ieventpart;
	  ret = fObjectMap.equal_range(evRIPartTrio);
	  fObjectMap.erase(ret.first,ret.second);

	}
	
	TMessage* messageFEH;
	TMessage* messageTCA[10];
	FairMQParts partsOut;
        
	messageFEH = new TMessage(kMESS_OBJECT);
	messageFEH->WriteObject(fEventHeader);
	partsOut.AddPart(NewMessage(messageFEH->Buffer(), 
				    messageFEH->BufferSize(), 
				    [](void* /*data*/, void* hint) { delete (TMessage*)hint;},
				    messageFEH));
	for (int iarray = 0; iarray < nofArrays; iarray++) {

	  messageTCA[iarray] = new TMessage(kMESS_OBJECT);
	  messageTCA[iarray]->WriteObject(tempArrays[iarray]);
	  partsOut.AddPart(NewMessage(messageTCA[iarray]->Buffer(), 
				      messageTCA[iarray]->BufferSize(), 
				      [](void* /*data*/, void* hint) { delete (TMessage*)hint;},
				      messageTCA[iarray]));

	}

	Send(partsOut, "data-out");
	fNofSentMessages++;
	
	if (printInfo)
	  LOG(INFO) << "Merger::Run::printInfo >>>>> after Send() [" << fEventHeader->GetRunId() << "][" << fEventHeader->GetMCEntryNumber() << "][" << fEventHeader->GetPartNo() << "] Received: " << fNofReceivedMessages << " // Buffered: " << fObjectMap.size() << " // Sent: " << fNofSentMessages << " <<";
	
      }
    }
  }
  
}

