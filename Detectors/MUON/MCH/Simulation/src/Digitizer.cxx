// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/Digitizer.h"
#include "Mapping/Interface/SegmentationCInterface.h"


#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>

using namespace o2::mch;

ClassImp(Digitizer);

void Digitizer::init()
{

  // method to initialize the array of digits
  // To be done:
  //0) adding processing steps and proper translation of charge to adc counts
  //need for "sdigits" (one sdigit per Hit in aliroot) vs. digits (comb. signal per pad) separation?
  //for embedding
  //do we need this level nevertheless for monitoring purposes or to check w.r.t. aliroot?
  //merging at which level?
  //in the old implementation: where does the wire place appear?
  //if it was disregarded: could it cause simu-data differences?
  //1) differentiate between chamber types for signal generation:
  //mapping interface?
  // only one class providing also input for Mathieson function?
  //see: AliMUONResponseV0.h for Mathieson
  //any container structure for digits per plane/station to avoid long loops?
  //2) add initialisation of parameters to be set for digitisation (pad response, hard-ware):
  //check if mathieson or something else already defined centrally
  //3) add a test
  //4) check read-out chain efficiency handling in current code: simple efficiency values? masking?
  //different strategy?
  //5) handling of time dimension: what changes w.r.t. aliroot?
  //probably easiest to have it as a member for DigitStruct
  //and take it into account on Hit generation
  //6) handling of MCtruth information
 
  //list of potentially useful class/struct modifications
  //constructor for DigitStruct (would like emplace_back usage)
  //some containers for different pad structure?
  //containers to have MCtruth, Digit and Hits together?
  //sDigit separately from Digit? really needed?
  //alternative: member: mergeDigits(), how to avoid unnecessary looping
  //create a vector with number of hits or with number of pads?
  
  //TODO time dimension
  //can one avoid these initialisation with this big for-loop as TOF?
  }
}

//______________________________________________________________________

void Digitizer::process(const std::vector<HitType>* hits, std::vector<DigitStruct>* digits)
{
  // hits array of MCH hits for a given simulated event
  for (auto& hit : *hits) {
    //TODO: check if timing handling
    //changes anything for loop structure
    //question: keep memory of DE
    //that are hit several times for merging?
    //vector with detector-ID as column and Pad-ID,hit-ID here?
    //idea would be to loop later over DE entries and add-up charge
    // checking for same pad-ID
    // What is more problematic: memory(loop over everything instead)
    //or cpu (save in more complicated array to avoid looping)?
    //what are typical array and occurences of double hits?
    processHit(hit, mEventTime);
   } // end loop over hits
  //TODO: merge (new member function, add charge) of digits that are on same pad: things to think about in terms of time costly
  //array operations?
  //TODO: any information in Hit order? Any use of order of digit order like in real data?
  //Timing overhead or useful for sth?
  //TODO: threshold effects? anything different w.r.t. old system? HV/gas always same?
  //parameters interface to handle this?
  //TODO: implement FEE response: acts on adc and time info, difference w.r.t. old?
  //what should I know about SAMPA and rest of chain?             

  //  if (!mContinuous) { // fill output container per event, check what happens when filled
    digits->clear();//why in TOF code?
    fillOutputContainer(*digits);
    // }
}

//______________________________________________________________________

Int_t Digitizer::processHit(const HitType &hit,Double_t event_time)
{

  //hit position
  Float_t pos[3] = { hit.GetX(), hit.GetY(), hit.GetZ() };
  //hit energy deposition
  Float_t edepos = hit.GetEnergyLoss();
  //hit time information
  Float_t time = hit.GetTime();

  //number of digits added for this hit
  Int_t ndigits=0;
  //overhead to create these 2 struct?
  MchSegmentationHandle seghandbend = mchSegmentationConstruct(hit.GetDetectorID(),kTRUE);
  MchSegmentationHandle seghandnon  = mchSegmentationConstruct(hit.GetDetectorID(),kFALSE);
  
  Int_t paduidbend = mchSegmentationFindPadByPosition(seghandbend,pos[0],pos[1]);
  Int_t paduidnon  = mchSegmentationFindPadByPosition(seghandnon,pos[0],pos[1]);
  //TODO: charge sharing between planes, possibility to do random seeding in some controlled way to
  // be able to be 100% reproducible if wanted? or already given up on geant level?
  Float_t fracplane = 0.5;
  Float_t chargebend= fracplane*edepos;
  Float_t chargenon = (1.0-fracplane)*edepos;
  //TODO: charge spread on planes (new member function): strange wire position not used in AliMUONResponseV0.cxx?
  //TODO get neighbouring pads (new member function): via extension of MpPad?
  //just doing the 9 pads sufficient or need to go for variable area?
  //TODO: loop over neighbouring pads having charge from the one hit treated here
  //TODO: provide interface to retrieve equivalent of sDigits for embedding: not clear how exactly used in past
  //DigitStruct has no constructor yet!
  //Is it ok to abuse digits to  here the charge instead of adc?
  //or different class/struct to make it clear?
  mDigits.emplace_back(paduidbend,chargebend);
  mDigits.emplace_back(paduidnon, chargenon);
   
  return ndigits;

}

//______________________________________________________________________
Float_t Digitizer::getCharge(Float_t eDep)
{
  // transform deposited energy in collected charge
  // to be modified with Mathieson function
  //TODO put corresponding path in old implementation

  return 0.0;
}
//______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<DigitStruct>& digits)
{

  // filling the digit container
  if (mDigits.empty())
    return;
  
  auto itBeg = mDigits.begin();
  auto iter = itBeg;
  for (; iter != mDigits.end(); ++iter) {
    Digit& dig = iter->second;
    digits.emplace_back(dig);
    //Problem DigitStruct has no constructor
    //add one?
  }

  mDigits.erase(itBeg, iter);//need to erase 
  //need to clear hits?
}
//______________________________________________________________________
void Digitizer::flushOutputContainer(std::vector<DigitStruct>& digits)
{ // flush all residual buffered data
  // TO be implemented
  //TODO: check if used in Task
  fillOutputContainer(digits);
}
