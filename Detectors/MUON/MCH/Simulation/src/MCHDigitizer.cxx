// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/MCHDigitizer.h"

#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>

using namespace o2::mch;

namespace
{
std::map<int, int> createDEMap()
{
  std::map<int, int> m;
  int i{ 0 };
  o2::mch::mapping::forEachDetectionElement([&m, &i](int deid) {
    m[deid] = i++;
  });
  return m;
}

std::vector<o2::mch::mapping::Segmentation> createSegmentations()
{
  std::vector<o2::mch::mapping::Segmentation> segs;

  o2::mch::mapping::forEachDetectionElement([&segs](int deid) {
    segs.emplace_back(deid);
  });
  return segs;
}
} // namespace

MCHDigitizer::MCHDigitizer(Int_t) : mReadoutWindowCurrent(0), mdetID{ createDEMap() }, mSeg{ createSegmentations() } 
{
}

void MCHDigitizer::init()
{
// initialize the array of detector segmentation's
  //std::vector<mapping::Segmentation*> mSegbend;
  // std::vector<mapping::Segmentation*> mSegbend;
  //how does programme know about proper storage footprint
  //  for(Int_t i=0; i<mNdE; ++i){
  //for (auto deid : mdetID) {
  //  MCHDigitizer::mSegbend.push_back(mapping::Segmentation{i, true});
      //    //mapping::forEachDetectionElement([&mSegbend](int deid){ segs.emplace_back(deid, true); });
  // }
    //mapping::Segmentation itb{i,true};
    //  mSegbend.emplace_back(i,true); //= mapping::Segmentation(i,kTRUE);
    //    mapping::Segmentation itn{i, mapping::Segmentation{i,false}};
    //mSegnon.emplace_back(i, mapping::Segmentation{i,false});
  // }
  
  //  std::vector<mapping::Segmentation> mSegbend;
  //for(auto deid : mdetID) mSegbend.push_back(mapping::Segmentation{deid, true});
  //  for(auto deid : mdetID) mSegbend.emplace_back(deid, mapping::Segmentation{deid, true});

  // To be done:
  //0) adding processing steps and proper translation of charge to adc counts
  //need for "sdigits" (one sdigit per Hit in aliroot) vs. digits (comb. signal per pad) two steps
  //1) differentiate between chamber types for signal generation:
  //2) add initialisation of parameters to be set for digitisation (pad response, hard-ware) at central place
  //3) add a test
  //4) check read-out chain efficiency handling in current code: simple efficiency values? masking?
  //different strategy?
  //5) handling of time dimension: what changes w.r.t. aliroot check HMPID
  //6) handling of MCtruth information
 
  //TODO time dimension
  //can one avoid these initialisation with this big for-loop as TOF?
}

//______________________________________________________________________

void MCHDigitizer::process(const std::vector<Hit> hits, std::vector<Digit>& digits)
{
  // hits array of MCH hits for a given simulated event
  for (auto& hit : hits) {
    //TODO: check if change for time structure
    processHit(hit, mEventTime);
   } // end loop over hits
  //TODO: merge (new member function, add charge) of digits that are on same pad:
  //things to think about in terms of time costly

    digits.clear();
    fillOutputContainer(digits);

}

//______________________________________________________________________
Int_t MCHDigitizer::processHit(const Hit &hit,Double_t event_time)
{

  //hit position(cm)
  Float_t pos[3] = { hit.GetX(), hit.GetY(), hit.GetZ() };
  //convert energy to charge, float enough?
  Float_t charge = mMuonresponse.etocharge(hit.GetEnergyLoss());
  //time information
  Float_t time = hit.GetTime();//how to trace
  Int_t detID = hit.GetDetectorID();
  //get index for this detID
  Int_t indexID = mdetID.at(detID);
  //# digits for hit
  Int_t ndigits=0;
  
  Float_t anodpos = mMuonresponse.getAnod(pos[0],detID);

  //TODO: charge sharing between planes,
  //possibility to do random seeding in controlled way
  // be able to be 100% reproducible if wanted? or already given up on geant level?
  //signal will be around neighbouring anode-wire 
  //distance of impact point and anode, needed for charge sharing
  Float_t anoddis = TMath::Abs(pos[0]-anodpos);
  //question on how to retrieve charge fraction deposited in both half spaces
  //throw a dice?
  //should be related to electrons fluctuating out/in one/both halves (independent x)
  //  Float_t fracplane = 0.5;//to be replaced by function of annodis
  Float_t fracplane = mMuonresponse.chargeCorr();//should become a function of anoddis
  Float_t chargebend= fracplane*charge;
  Float_t chargenon = charge/fracplane;
  //last line  from Aliroot, not understood why
  //since charge = charchbend+chargenon and not multiplication
  Float_t signal = 0.0;

  //borders of charge gen. 
  Double_t xMin = anodpos-mMuonresponse.getQspreadX()*0.5;
  Double_t xMax = anodpos+mMuonresponse.getQspreadX()*0.5;

  Double_t yMin = pos[1]-mMuonresponse.getQspreadY()*0.5;
  Double_t yMax = pos[1]+mMuonresponse.getQspreadY()*0.5;
  
  //pad-borders
  Float_t xmin =0.0;
  Float_t xmax =0.0;
  Float_t ymin =0.0;
  Float_t ymax =0.0;
 
  //use DetectorID to get area for signal induction               
  // SegmentationImpl3.h: Return the list of paduids for the pads contained in the box {xmin,ymin,xmax,ymax}.      
  //  std::vector<int> getPadUids(double xmin, double ymin, double xmax, double ymax) const;
  //  mPadIDsbend = getPadUid(xMin,xMax,yMin,yMax);
  
  //is this available via Segmentation.h interface already?
  
  //TEST with only one pad
  Int_t padidbend=0;
  Int_t padidnon=0;
  bool padexists = mSeg[indexID].findPadPairByPosition(anodpos,pos[1],padidbend,padidnon);
  if(!padexists) return 0; //to be decided what to do
  //correct coordinate system? how misalignment enters?
  /*mPadIDsbend = mSegbend.getPadUids(xMin,xMax,yMin,yMax);
    mPadIDsnon  = mSegnon.getPadUids(xMin,xMax,yMin,yMax);
  */
    /* for(auto & padidbend : mPadIDsbend){
    //retrieve coordinates for each pad*/
  xmin =  mSeg[indexID].padPositionX(padidbend)-mSeg[indexID].padSizeX(padidbend)*0.5;
  xmax =  mSeg[indexID].padPositionX(padidbend)+mSeg[indexID].padSizeX(padidbend)*0.5;
  ymin =  mSeg[indexID].padPositionY(padidbend)-mSeg[indexID].padSizeY(padidbend)*0.5;
  ymax =  mSeg[indexID].padPositionY(padidbend)+mSeg[indexID].padSizeY(padidbend)*0.5;
  //what happens if at edge of detector?
  
  // 1st step integrate induced charge for each pad
  signal = mMuonresponse.chargePad(anodpos,pos[1],xmin,xmax,ymin,ymax,detID,chargebend);
  if(signal>mMuonresponse.getChargeThreshold() && signal<mMuonresponse.getChargeSat()){
      //2nd condition in Aliroot said to be only for backward compatibility
      //to be seen...means that there is no digit, if signal above... strange!
      //2n step TODO: pad response function, electronic response
      signal = mMuonresponse.response(detID,signal);
      mDigits.emplace_back(padidbend,signal);//how trace time?
      ++ndigits;
    }
    /*}
	   for(auto & padidnon : mPadIDsnon){*/
    //retrieve coordinates for each pad
    xmin =  mSeg[indexID].padPositionX(padidnon)-mSeg[indexID].padSizeX(padidnon)*0.5;
    xmax =  mSeg[indexID].padPositionX(padidnon)+mSeg[indexID].padSizeX(padidnon)*0.5;
    ymin =  mSeg[indexID].padPositionY(padidnon)-mSeg[indexID].padSizeY(padidnon)*0.5;
    ymax =  mSeg[indexID].padPositionY(padidnon)+mSeg[indexID].padSizeY(padidnon)*0.5;
    
    //retrieve charge for given x,y with Mathieson
    signal = mMuonresponse.chargePad(anodpos,pos[1],xmin,xmax,ymin,ymax,detID,chargenon);
    if(signal>mMuonresponse.getChargeThreshold() && signal<mMuonresponse.getChargeSat()){
      signal = mMuonresponse.response(detID,signal);
      mDigits.emplace_back(padidnon,signal);//how is time propagated?
      ++ndigits;
    }
    /*}*/	

    return ndigits;
}
//_____________________________________________________________________
 std::vector<int> MCHDigitizer::getPadUid(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax, bool bend){
  //to be implemented?
   //getting pad-ID of xMin if existing?
   
  return mPadIDsbend;

}
//______________________________________________________________________
//not clear if needed for DPL or modifications required
void MCHDigitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  // filling the digit container
  if (mDigits.empty())
    return;
  
  auto itBeg = mDigits.begin();
  auto iter = itBeg;
  for (; iter != mDigits.end(); ++iter) {
    digits.emplace_back(*iter);
  }
  
  mDigits.erase(itBeg, iter);
}
//______________________________________________________________________
void MCHDigitizer::flushOutputContainer(std::vector<Digit>& digits)
{ // flush all residual buffered data
  //not clear if neede in DPL
  fillOutputContainer(digits);
}

