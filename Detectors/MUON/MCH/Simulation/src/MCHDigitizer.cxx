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

MCHDigitizer::MCHDigitizer(int) :  mdetID{ createDEMap() }, mSeg{ createSegmentations() } 
{
}

void MCHDigitizer::init()
{
  // To be done:
  //1) add a test
  //2) check read-out chain efficiency handling in current code: simple efficiency values? masking?
  //different strategy?
  //3) handling of time dimension: what changes w.r.t. aliroot check HMPID
  //4) handling of MCtruth information 
}

//______________________________________________________________________

void MCHDigitizer::process(const std::vector<Hit> hits, std::vector<Digit>& digits)
{
  //array of MCH hits for a given simulated event
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
int MCHDigitizer::processHit(const Hit &hit,double event_time)
{

  //hit position(cm)
  //float pos[3] = { hit.GetX(), hit.GetY(), hit.GetZ() };
  Vector3D<float> pos(hit.GetX(),hit.GetY(),hit.GetZ());
  //hit has global coordinates
  //convert energy to charge, float enough?
  float charge = mMuonresponse.etocharge(hit.GetEnergyLoss());
  std::cout << "charge " << charge << std::endl;
  std::cout << "pos.X() " << pos.X() << std::endl;
  std::cout << "pos.Y() " << pos.Y() << std::endl;
  std::cout << "pos.Z() " << pos.Z() << std::endl;


  //time information
  float time = hit.GetTime();//how to trace
  int detID = hit.GetDetectorID();
  //get index for this detID
  int indexID = mdetID.at(detID);
  //# digits for hit
  int ndigits=0;
  //transformation from local to global
  // auto t = o2::mch::getTransformation(detID, gMgr);
  //get trafo from global to local
  //  t.Invert();
  // t(pos);
  
  float anodpos = mMuonresponse.getAnod(pos.X(),detID);//already local needed?
  std::cout << "anodpos " << anodpos << std::endl; 
  //TODO/Questions:
  //- charge sharing between planes wrong in Aliroot copied here?
  //- possibility to do random seeding in controlled way? 
  // 100% reproducible if wanted? or already given up on geant level?
  //signal will be around neighbouring anode-wire 
  //distance of impact point and anode, needed for charge sharing
  float anoddis = TMath::Abs(pos.X()-anodpos);
  //throw a dice?
  //should be related to electrons fluctuating out/in one/both halves (independent x)
  float fracplane = mMuonresponse.chargeCorr();//should become a function of anoddis
  std::cout << "fracplane " << fracplane << std::endl;
  float chargebend= fracplane*charge;
  float chargenon = charge/fracplane;
  //last line  from Aliroot, not understood why
  //since charge = charchbend+chargenon and not multiplication
  float signal = 0.0;

  //borders of charge gen. 
  double xMin = anodpos-mMuonresponse.getQspreadX()*0.5;
  double xMax = anodpos+mMuonresponse.getQspreadX()*0.5;
  double yMin = pos.Y()-mMuonresponse.getQspreadY()*0.5;
  double yMax = pos.Y()+mMuonresponse.getQspreadY()*0.5;
  std::cout << " xMin " << xMin << std::endl;
  std::cout << " xMax " << xMax << std::endl;
  std::cout << " yMin " << yMin << std::endl;
  std::cout << " yMax " << yMax << std::endl;


  //pad-borders
  float xmin =0.0;
  float xmax =0.0;
  float ymin =0.0;
  float ymax =0.0;
 
  //use DetectorID to get area for signal induction               
  //single pad, used only as check...
  //to be seen if needed
  int padidbendcent=0;
  int padidnoncent=0;
  bool padexists = mSeg[indexID].findPadPairByPosition(anodpos,pos.Y(),padidbendcent,padidnoncent);
  std::cout << "padexists " << padexists << std::endl;
  std::cout << "padidbendcent " << padidbendcent << std::endl;
  std::cout << "padidnoncent "	<< padidnoncent << std::endl;
  if(!padexists) return 0; //to be decided if needed
  
  //need to keep both electros separated since afterwards signal generation on each plane
  //correct coordinate system? how misalignment enters?
  std::vector<int> padIDsbend;
  std::vector<int> padIDsnon;

  //retrieve pads with signal
  mSeg[indexID].Bending().forEachPadInArea(xMin,yMin,xMax,yMax, [&padIDsbend](int padid){
      padIDsbend.emplace_back(padid); });
  mSeg[indexID].NonBending().forEachPadInArea(xMin,yMin,xMax,yMax, [&padIDsnon](int padid){
      padIDsnon.emplace_back(padid); });

  //induce signal pad-by-pad: bending
  for(auto & padidbend : padIDsbend){
    //retrieve coordinates for each pad
    xmin =  (anodpos-mSeg[indexID].padPositionX(padidbend))-mSeg[indexID].padSizeX(padidbend)*0.5;
    xmax =  xmin+mSeg[indexID].padSizeX(padidbend);
    ymin =  (pos.Y()-mSeg[indexID].padPositionY(padidbend))-mSeg[indexID].padSizeY(padidbend)*0.5;
    ymax =  ymin+mSeg[indexID].padSizeY(padidbend);
  // 1st step integrate induced charge for each pad
    
  signal = mMuonresponse.chargePad(xmin,xmax,ymin,ymax,detID,chargebend);
  std::cout << "signal before response " <<  signal <<std::endl;
  // if(signal>mMuonresponse.getChargeThreshold()
  //&&     signal<mMuonresponse.getChargeSat()//not yet tuned
  //  ){
  //translate charge in signal
  signal = mMuonresponse.response(signal,detID);
  //write digit
  mDigits.emplace_back(padidbend,signal);//how trace time?
  ++ndigits;
  //  }
  }
  //induce signal pad-by-pad: nonbending
  for(auto & padidnon : padIDsnon){
    //retrieve coordinates for each pad
    xmin =  (anodpos-mSeg[indexID].padPositionX(padidnon))-mSeg[indexID].padSizeX(padidnon)*0.5;
    xmax =  xmin+mSeg[indexID].padSizeX(padidnon);
    ymin =  (pos.Y()-mSeg[indexID].padPositionY(padidnon))-mSeg[indexID].padSizeY(padidnon)*0.5;
    ymax =  ymin+mSeg[indexID].padSizeY(padidnon);
    
    // 1st step integrate induced charge for each pad
    signal = mMuonresponse.chargePad(xmin,xmax,ymin,ymax,detID,chargenon);
    std::cout	<< "signal before response " <<	 signal <<std::endl;
    //check if signal above threshold
    // if(signal>mMuonresponse.getChargeThreshold()
       //&& signal<mMuonresponse.getChargeSat()//not yet tuned
    // ){
    //translate charge in signal
    signal = mMuonresponse.response(signal,detID);
    //write digit
    mDigits.emplace_back(padidnon,signal);//how trace time?
    ++ndigits;
    // }
  }	
  return ndigits;
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

