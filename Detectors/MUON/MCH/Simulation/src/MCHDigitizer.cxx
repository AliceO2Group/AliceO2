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

std::map<int,int> MCHDigitizer::mdetID{
  {100,0}, {101,0}, {102,2}, {103,3},
  {200,4}, {201,5}, {202,6}, {203,7},
  {300,8}, {301,9}, {302,10}, {303,11},
  {400,12}, {401,13}, {402,14}, {403,15},
  {500,16}, {501,17}, {502,18}, {503,19}, {504,20}, {505,21}, {506,22}, {507,23}, {508,24},
  {509,25}, {510,26}, {511,27}, {512,28}, {513,29}, {514,30}, {515,31}, {516,32}, {517,33},  
  {600,34}, {601,35}, {602,36}, {603,37}, {604,38}, {605,39}, {606,40}, {607,41}, {608,42},
  {609,43}, {610,44}, {611,45}, {612,46}, {613,47}, {614,48}, {615,49}, {616,50}, {617,51},
  {700,52}, {701,53}, {702,54}, {703,55}, {704,56}, {705,57}, {706,58}, {707,59}, {708,60},
  {709,61}, {710,62}, {711,63}, {712,64}, {713,65}, {714,66}, {715,67}, {716,68}, {717,69},
  {718,70}, {719,71}, {720,72}, {721,73}, {722,74}, {723,75}, {724,76}, {725,77},
  {800,78}, {801,79}, {802,80}, {803,81}, {804,82}, {805,83}, {806,84}, {807,85}, {808,86},
  {809,87}, {810,88}, {811,89}, {812,90}, {813,91}, {814,92}, {815,93}, {816,94}, {817,95},
  {818,96}, {819,97}, {820,98}, {821,99}, {822,100}, {823,101},{824,102},{825,103},
  {900,104}, {901,105}, {902,106}, {903,107}, {904,108}, {905,109}, {906,110}, {907,111}, {908,112},
  {909,113}, {910,114}, {911,115}, {912,116}, {913,117}, {914,118}, {915,119}, {916,120}, {917,121},
  {918,122}, {919,123}, {920,124}, {921,125}, {922,126}, {923,127}, {924,128}, {925,129},
  {1000,130}, {1001,131}, {1002,132}, {1003,133}, {1004,134}, {1005,135}, {1006,136}, {1007,137}, {1008,138},
  {1009,139}, {1010,140}, {1011,141}, {1012,142}, {1013,143}, {1014,144}, {1015,145}, {1016,146},  {1017,147},
  {1018,148}, {1019,149}, {1020,150}, {1021,151}, {1022,152}, {1023,153}, {1024,154}, {1025,155}
};

std::vector<mapping::Segmentation> MCHDigitizer::mSegbend(
							  {mapping::Segmentation(100,true), mapping::Segmentation(101,true), mapping::Segmentation(102,true) , mapping::Segmentation(103,true),
							      mapping::Segmentation(200,true),  mapping::Segmentation(201,true),  mapping::Segmentation(202,true),  mapping::Segmentation(203,true),
							      mapping::Segmentation(300,true),  mapping::Segmentation(301,true),  mapping::Segmentation(302,true),  mapping::Segmentation(303,true),
							      mapping::Segmentation(400,true),  mapping::Segmentation(401,true),  mapping::Segmentation(402,true),  mapping::Segmentation(403,true),
							      mapping::Segmentation(500,true),  mapping::Segmentation(501,true),  mapping::Segmentation(502,true),  mapping::Segmentation(503,true),
							      mapping::Segmentation(504,true),  mapping::Segmentation(505,true),  mapping::Segmentation(506,true),  mapping::Segmentation(507,true),
							      mapping::Segmentation(508,true),  mapping::Segmentation(509,true),  mapping::Segmentation(510,true),  mapping::Segmentation(511,true),
							      mapping::Segmentation(512,true),  mapping::Segmentation(513,true),  mapping::Segmentation(514,true),  mapping::Segmentation(515,true),
							      mapping::Segmentation(516,true),  mapping::Segmentation(517,true),
							      mapping::Segmentation(600,true),  mapping::Segmentation(601,true),  mapping::Segmentation(602,true),  mapping::Segmentation(603,true),
							      mapping::Segmentation(604,true),  mapping::Segmentation(605,true),  mapping::Segmentation(606,true),  mapping::Segmentation(607,true),
							      mapping::Segmentation(608,true),  mapping::Segmentation(609,true),  mapping::Segmentation(610,true),  mapping::Segmentation(611,true),
							      mapping::Segmentation(612,true),  mapping::Segmentation(613,true),  mapping::Segmentation(614,true),  mapping::Segmentation(615,true),
							      mapping::Segmentation(616,true),  mapping::Segmentation(617,true),  mapping::Segmentation(700,true),  mapping::Segmentation(701,true),
							      mapping::Segmentation(702,true),  mapping::Segmentation(703,true),  mapping::Segmentation(704,true),  mapping::Segmentation(705,true),
							      mapping::Segmentation(706,true),  mapping::Segmentation(707,true),  mapping::Segmentation(708,true),  mapping::Segmentation(709,true),
							      mapping::Segmentation(710,true),  mapping::Segmentation(711,true),  mapping::Segmentation(712,true),  mapping::Segmentation(713,true),
							      mapping::Segmentation(714,true),  mapping::Segmentation(715,true),  mapping::Segmentation(716,true),  mapping::Segmentation(717,true),
							      mapping::Segmentation(718,true),  mapping::Segmentation(719,true),  mapping::Segmentation(720,true),  mapping::Segmentation(721,true),
							      mapping::Segmentation(722,true),  mapping::Segmentation(723,true),  mapping::Segmentation(724,true),  mapping::Segmentation(725,true),
							      mapping::Segmentation(800,true),  mapping::Segmentation(801,true),  mapping::Segmentation(802,true),  mapping::Segmentation(803,true),
							      mapping::Segmentation(804,true),  mapping::Segmentation(805,true),  mapping::Segmentation(806,true),  mapping::Segmentation(807,true),
							      mapping::Segmentation(808,true),  mapping::Segmentation(809,true),  mapping::Segmentation(810,true),  mapping::Segmentation(811,true),
							      mapping::Segmentation(812,true),  mapping::Segmentation(813,true),  mapping::Segmentation(814,true),  mapping::Segmentation(815,true),
							      mapping::Segmentation(816,true),  mapping::Segmentation(817,true),
							      mapping::Segmentation(818,true),  mapping::Segmentation(819,true),  mapping::Segmentation(820,true),  mapping::Segmentation(821,true),
							      mapping::Segmentation(822,true),  mapping::Segmentation(823,true),  mapping::Segmentation(824,true),  mapping::Segmentation(825,true),
							      mapping::Segmentation(900,true),  mapping::Segmentation(901,true),  mapping::Segmentation(902,true),  mapping::Segmentation(903,true),
							      mapping::Segmentation(904,true),  mapping::Segmentation(905,true),  mapping::Segmentation(906,true),  mapping::Segmentation(907,true),
							      mapping::Segmentation(908,true),  mapping::Segmentation(909,true),  mapping::Segmentation(910,true),  mapping::Segmentation(911,true),
							      mapping::Segmentation(912,true),  mapping::Segmentation(913,true),  mapping::Segmentation(914,true),  mapping::Segmentation(915,true),
							      mapping::Segmentation(916,true),  mapping::Segmentation(917,true),  mapping::Segmentation(918,true),  mapping::Segmentation(919,true),
							      mapping::Segmentation(920,true),  mapping::Segmentation(921,true),  mapping::Segmentation(922,true),  mapping::Segmentation(923,true),
							      mapping::Segmentation(924,true),  mapping::Segmentation(925,true),  mapping::Segmentation(1000,true), mapping::Segmentation(1001,true),
							      mapping::Segmentation(1002,true), mapping::Segmentation(1003,true), mapping::Segmentation(1004,true), mapping::Segmentation(1005,true),
							      mapping::Segmentation(1006,true), mapping::Segmentation(1007,true), mapping::Segmentation(1008,true), mapping::Segmentation(1009,true),
							      mapping::Segmentation(1010,true), mapping::Segmentation(1011,true), mapping::Segmentation(1012,true), mapping::Segmentation(1013,true),
							      mapping::Segmentation(1014,true), mapping::Segmentation(1015,true), mapping::Segmentation(1016,true), mapping::Segmentation(1017,true),
							      mapping::Segmentation(1018,true), mapping::Segmentation(1019,true), mapping::Segmentation(1020,true), mapping::Segmentation(1021,true),
							      mapping::Segmentation(1022,true), mapping::Segmentation(1023,true), mapping::Segmentation(1024,true), mapping::Segmentation(1025,true)}
							  );
std::vector<mapping::Segmentation> MCHDigitizer::mSegnon(
							 {mapping::Segmentation(100,false), mapping::Segmentation(101,false), mapping::Segmentation(102,false), mapping::Segmentation(103,false),
							     mapping::Segmentation(200,false),  mapping::Segmentation(201,false),  mapping::Segmentation(202,false),  mapping::Segmentation(203,false),
							     mapping::Segmentation(300,false),  mapping::Segmentation(301,false),  mapping::Segmentation(302,false),  mapping::Segmentation(303,false),
							     mapping::Segmentation(400,false),  mapping::Segmentation(401,false),  mapping::Segmentation(402,false),  mapping::Segmentation(403,false),
							     mapping::Segmentation(500,false),  mapping::Segmentation(501,false),  mapping::Segmentation(502,false),  mapping::Segmentation(503,false),
							     mapping::Segmentation(504,false),  mapping::Segmentation(505,false),  mapping::Segmentation(506,false),  mapping::Segmentation(507,false),
							     mapping::Segmentation(508,false),  mapping::Segmentation(509,false),  mapping::Segmentation(510,false),  mapping::Segmentation(511,false),
							     mapping::Segmentation(512,false),  mapping::Segmentation(513,false),  mapping::Segmentation(514,false),  mapping::Segmentation(515,false),
							     mapping::Segmentation(516,false),  mapping::Segmentation(517,false),  mapping::Segmentation(600,false),  mapping::Segmentation(601,false),
							     mapping::Segmentation(602,false),  mapping::Segmentation(603,false),  mapping::Segmentation(604,false),  mapping::Segmentation(605,false),
							     mapping::Segmentation(606,false),  mapping::Segmentation(607,false),  mapping::Segmentation(608,false),  mapping::Segmentation(609,false),
							     mapping::Segmentation(610,false),  mapping::Segmentation(611,false),  mapping::Segmentation(612,false),  mapping::Segmentation(613,false),
							     mapping::Segmentation(614,false),  mapping::Segmentation(615,false),  mapping::Segmentation(616,false),  mapping::Segmentation(617,false),
							     mapping::Segmentation(700,false),  mapping::Segmentation(701,false),  mapping::Segmentation(702,false),  mapping::Segmentation(703,false),
							     mapping::Segmentation(704,false),  mapping::Segmentation(705,false),  mapping::Segmentation(706,false),  mapping::Segmentation(707,false),
							     mapping::Segmentation(708,false),  mapping::Segmentation(709,false),  mapping::Segmentation(710,false),  mapping::Segmentation(711,false),
							     mapping::Segmentation(712,false),  mapping::Segmentation(713,false),  mapping::Segmentation(714,false),  mapping::Segmentation(715,false),
							     mapping::Segmentation(716,false),  mapping::Segmentation(717,false),  mapping::Segmentation(718,false),  mapping::Segmentation(719,false),
							     mapping::Segmentation(720,false),  mapping::Segmentation(721,false),  mapping::Segmentation(722,false),  mapping::Segmentation(723,false),
							     mapping::Segmentation(724,false),  mapping::Segmentation(725,false),  mapping::Segmentation(800,false),  mapping::Segmentation(801,false),
							     mapping::Segmentation(802,false),  mapping::Segmentation(803,false),  mapping::Segmentation(804,false),  mapping::Segmentation(805,false),
							     mapping::Segmentation(806,false),  mapping::Segmentation(807,false),  mapping::Segmentation(808,false),  mapping::Segmentation(809,false),
							     mapping::Segmentation(810,false),  mapping::Segmentation(811,false),  mapping::Segmentation(812,false),  mapping::Segmentation(813,false),
							     mapping::Segmentation(814,false),  mapping::Segmentation(815,false),  mapping::Segmentation(816,false),  mapping::Segmentation(817,false),
							     mapping::Segmentation(818,false),  mapping::Segmentation(819,false),  mapping::Segmentation(820,false),  mapping::Segmentation(821,false),
							     mapping::Segmentation(822,false),  mapping::Segmentation(823,false),  mapping::Segmentation(824,false),  mapping::Segmentation(825,false),
							     mapping::Segmentation(900,false),  mapping::Segmentation(901,false),  mapping::Segmentation(902,false),  mapping::Segmentation(903,false),
							     mapping::Segmentation(904,false),  mapping::Segmentation(905,false),  mapping::Segmentation(906,false),  mapping::Segmentation(907,false),
							     mapping::Segmentation(908,false),  mapping::Segmentation(909,false),  mapping::Segmentation(910,false),  mapping::Segmentation(911,false),
							     mapping::Segmentation(912,false),  mapping::Segmentation(913,false),  mapping::Segmentation(914,false),  mapping::Segmentation(915,false),
							     mapping::Segmentation(916,false),  mapping::Segmentation(917,false),  mapping::Segmentation(918,false),  mapping::Segmentation(919,false),
							     mapping::Segmentation(920,false),  mapping::Segmentation(921,false),  mapping::Segmentation(922,false),  mapping::Segmentation(923,false),
							     mapping::Segmentation(924,false),  mapping::Segmentation(925,false),  mapping::Segmentation(1000,false), mapping::Segmentation(1001,false),
							     mapping::Segmentation(1002,false), mapping::Segmentation(1003,false), mapping::Segmentation(1004,false), mapping::Segmentation(1005,false),
							     mapping::Segmentation(1006,false), mapping::Segmentation(1007,false), mapping::Segmentation(1008,false), mapping::Segmentation(1009,false),
							     mapping::Segmentation(1010,false), mapping::Segmentation(1011,false), mapping::Segmentation(1012,false), mapping::Segmentation(1013,false),
							     mapping::Segmentation(1014,false), mapping::Segmentation(1015,false), mapping::Segmentation(1016,false), mapping::Segmentation(1017,false),
							     mapping::Segmentation(1018,false), mapping::Segmentation(1019,false), mapping::Segmentation(1020,false), mapping::Segmentation(1021,false),
							     mapping::Segmentation(1022,false), mapping::Segmentation(1023,false), mapping::Segmentation(1024,false), mapping::Segmentation(1025,false)}
                                                          );


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

void MCHDigitizer::process(const std::vector<Hit>* hits, std::vector<Digit>* digits)
{
  // hits array of MCH hits for a given simulated event
  for (auto& hit : *hits) {
    //TODO: check if change for time structure
    processHit(hit, mEventTime);
   } // end loop over hits
  //TODO: merge (new member function, add charge) of digits that are on same pad:
  //things to think about in terms of time costly

    digits->clear();
    fillOutputContainer(*digits);

}

//______________________________________________________________________

Int_t MCHDigitizer::processHit(const Hit &hit,Double_t event_time)
{


  //hit position(cm)
  Float_t pos[3] = { hit.GetX(), hit.GetY(), hit.GetZ() };
  //convert energy to charge, float enough?
  Float_t charge = etocharge(hit.GetEnergyLoss());
  //time information
  Float_t time = hit.GetTime();//how to trace
  Int_t detID = hit.GetDetectorID();
  //get index for this detID
  Int_t indexID = mdetID.at(detID);
  //# digits for hit
  Int_t ndigits=0;
  
  Float_t anodpos = getAnod(pos[0],detID);

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
  Float_t fracplane = chargeCorr();//should become a function of anoddis
  Float_t chargebend= fracplane*charge;
  Float_t chargenon = charge/fracplane;
  //last line  from Aliroot, not understood why
  //since charge = charchbend+chargenon and not multiplication
  Float_t signal = 0.0;

  //borders of charge gen. 
  Double_t xMin = anodpos-mQspreadX*0.5;
  Double_t xMax = anodpos+mQspreadX*0.5;

  Double_t yMin = pos[1]-mQspreadY*0.5;
  Double_t yMax = pos[1]+mQspreadY*0.5;
  
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
  Int_t padidbend= mSegbend[indexID].findPadByPosition(anodpos,pos[1]);
  Int_t padidnon= mSegnon[indexID].findPadByPosition(anodpos,pos[1]);
  //correct coordinate system? how misalignment enters?
  /*mPadIDsbend = mSegbend.getPadUids(xMin,xMax,yMin,yMax);
    mPadIDsnon  = mSegnon.getPadUids(xMin,xMax,yMin,yMax);
  */
    /* for(auto & padidbend : mPadIDsbend){
    //retrieve coordinates for each pad*/
  xmin =  mSegbend[indexID].padPositionX(padidbend)-mSegbend[indexID].padSizeX(padidbend)*0.5;
  xmax =  mSegbend[indexID].padPositionX(padidbend)+mSegbend[indexID].padSizeX(padidbend)*0.5;
  ymin =  mSegbend[indexID].padPositionY(padidbend)-mSegbend[indexID].padSizeY(padidbend)*0.5;
  ymax =  mSegbend[indexID].padPositionY(padidbend)+mSegbend[indexID].padSizeY(padidbend)*0.5;
      
  // 1st step integrate induced charge for each pad
  signal = chargePad(anodpos,pos[1],xmin,xmax,ymin,ymax,detID,chargebend);
    if(signal>mChargeThreshold && signal<mChargeSat){
      //2nd condition in Aliroot said to be only for backward compatibility
      //to be seen...means that there is no digit, if signal above... strange!
      //2n step TODO: pad response function, electronic response
      signal = response(detID,signal);
      mDigits.emplace_back(padidbend,signal);//how trace time?
      ++ndigits;
    }
    /*}
	   for(auto & padidnon : mPadIDsnon){*/
    //retrieve coordinates for each pad
    xmin =  mSegnon[indexID].padPositionX(padidnon)-mSegnon[indexID].padSizeX(padidnon)*0.5;
    xmax =  mSegnon[indexID].padPositionX(padidnon)+mSegnon[indexID].padSizeX(padidnon)*0.5;
    ymin =  mSegnon[indexID].padPositionY(padidnon)-mSegnon[indexID].padSizeY(padidnon)*0.5;
    ymax =  mSegnon[indexID].padPositionY(padidnon)+mSegnon[indexID].padSizeY(padidnon)*0.5;
    
    //retrieve charge for given x,y with Mathieson
    signal = chargePad(anodpos,pos[1],xmin,xmax,ymin,ymax,detID,chargenon);
    if(signal>mChargeThreshold && signal<mChargeSat){
      signal = response(detID,signal);
      mDigits.emplace_back(padidnon,signal);//how is time propagated?
      ++ndigits;
    }
    /*}*/	
    
    
    return ndigits;
}
//_____________________________________________________________________
Float_t MCHDigitizer::etocharge(Float_t edepos){
  //Todo convert in charge in number of electrons
  //equivalent if IntPH in AliMUONResponseV0 in Aliroot
  //to be clarified:
  //1) why effective parameterisation with Log?
  //2) any will to provide random numbers
  //3) Float in aliroot, Double needed?
  //with central seed to be reproducible?
  //TODO: dependence on station
  //TODO: check slope meaning in thesis
  Int_t nel = Int_t(edepos*1.e9/27.4);
  Float_t charge=0;
  if (nel ==0) nel=1;
  for (Int_t i=1; i<=nel;i++) {
    Float_t arg=0.;
    while(!arg) arg = gRandom->Rndm();
    charge -= mChargeSlope*TMath::Log(arg);
    
  }
  return charge;
}
//_____________________________________________________________________
 std::vector<int> MCHDigitizer::getPadUid(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax, bool bend){
  //to be implemented?
  
  return mPadIDsbend;

}

//_____________________________________________________________________
Double_t MCHDigitizer::chargePad(Float_t x, Float_t y, Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax, Int_t detID, Float_t charge ){
  //see AliMUONResponseV0.cxx (inside DisIntegrate)
  // and AliMUONMathieson.cxx (IntXY)
  Int_t station = 0;
  if(detID>299) station = 1;//wrong numbers!
  //correct? should take info from segmentation
  // normalise w.r.t. Pitch
  xmin *= mInversePitch[station];
  xmax *= mInversePitch[station];
  ymin *= mInversePitch[station];
  ymax *= mInversePitch[station];
  // The Mathieson function
  Double_t ux1=mSqrtK3x[station]*TMath::TanH(mK2x[station]*xmin);
  Double_t ux2=mSqrtK3x[station]*TMath::TanH(mK2x[station]*xmax);
  
  Double_t uy1=mSqrtK3y[station]*TMath::TanH(mK2y[station]*ymin);
  Double_t uy2=mSqrtK3y[station]*TMath::TanH(mK2y[station]*ymax);
  
  return 4.*mK4x[station]*(TMath::ATan(ux2)-TMath::ATan(ux1))*
    mK4y[station]*(TMath::ATan(uy2)-TMath::ATan(uy1))*charge;
}
//______________________________________________________________________
Double_t MCHDigitizer::response(Float_t charge, Int_t detID){
  //to be done: calculate from induced charge signal
  return charge;
}
//______________________________________________________________________
Float_t MCHDigitizer::getAnod(Float_t x, Int_t detID){

  Float_t pitch = mInversePitch[1];
  if(detID<299) pitch = mInversePitch[0]; //guess for numbers!
  
  Int_t n = Int_t(x/pitch);
  Float_t wire = (x>0) ? n+0.5 : n-0.5;
  return pitch*wire;
}
//______________________________________________________________________
Float_t MCHDigitizer::chargeCorr(){
  //taken from AliMUONResponseV0
  //conceptually not at all understood why this should make sense
  //mChargeCorr not taken
  return TMath::Exp(gRandom->Gaus(0.0, mChargeCorr/2.0));
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

