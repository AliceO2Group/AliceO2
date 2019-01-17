// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_MCHDIGITIZER_H_
#define O2_MCH_SIMULATION_MCHDIGITIZER_H_
 
#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Geometry.h"
#include "MCHSimulation/Hit.h"
#include "MCHSimulation/Response.h"
#include "MCHMappingInterface/Segmentation.h"

#include "TGeoManager.h"


namespace o2
{
namespace mch
{
  
class MCHDigitizer
{
 public:
  MCHDigitizer(Int_t mode = 0);
  
  ~MCHDigitizer() = default;

  void init();

    
  void setEventTime(double timeNS) { mEventTime = timeNS; }
  void setEventID(int eventID) { mEventID = eventID; }
  void setSrcID(int sID) { mSrcID = sID; }
  
  //this will process hits and fill the digit vector with digits which are finalized
  void process(const std::vector<Hit> hits, std::vector<Digit>& digits);

  void fillOutputContainer(std::vector<Digit>& digits);
  void flushOutputContainer(std::vector<Digit>& digits); // flush all residual buffered data

  void setContinuous(bool val) { mContinuous = val; }
  bool isContinuous() const { return mContinuous; }
  
 private:  
  double mEventTime;
  int mReadoutWindowCurrent{0};  
  int mEventID = 0;
  int mSrcID = 0;
  
  bool mContinuous = false; 

  //number of detector elements
  const static int mNdE = 156;
  // digit per pad
  std::vector<Digit> mDigits;
  //
  //  static std::map<int,int> mdetID;

  std::map<int,int> mdetID;
  
  std::vector<mapping::Segmentation> mSeg;
                                                
  // std::vector<mapping::Segmentation> mSegnon;

  Response mMuonresponse;

  const TGeoManager gMgr = TGeoManager("MCH-ONLY", "ALICE MCH Standalone Geometry");
  
  //proper parameter in aliroot in AliMUONResponseFactory.cxx
  //to be discussed n-sigma to be put, use detID to choose value?
  //anything in segmentation foreseen?
  //seem to be only two different values (st. 1 and st. 2-5)...overhead limited
  //any need for separate values as in old code? in principle not...I think
  
  int processHit(const Hit& hit, double event_time);
};

} // namoespace mch
} // namespace o2
#endif
