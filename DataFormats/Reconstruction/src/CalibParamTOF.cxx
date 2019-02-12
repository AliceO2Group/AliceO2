// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibParamTOF.cxx
/// \brief Class to store the output of the matching to TOF for calibration

#include <algorithm>
#include "ReconstructionDataFormats/CalibParamTOF.h"

using namespace o2::dataformats;

//ClassImp(o2::dataformats::CalibParamTOF);

CalibParamTOF::CalibParamTOF(){
  for(int i=0;i < NSECTORS;i++){
    for(int j=0;j < NCHANNELXSECTORSECTOR;j++){
      mChannelStart[i][j] = -1;
    }
  }
}
//______________________________________________

float CalibParamTOF::evalTimeSlewing(int channel,float tot) const {
  int sector = channel/NCHANNELXSECTORSECTOR;
  channel = channel%NCHANNELXSECTORSECTOR;
  
  if(sector >= NSECTORS) return 0.; // something went wrong!
  
  int n=mChannelStart[sector][channel];
  if(n < 0) return 0.;
  
  int nstop=mTimeSlewing[sector].size();
  if(channel < NCHANNELXSECTORSECTOR-1) nstop=mChannelStart[sector][channel+1];

  if(n >= nstop) return 0.; // something went wrong!
 
  while(n < nstop && tot < mTimeSlewing[sector][n].first) n++;
  n--;

  if(n < 0){ // tot is lower than the first available value
    return 0;
  }

  if(n == nstop-1) return mTimeSlewing[sector][n].second; // use the last value stored for that channel

  float w1 = tot - mTimeSlewing[sector][n].first;
  float w2 = mTimeSlewing[sector][n+1].first - tot;

  return (mTimeSlewing[sector][n].second*w2 + mTimeSlewing[sector][n+1].second*w1)/(mTimeSlewing[sector][n+1].first - mTimeSlewing[sector][n].first);
}
//______________________________________________

float CalibParamTOF::getLHCphase(int timestamp) const{
  int n=0;
  while(n < mLHCphase.size() && mLHCphase[n].first < timestamp) n++;
  n--;
  
  if(n < 0){ // timestamp is before of the first available value
    return 0;
  }
  return mLHCphase[n].second;
}
//______________________________________________

void CalibParamTOF::addLHCphase(int timestamp, float phaseLHC){
  // optimized if timestamp are given in increasing order
  int n = mLHCphase.size();
  mLHCphase.emplace_back(timestamp,phaseLHC);
  
  if(n && mLHCphase[n+1].first < mLHCphase[n].first){ // in the wrong order sort!
    std::sort( mLHCphase.begin( ), mLHCphase.end( ), [ ]( const auto& lhs, const auto& rhs ){
	return lhs.first < rhs.first;
      });
  }
}
//______________________________________________
void CalibParamTOF::addTimeSlewingInfo(int channel, float tot, float time){
  // WE ARE ASSUMING THAT:
  // channels have to be filled in increasing order (within the sector)
  // tots have to be filled in increasing order (within the channel)
  int sector = channel/NCHANNELXSECTORSECTOR;
  channel = channel%NCHANNELXSECTORSECTOR;
  
  if(sector >= NSECTORS) return; // something went wrong!
  
  int currentch = channel;
  while(mChannelStart[sector][currentch] == -1 && currentch > -1){
    // set also all the previous ones which were not filled
    mChannelStart[sector][currentch] = mTimeSlewing[sector].size();
    currentch--;
  }

  mTimeSlewing[sector].emplace_back(tot,time);
}
//______________________________________________
