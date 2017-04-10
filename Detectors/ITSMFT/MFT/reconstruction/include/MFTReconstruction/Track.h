/// \file Track.h
/// \brief Simple track obtained from hits
/// \author bogdan.vulpescu@cern.ch 
/// \date 11/10/2016

#ifndef ALICEO2_MFT_TRACK_H_
#define ALICEO2_MFT_TRACK_H_

#include "FairTrackParam.h"

namespace o2 {
namespace MFT {

class Track : public FairTrackParam
{

 public:

  Track();
  ~Track() override;

  Track(const Track& track);

 private:

  Track& operator=(const Track& track);

  ClassDefOverride(Track,1);

};

}
}

#endif
