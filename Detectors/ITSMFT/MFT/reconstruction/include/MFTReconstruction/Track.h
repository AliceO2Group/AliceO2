// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
