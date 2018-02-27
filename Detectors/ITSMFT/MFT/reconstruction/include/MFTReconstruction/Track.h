// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.h
/// \brief Definition of the MFT track
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 8, 2018

#ifndef ALICEO2_ITS_TRACK_H
#define ALICEO2_ITS_TRACK_H

#include <vector>

#include "ReconstructionDataFormats/Track.h"

namespace o2
{

namespace ITSMFT {
class Cluster;
}

namespace MFT
{
class Track : public o2::track::TrackParCov
{
  using Cluster = o2::ITSMFT::Cluster;

  public:
    using o2::track::TrackParCov::TrackParCov;
    
    Track() = default;
    Track(const Track& t) = default;
    Track& operator=(const Track& tr) = default;
    ~Track() = default;

    // These functions must be provided
    //Bool_t propagate(Float_t alpha, Float_t x, Float_t bz);
    //Bool_t update(const Cluster& c, Float_t chi2, Int_t idx);

  private:

    ClassDef(Track, 1)
};
 
}
}

#endif
