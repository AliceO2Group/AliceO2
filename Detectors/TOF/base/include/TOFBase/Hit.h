// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_HIT_H
#define ALICEO2_TOF_HIT_H

#include "Rtypes.h"

namespace o2 {
  namespace TOF {
    
    /// \class Hit
    /// \brief TOF simulation hit information
    class Hit {
    public:
      
      
    private:
      Double32_t        mTempo;     ///temporary
      
      ClassDefNV(Hit, 1);
    };
    
  }
}

#endif
