// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MFT_EVENTHEADER_H_
#define ALICEO2_MFT_EVENTHEADER_H_

#include "FairEventHeader.h"

#ifndef __CINT__
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#endif

namespace o2 {
namespace MFT {

class EventHeader : public FairEventHeader
{

 public:
  
  EventHeader();
  ~EventHeader() override;

  void  setPartNo(Int_t ipart) { mPartNo = ipart;}
  Int_t getPartNo()            { return mPartNo; }
  
  template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/)
    {
      ar& boost::serialization::base_object<FairEventHeader>(*this);
    }
  
 private:

  Int_t mPartNo;
  
#ifndef __CINT__ // for BOOST serialization
  friend class boost::serialization::access;
#endif // for BOOST serialization
  
  ClassDefOverride(EventHeader, 1);

};

}
}

#endif
