#ifndef ALICEO2_MFT_EVENTHEADER_H_
#define ALICEO2_MFT_EVENTHEADER_H_

#include "FairEventHeader.h"

#ifndef __CINT__
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#endif

namespace AliceO2 {
namespace MFT {

class EventHeader : public FairEventHeader
{

 public:
  
  EventHeader();
  virtual ~EventHeader();

  void  SetPartNo(Int_t ipart) { fPartNo = ipart;}
  Int_t GetPartNo()            { return fPartNo; }
  
  template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/)
    {
      ar& boost::serialization::base_object<FairEventHeader>(*this);
    }
  
 private:

  Int_t fPartNo;
  
#ifndef __CINT__ // for BOOST serialization
  friend class boost::serialization::access;
#endif // for BOOST serialization
  
  ClassDef(EventHeader, 1);

};

}
}

#endif
