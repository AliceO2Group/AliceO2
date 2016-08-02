/// \file Constants.h
/// \brief Constants for the MFT
/// \author bogdan.vulpescu@cern.ch - 01/08/2016

#ifndef ALICEO2_MFT_CONSTANTS_H_
#define ALICEO2_MFT_CONSTANTS_H_

#include "TObject.h"

namespace AliceO2 {
namespace MFT {

class Constants : public TObject {

public:

  static const Int_t sNofDisks = 5;             ///< \brief Number of Disks

protected:

  Constants() : TObject() {}
  virtual ~Constants(){}

  ClassDef(Constants, 1);

};

}
}

#endif
