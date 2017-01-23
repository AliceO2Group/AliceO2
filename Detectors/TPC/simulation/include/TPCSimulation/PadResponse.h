/// \file PadResponse.h
/// \brief Pad Response object
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_PadResponse_H_
#define ALICEO2_TPC_PadResponse_H_

#include "Rtypes.h"

namespace AliceO2 {
namespace TPC {
    
/// \class PadResponse
/// \brief Object for the pad hits due to the PRF
    
class PadResponse {
  public:
      
    /// Default constructor
    PadResponse();
      
    /// Constructor
    /// @param pad Pad of the signal
    /// @param row Row of the signal
    /// @param weight Weight of the signal
    PadResponse(Int_t pad, Int_t row, Float_t weight);
      
    /// Destructor
    virtual ~PadResponse();
      
    /// Get the pad
    /// @return Pad
    Float_t getPad() const { return mPad; }
      
    /// Get the row
    /// @return Row
    Float_t getRow() const { return mRow; }
      
    /// Get the weighted signal
    /// @return Weighted signal
    Float_t getWeight() const { return mWeight; }
      
  private:
    UChar_t           mPad;
    UChar_t           mRow;
    Float_t           mWeight;
};

}
}

#endif // ALICEO2_TPC_PadResponse_H_
