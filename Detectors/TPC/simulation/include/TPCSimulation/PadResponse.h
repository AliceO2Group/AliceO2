/// \file PadResponse.h
/// \brief Pad Response object
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_PadResponse_H_
#define ALICEO2_TPC_PadResponse_H_

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
    PadResponse(int pad, int row, float weight);
      
    /// Destructor
    virtual ~PadResponse();
      
    /// Get the pad
    /// @return Pad
    float getPad() const { return mPad; }
      
    /// Get the row
    /// @return Row
    float getRow() const { return mRow; }
      
    /// Get the weighted signal
    /// @return Weighted signal
    float getWeight() const { return mWeight; }
      
  private:
    float             mWeight;  ///< Weight of the signal on the specific pad, wrt to the central pad above which the electron arrives
    unsigned char     mPad;     ///< Pad with signal, wrt to the central pad above which the electron arrives
    unsigned char     mRow;     ///< Row with signal, wrt to the central pad above which the electron arrives
};

}
}

#endif // ALICEO2_TPC_PadResponse_H_
