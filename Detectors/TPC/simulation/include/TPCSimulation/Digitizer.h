/// \file Digitizer.h
/// \brief Task for ALICE TPC digitization
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Constants.h"

#include "TPCBase/RandomRing.h"
#include "TPCBase/Mapper.h"

#include "Rtypes.h"
#include "TObject.h"

#include <cmath>
#include <iostream>
#include <Vc/Vc>

using std::vector;


class TClonesArray;

namespace AliceO2{
namespace TPC{

class DigitContainer;

/// \class Digitizer
/// \brief Digitizer class for the TPC

class Digitizer : public TObject {
  public:

    /// Default constructor
    Digitizer();

    /// Destructor
    ~Digitizer();

    /// Initializer
    void init();

    /// Steer conversion of points to digits
    /// @param points Container with TPC points
    /// @return digits container
    DigitContainer *Process(TClonesArray *points);

    /// Pad Response
    /// @param xabs Position in x
    /// @param yabs Position in y
    /// @return Vector with PadResponse objects with pad and row position and the correponding fraction of the induced signal
    void getPadResponse(Float_t xabs, Float_t yabs, vector<PadResponse> &);

    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Conversion functions that at some point should go someplace else

    /// Compute time bin from z position
    /// @param zPos z position of the charge
    /// @return Time bin of the charge
    Int_t getTimeBin(Float_t zPos) const;

    /// Compute time bin from time
    /// @param time time of the charge
    /// @return Time bin of the charge
    Int_t getTimeBinFromTime(Float_t time) const;

    /// Compute time from time bin
    /// @param timeBin time bin of the charge
    /// @return Time of the charge      
    Float_t getTimeFromBin(Int_t timeBin) const;

    /// Compute time from z position
    /// @param zPos z position of the charge
    /// @return Time of the charge
    Float_t getTime(Float_t zPos) const;


  private:
    Digitizer(const Digitizer &);
    Digitizer &operator=(const Digitizer &);

    DigitContainer          *mDigitContainer;   ///< Container for the Digits      

  ClassDef(Digitizer, 1);
};

// inline implementations
inline
Int_t Digitizer::getTimeBin(Float_t zPos) const 
{
  Float_t timeBin = (TPCLENGTH-std::fabs(zPos))/(DRIFTV*ZBINWIDTH);
  return static_cast<int>(timeBin);
}

inline
Int_t Digitizer::getTimeBinFromTime(Float_t time) const 
{
  Float_t timeBin = time / ZBINWIDTH;
  return static_cast<int>(timeBin);
}

inline
Float_t Digitizer::getTimeFromBin(Int_t timeBin) const 
{
  Float_t time = static_cast<float>(timeBin)*ZBINWIDTH;
  return time;
}

inline
Float_t Digitizer::getTime(Float_t zPos) const 
{
  Float_t time = (TPCLENGTH-std::fabs(zPos))/DRIFTV;
  return time;
}

inline
void Digitizer::getPadResponse(Float_t xabs, Float_t yabs, std::vector<PadResponse> &response)
{
  response.resize(0);
  /// @todo include actual response, this is now only for a signal on the central pad (0, 0) with weight 1.
  response.emplace_back(0, 0, 1);
}

}
}

#endif // ALICEO2_TPC_Digitizer_H_
