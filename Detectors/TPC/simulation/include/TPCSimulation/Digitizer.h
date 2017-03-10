/// \file Digitizer.h
/// \brief Task for ALICE TPC digitization
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Constants.h"

#include "TPCBase/Mapper.h"

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
    void getPadResponse(float xabs, float yabs, vector<PadResponse> &);

    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Conversion functions that at some point should go someplace else

    /// Compute time bin from z position
    /// @param zPos z position of the charge
    /// @return Time bin of the charge
    int getTimeBin(float zPos) const;

    /// Compute time bin from time
    /// @param time time of the charge
    /// @return Time bin of the charge
    int getTimeBinFromTime(float time) const;

    /// Compute time from time bin
    /// @param timeBin time bin of the charge
    /// @return Time of the charge      
    float getTimeFromBin(int timeBin) const;

    /// Compute time from z position
    /// @param zPos z position of the charge
    /// @return Time of the charge
    float getTime(float zPos) const;


  private:
    Digitizer(const Digitizer &);
    Digitizer &operator=(const Digitizer &);

    DigitContainer          *mDigitContainer;   ///< Container for the Digits      

  ClassDef(Digitizer, 1);
};

// inline implementations
inline
int Digitizer::getTimeBin(float zPos) const
{
  float timeBin = (TPCLENGTH-std::fabs(zPos))/(DRIFTV*ZBINWIDTH);
  return static_cast<int>(timeBin);
}

inline
int Digitizer::getTimeBinFromTime(float time) const
{
  float timeBin = time / ZBINWIDTH;
  return static_cast<int>(timeBin);
}

inline
float Digitizer::getTimeFromBin(int timeBin) const
{
  float time = static_cast<float>(timeBin)*ZBINWIDTH;
  return time;
}

inline
float Digitizer::getTime(float zPos) const
{
  float time = (TPCLENGTH-std::fabs(zPos))/DRIFTV;
  return time;
}

}
}

#endif // ALICEO2_TPC_Digitizer_H_
