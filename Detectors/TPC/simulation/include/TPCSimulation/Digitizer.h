/// \file Digitizer.h
/// \brief Definition of the ALICE TPC digitizer
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Constants.h"

#include "TPCBase/Mapper.h"

#include <cmath>
#include <Vc/Vc>

using std::vector;

class TTree;
class TClonesArray;

namespace o2 {
namespace TPC {

class DigitContainer;

/// Debug output
typedef struct {
    float CRU;
    float time;
    float row;
    float pad;
    float nElectrons;
} GEMRESPONSE;
static GEMRESPONSE GEMresponse;

/// \class Digitizer
/// This is the digitizer for the ALICE GEM TPC.
/// It is the main class and steers all relevant physical processes for the signal formation in the detector.
/// -# Transformation of energy deposit of the incident particle to a number of primary electrons
/// -# Drift and diffusion of the primary electrons while moving in the active volume towards the readout chambers (ElectronTransport)
/// -# Amplification of the electrons in the stack of four GEM foils (GEMAmplification)
/// -# Induction of the signal on the pad plane, including a spread of the signal due to the pad response (PadResponse)
/// -# Shaping and further signal processing in the Front-End Cards (SampaProcessing)
/// The such created Digits and then sorted in an intermediate Container (DigitContainer) and after processing of the full event/drift time summed up
/// and sorted as Digits into a TClonesArray which is then passed further on

class Digitizer {
  public:

    /// Default constructor
    Digitizer();

    /// Destructor
    ~Digitizer();

    /// Initializer
    void init();

    /// Steer conversion of points to digits
    /// \param points Container with TPC points
    /// \return digits container
    DigitContainer *Process(TClonesArray *points);

    DigitContainer *getDigitContainer() const { return mDigitContainer; }

    /// Enable the debug output after application of the PRF
    /// Can be set via DigitizerTask::setDebugOutput("PRFdebug")
    static void setPRFDebug() { mDebugFlagPRF = true; }

    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Conversion functions that at some point should go someplace else

    /// Compute time bin from z position
    /// \param zPos z position of the charge
    /// \return Time bin of the charge
    static int getTimeBin(float zPos);

    /// Compute z position from time bin
    /// \param Time bin of the charge
    /// \param
    /// \return zPos z position of the charge
    static float getZfromTimeBin(float timeBin, Side s);

    /// Compute time bin from time
    /// \param time time of the charge
    /// \return Time bin of the charge
    static int getTimeBinFromTime(float time);

    /// Compute time from time bin
    /// \param timeBin time bin of the charge
    /// \return Time of the charge
    static float getTimeFromBin(int timeBin);

    /// Compute time from z position
    /// \param zPos z position of the charge
    /// \return Time of the charge
    static float getTime(float zPos);

    /// Compute the time of a given time bin
    /// \param time Time of the charge
    /// \return Time of the time bin of the charge
    static float getTimeBinTime(float time);

  private:
    Digitizer(const Digitizer &);
    Digitizer &operator=(const Digitizer &);

    DigitContainer          *mDigitContainer;   ///< Container for the Digits

    std::unique_ptr<TTree>  mDebugTreePRF;      ///< Output tree for the output after the PRF
    static bool             mDebugFlagPRF;      ///< Flag for debug output after the PRF

  ClassDefNV(Digitizer, 1);
};

// inline implementations
inline
int Digitizer::getTimeBin(float zPos)
{
  float timeBin = (TPCLENGTH-std::fabs(zPos))/(DRIFTV*ZBINWIDTH);
  return static_cast<int>(timeBin);
}

inline
float Digitizer::getZfromTimeBin(float timeBin, Side s)
{
  float zSign = (s==0) ? 1 : -1;
  float zAbs =  zSign * (TPCLENGTH- (timeBin*DRIFTV*ZBINWIDTH));
  return zAbs;
}

inline
int Digitizer::getTimeBinFromTime(float time)
{
  float timeBin = time / ZBINWIDTH;
  return static_cast<int>(timeBin);
}

inline
float Digitizer::getTimeFromBin(int timeBin)
{
  float time = static_cast<float>(timeBin)*ZBINWIDTH;
  return time;
}

inline
float Digitizer::getTime(float zPos)
{
  float time = (TPCLENGTH-std::fabs(zPos))/DRIFTV;
  return time;
}

inline
float Digitizer::getTimeBinTime(float time)
{
  int timeBin = getTimeBinFromTime(time);
  return getTimeFromBin(timeBin);

}

}
}

#endif // ALICEO2_TPC_Digitizer_H_
