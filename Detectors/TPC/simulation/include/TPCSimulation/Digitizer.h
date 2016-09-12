/// \file Digitizer.h
/// \brief Task for ALICE TPC digitization
#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/HitContainer.h"
#include "TPCSimulation/PadHit.h"
#include "TPCSimulation/PadResponse.h"

#include "Rtypes.h"
#include "TObject.h"
#include "TF1.h"

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
      
      /// Drift of electrons in electric field taking into account diffusion
      /// @param *xyz Array with 3d position of the electrons
      /// @return Array with 3d position of the electrons after the drift taking into account diffusion
      void ElectronDrift(Float_t *xyz);
      
      /// Simulation of the GEM response
      /// @return Number of electrons after GEM amplification taking into account exponential fluctuations of the gain
      Float_t GEMAmplification();
      
      
      /// Simulation of the GEM response of a single GEM
      /// @param nEle Number of incoming electrons
      /// @param gain Gain of the single GEM, should go to OCDB
      /// @return Number of electrons after GEM amplification taking into account fluctuations of the gain according to a Polya distribution
      Int_t SingleGEMAmplification(Int_t nEle, Float_t gain);
      
      
      /// Pad Response
      /// @param xabs Position in x
      /// @param yabs Position in y
      /// @return Vector with PadResponse objects with pad and row position and the correponding fraction of the induced signal
      vector< PadResponse*> getPadResponse(Float_t xabs, Float_t yabs);
      
      
      /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      /// Conversion functions that at some point should go someplace else

      /// Conversion from a given number of electrons into ADC value
      /// @param nElectrons Number of electrons in time bin
      /// @return ADC value
      Int_t ADCvalue(Float_t nElectrons);
      
      /// Compute time bin from z position
      /// @param zPos z position of the charge
      /// @return Time bin of the charge
      const Int_t getTimeBin(Float_t zPos);
      
      /// Compute time bin from time
      /// @param time time of the charge
      /// @return Time bin of the charge
      const Int_t getTimeBinFromTime(Float_t time);
      
      /// Compute time from time bin
      /// @param timeBin time bin of the charge
      /// @return Time of the charge      
      const Float_t getTimeFromBin(Int_t timeBin);
      
      /// Compute time from z position
      /// @param zPos z position of the charge
      /// @return Time of the charge
      const Double_t getTime(Float_t zPos);
      
      /// Gamma4 shaping function
      /// @param time Time of the ADC value with respect to the first bin in the pulse
      /// @param startTime First bin in the pulse
      /// @param ADC ADC value of the corresponding time bin
      Double_t Gamma4(Double_t time, Double_t startTime, Double_t ADC);

    private:
      Digitizer(const Digitizer &);
      Digitizer &operator=(const Digitizer &);

      TF1                     *mPolya;
      DigitContainer          *mDigitContainer;
      HitContainer            *mHitContainer;
      std::vector < PadHit* > mPadHit;

      ClassDef(Digitizer, 1);
    };
}
}

#endif /* ALICEO2_TPC_Digitizer_H_ */
