/// \file Digitizer.h
/// \brief Task for ALICE TPC digitization
#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCsimulation/DigitContainer.h"

#include "Rtypes.h"
#include "TObject.h"


class TClonesArray;

namespace AliceO2{

  namespace TPC {

    class DigitContainer;

    class Digitizer : public TObject {
    public:
      Digitizer();

      /// Destructor
      ~Digitizer();

      void init();

      /// Steer conversion of points to digits
      /// @param points Container with TPC points
      /// @return digits container
      DigitContainer *Process(TClonesArray *points);
      Int_t getADCvalue(Float_t nElectrons);
      void getElectronDrift(Float_t *xyz);
      Float_t getGEMAmplification();
      const Int_t getTimeBin(Float_t zPos);



      Double_t Gamma4(Double_t x, Double_t p0, Double_t p1);

      void setGainFactor(Double_t gain) { mGain = gain; }


    private:
      Digitizer(const Digitizer &);
      Digitizer &operator=(const Digitizer &);

      DigitContainer          *mDigitContainer;           ///< Internal digit storage
      Double_t                mGain;                      ///< pad gain factor (global for the moment)

      ClassDef(Digitizer, 1);
    };
}
}

#endif /* ALICEO2_ITS_Digitizer_H_ */
