/// \file Digitizer.h
/// \brief Task for ALICE TPC digitization
#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "DigitContainer.h"

#include "Rtypes.h"   // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h"  // for TObject


class TClonesArray;  // lines 13-13
// namespace AliceO2 { namespace TPC { class DigitContainer; } }  // lines 19-19
// namespace AliceO2 { namespace TPC { class UpgradeGeometryTGeo; } }  // lines 20-20

namespace AliceO2{

  namespace TPC {

    class DigitContainer;
//     class UpgradeGeometryTGeo;

    class Digitizer : public TObject {
    public:
      Digitizer();

      /// Destructor
      ~Digitizer();

      void Init();

      /// Steer conversion of points to digits
      /// @param points Container with ITS points
      /// @return digits container
      DigitContainer *Process(TClonesArray *points);
      void DriftElectrons(Float_t *xyz);
      void GEMAmplification(Float_t nele);

      void SetGainFactor(Double_t gain) { mGain = gain; }


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
