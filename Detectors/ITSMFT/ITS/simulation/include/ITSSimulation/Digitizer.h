/// \file Digitizer.h
/// \brief Task for ALICE ITS digitization
#ifndef ALICEO2_ITS_Digitizer_H_
#define ALICEO2_ITS_Digitizer_H_

#include "Rtypes.h"   // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h"  // for TObject

class TClonesArray;

namespace AliceO2 {

namespace ITS {

class Chip;
class SimulationAlpide;
class DigitContainer;
class GeometryTGeo;
class SegmentationPixel;

class Digitizer : public TObject
{
  public:
    Digitizer();
   ~Digitizer();

    /// Steer conversion of points to digits
    /// @param points Container with ITS points
    /// @return digits container
    DigitContainer *Process(TClonesArray *points);
    void Process(TClonesArray *points, TClonesArray *digits);

  private:
    Digitizer(const Digitizer &);
    Digitizer &operator=(const Digitizer &);

    Chip *fChips; // Array of chips
    SimulationAlpide *fSimulations; // Array of chips response simulations
    DigitContainer *fDigitContainer;           ///< Internal digit storage
    GeometryTGeo *fGeometry;            ///< ITS upgrade geometry

  ClassDef(Digitizer, 2);
};
}
}

#endif /* ALICEO2_ITS_Digitizer_H_ */
