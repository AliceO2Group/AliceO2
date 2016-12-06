/// \file Digitizer.h
/// \brief Task for ALICE ITS digitization
#ifndef ALICEO2_ITS_Digitizer_H_
#define ALICEO2_ITS_Digitizer_H_

#include <vector>

#include "Rtypes.h"   // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h"  // for TObject

#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/DigitContainer.h"

class TClonesArray;

namespace AliceO2 {

namespace ITS {

class Chip;
class SimulationAlpide;

class Digitizer : public TObject
{
  public:
    Digitizer();
   ~Digitizer();

    /// Steer conversion of points to digits
    /// @param points Container with ITS points
    /// @return digits container
    DigitContainer &Process(TClonesArray *points);
    void Process(TClonesArray *points, TClonesArray *digits);

  private:
    Digitizer(const Digitizer &);
    Digitizer &operator=(const Digitizer &);

    GeometryTGeo fGeometry;            ///< ITS upgrade geometry
    Int_t fNChips;                     // Number of chips
    std::vector<Chip> fChips;          // Array of chips
    std::vector<SimulationAlpide> fSimulations;// Array of chips response simulations
    DigitContainer fDigitContainer;           ///< Internal digit storage

  ClassDef(Digitizer, 2);
};
}
}

#endif /* ALICEO2_ITS_Digitizer_H_ */
