/// \file Digitizer.h
/// \brief Definition of the ITS digitizer
#ifndef ALICEO2_ITS_DIGITIZER_H
#define ALICEO2_ITS_DIGITIZER_H

#include <vector>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject

#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/DigitContainer.h"

class TClonesArray;

namespace AliceO2
{
  namespace ITS
  {
    class Chip;
    class SimulationAlpide;

    class Digitizer : public TObject
    {
    public:
      Digitizer();
      ~Digitizer();

      void init(Bool_t build = kTRUE);

      /// Steer conversion of points to digits
      /// @param points Container with ITS points
      /// @return digits container
      DigitContainer& process(TClonesArray* points);
      void process(TClonesArray* points, TClonesArray* digits);

    private:
      Digitizer(const Digitizer&);
      Digitizer& operator=(const Digitizer&);

      GeometryTGeo mGeometry;                     ///< ITS upgrade geometry
      Int_t mNumOfChips;                          ///< Number of chips
      std::vector<Chip> mChips;                   ///< Array of chips
      std::vector<SimulationAlpide> mSimulations; ///< Array of chips response simulations
      DigitContainer mDigitContainer;             ///< Internal digit storage

      ClassDef(Digitizer, 2);
    };
  }
}

#endif /* ALICEO2_ITS_DIGITIZER_H */
