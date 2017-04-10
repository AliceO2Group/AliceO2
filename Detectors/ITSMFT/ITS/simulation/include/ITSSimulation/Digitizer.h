/// \file Digitizer.h
/// \brief Definition of the ITS digitizer
#ifndef ALICEO2_ITS_DIGITIZER_H
#define ALICEO2_ITS_DIGITIZER_H

#include <vector>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject

#include "ITSMFTSimulation/Chip.h"
#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/DigitContainer.h"

class TClonesArray;

namespace o2
{
  namespace ITS
  {
    class Digitizer : public TObject
    {
    public:
      Digitizer();
      ~Digitizer() override;

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
      std::vector<o2::ITSMFT::Chip> mChips;  ///< Array of chips
      std::vector<o2::ITSMFT::SimulationAlpide> mSimulations; ///< Array of chips response simulations
      DigitContainer mDigitContainer;             ///< Internal digit storage

      ClassDefOverride(Digitizer, 2);
    };
  }
}

#endif /* ALICEO2_ITS_DIGITIZER_H */
