/// \file Digitizer.h
/// \brief Definition of the ITS digitizer
#ifndef ALICEO2_ITS_DIGITIZER_H
#define ALICEO2_ITS_DIGITIZER_H

#include <vector>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject

#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSMFTSimulation/DigitContainer.h"
#include "ITSBase/GeometryTGeo.h"

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
      Digitizer(const Digitizer&) = delete;
      Digitizer& operator=(const Digitizer&) = delete;


      void init(Bool_t build = kTRUE);

      /// Steer conversion of points to digits
      /// @param points Container with ITS points
      /// @return digits container
      o2::ITSMFT::DigitContainer& process(TClonesArray* points);
      void process(TClonesArray* points, TClonesArray* digits);

    private:
      GeometryTGeo mGeometry;                     ///< ITS upgrade geometry
      std::vector<o2::ITSMFT::SimulationAlpide> mSimulations; ///< Array of chips response simulations
      o2::ITSMFT::DigitContainer mDigitContainer; ///< Internal digit storage

      ClassDefOverride(Digitizer, 2);
    };
  }
}

#endif /* ALICEO2_ITS_DIGITIZER_H */
