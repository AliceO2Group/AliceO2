/// \file Digitizer.h
/// \brief Task for ALICE ITS digitization
#ifndef ALICEO2_ITS_Digitizer_H_
#define ALICEO2_ITS_Digitizer_H_

#include "Rtypes.h"   // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h"  // for TObject
#include "ITSSimulation/Chip.h"

class TClonesArray;

namespace AliceO2 {

namespace ITS {

class DigitContainer;
class UpgradeGeometryTGeo;
class UpgradeSegmentationPixel;

class Digitizer : public TObject
{
  public:
    Digitizer();
   ~Digitizer();

    /// Steer conversion of points to digits
    /// @param points Container with ITS points
    /// @return digits container
    DigitContainer *Process(TClonesArray *points);

  private:
    Digitizer(const Digitizer &);
    Digitizer &operator=(const Digitizer &);

    DigitContainer *fDigitContainer;           ///< Internal digit storage
    UpgradeGeometryTGeo *fGeometry;            ///< ITS upgrade geometry
    UpgradeSegmentationPixel *fSeg;            ///< Pixelchip segmentation

  ClassDef(Digitizer, 2);
};
}
}

#endif /* ALICEO2_ITS_Digitizer_H_ */
