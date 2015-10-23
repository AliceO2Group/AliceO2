/// \file Digitizer.h
/// \brief Task for ALICE ITS digitization
#ifndef ALICEO2_ITS_Digitizer_H_
#define ALICEO2_ITS_Digitizer_H_

#include <vector>

#include "TObject.h"
#include "Rtypes.h"

#include "itsmft/its/Chip.h"

class TClonesArray;

namespace AliceO2{

  namespace ITS {
    
    class DigitContainer;
    class UpgradeGeometryTGeo;
    
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
        
      void SetGainFactor(Double_t gain) { fGain = gain; }
      
      void ClearChips();
            
    private:
      Digitizer(const Digitizer &);
      Digitizer &operator=(const Digitizer &);
      
      std::vector<Chip>       fChipContainer;             ///< Container for ITS chip
      
      DigitContainer          *fDigitContainer;           ///< Internal digit storage
      UpgradeGeometryTGeo     *fGeometry;                 ///< ITS upgrade geometry
      
      Double_t                fGain;                      ///< pad gain factor (global for the moment)
        
      ClassDef(Digitizer, 1);
    };
}
}

#endif /* ALICEO2_ITS_Digitizer_H_ */
