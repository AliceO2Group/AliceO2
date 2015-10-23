/// \file Digitizer.h
/// \brief Task for ALICE ITS digitization
#ifndef ALICEO2_ITS_Digitizer_H_
#define ALICEO2_ITS_Digitizer_H_

#include "Rtypes.h"   // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h"  // for TObject
#include "itsmft/its/Chip.h"

class TClonesArray;  // lines 13-13
namespace AliceO2 { namespace ITS { class DigitContainer; } }  // lines 19-19
namespace AliceO2 { namespace ITS { class UpgradeGeometryTGeo; } }  // lines 20-20

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
