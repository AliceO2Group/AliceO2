/// \file LadderSegmentation.h
/// \brief Description of the virtual segmentation of a ladder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_LADDERSEGMENTATION_H_
#define ALICEO2_MFT_LADDERSEGMENTATION_H_

#include "TClonesArray.h"

#include "MFTBase/VSegmentation.h"
#include "MFTBase/ChipSegmentation.h"

namespace AliceO2 {
namespace MFT {

class LadderSegmentation : public VSegmentation {

public:

  LadderSegmentation();
  LadderSegmentation(UInt_t uniqueID);
  LadderSegmentation(const LadderSegmentation& ladder);

  virtual ~LadderSegmentation() { if(fChips){fChips->Delete(); delete fChips; fChips=NULL;} }
  virtual void Print(Option_t* opt="");
  virtual void Clear(const Option_t* /*opt*/) { if(fChips){fChips->Clear();} }
  
  ChipSegmentation* GetSensor(Int_t sensor) const ;

  void CreateSensors();
  
  /// \brief Returns number of Sensor on the ladder
  Int_t GetNSensors() const { return fNSensors; };
  /// \brief Set number of Sensor on the ladder
  void SetNSensors(Int_t val) {fNSensors = val;};
  
  ChipSegmentation* GetChip(Int_t chipNumber) const {return GetSensor(chipNumber);};

private:
  
  Int_t fNSensors;      ///< \brief Number of Sensors holded by the ladder
  TClonesArray *fChips; ///< \brief Array of pointer to ChipSegmentation

  /// \cond CLASSIMP
  ClassDef(LadderSegmentation, 1);
  /// \endcond

};

}
}
	
#endif

