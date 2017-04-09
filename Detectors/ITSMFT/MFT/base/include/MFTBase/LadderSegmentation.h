/// \file LadderSegmentation.h
/// \brief Description of the virtual segmentation of a ladder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_LADDERSEGMENTATION_H_
#define ALICEO2_MFT_LADDERSEGMENTATION_H_

#include "TClonesArray.h"

#include "MFTBase/VSegmentation.h"
#include "MFTBase/ChipSegmentation.h"

namespace o2 {
namespace MFT {

class LadderSegmentation : public VSegmentation {

public:

  LadderSegmentation();
  LadderSegmentation(UInt_t uniqueID);
  LadderSegmentation(const LadderSegmentation& ladder);

  virtual ~LadderSegmentation() { if(mChips){mChips->Delete(); delete mChips; mChips=nullptr;} }
  virtual void Print(Option_t* opt="");
  virtual void Clear(const Option_t* /*opt*/) { if(mChips){mChips->Clear();} }
  
  ChipSegmentation* GetSensor(Int_t sensor) const ;

  void CreateSensors();
  
  /// \brief Returns number of Sensor on the ladder
  Int_t GetNSensors() const { return mNSensors; };
  /// \brief Set number of Sensor on the ladder
  void SetNSensors(Int_t val) {mNSensors = val;};
  
  ChipSegmentation* GetChip(Int_t chipNumber) const {return GetSensor(chipNumber);};

private:
  
  Int_t mNSensors;      ///< \brief Number of Sensors holded by the ladder
  TClonesArray *mChips; ///< \brief Array of pointer to ChipSegmentation

  ClassDef(LadderSegmentation, 1);

};

}
}
	
#endif

