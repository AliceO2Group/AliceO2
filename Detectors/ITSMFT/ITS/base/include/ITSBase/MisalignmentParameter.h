/// \file MisalignmentParameter.h
/// \brief Definition of the MisalignmentParameter class

#ifndef ALICEO2_ITS_MISALIGNMENTPARAMETER_H_
#define ALICEO2_ITS_MISALIGNMENTPARAMETER_H_

#include "FairParGenericSet.h" // for FairParGenericSet

#include "Rtypes.h" // for ClassDef

#include "TArrayD.h" // for TArrayD

class FairParamList;

namespace o2
{
namespace ITS
{
class MisalignmentParameter : public FairParGenericSet
{
 public:
  MisalignmentParameter(const char* name = "MisalignmentParameter",
                        const char* title = "Misalignment parameter for AliceO2ITSHitProducerIdealMisallign Parameters",
                        const char* context = "TestDefaultContext");

  ~MisalignmentParameter() override;

  void Clear();

  void putParams(FairParamList*) override;

  Bool_t getParams(FairParamList*) override;

  TArrayD getShiftX() { return mShiftX; }
  TArrayD getShiftY() { return mShiftY; }
  TArrayD getShiftZ() { return mShiftZ; }
  TArrayD getRotX() { return mRotX; }
  TArrayD getRotY() { return mRotY; }
  TArrayD getRotZ() { return mRotZ; }
  Int_t getNumberOfDetectors() { return mNumberOfDetectors; }
 private:
  TArrayD mShiftX;          ///< Array to hold the misalignment in x-direction
  TArrayD mShiftY;          ///< Array to hold the misalignment in y-direction
  TArrayD mShiftZ;          ///< Array to hold the misalignment in z-direction
  TArrayD mRotX;            ///< Array to hold the rotation in x-direction
  TArrayD mRotY;            ///< Array to hold the rotation in y-direction
  TArrayD mRotZ;            ///< Array to hold the rotation in z-direction
  Int_t mNumberOfDetectors; ///< Total number of detectors

  MisalignmentParameter(const MisalignmentParameter&);

  MisalignmentParameter& operator=(const MisalignmentParameter&);

  ClassDefOverride(MisalignmentParameter, 1)
};
}
}

#endif
