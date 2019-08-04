// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CDB_RUNRANGE_H_
#define ALICEO2_CDB_RUNRANGE_H_

//  defines the run validity range of the object:		   //
//  [mFirstRun, mLastRun] 					   //
#include <TObject.h> // for TObject
#include "Rtypes.h"  // for Int_t, Bool_t, IdRunRange::Class, ClassDef, etc

namespace o2
{
namespace ccdb
{
class IdRunRange : public TObject
{

 public:
  IdRunRange();

  IdRunRange(Int_t firstRun, Int_t lastRun);

  ~IdRunRange() override;

  Int_t getFirstRun() const
  {
    return mFirstRun;
  };

  Int_t getLastRun() const
  {
    return mLastRun;
  };

  void setFirstRun(Int_t firstRun)
  {
    mFirstRun = firstRun;
  };

  void setLastRun(Int_t lastRun)
  {
    mLastRun = lastRun;
  };

  void setIdRunRange(Int_t firstRun, Int_t lastRun)
  {
    mFirstRun = firstRun;
    mLastRun = lastRun;
  };

  Bool_t isValid() const;

  Bool_t isAnyRange() const
  {
    return mFirstRun < 0 && mLastRun < 0;
  };

  Bool_t isOverlappingWith(const IdRunRange& other) const;

  Bool_t isSupersetOf(const IdRunRange& other) const;

  virtual Bool_t isEqual(const TObject* obj) const;

  static Int_t Infinity()
  {
    return sInfinity;
  }

 private:
  Int_t mFirstRun; // first valid run
  Int_t mLastRun;  // last valid run

  static const Int_t sInfinity = 999999999; //! Flag for "infinity"

  ClassDefOverride(IdRunRange, 1);
};
} // namespace ccdb
} // namespace o2
#endif
