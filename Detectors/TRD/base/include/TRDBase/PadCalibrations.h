
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_PADCALIBRATIONS_H
#define O2_TRD_PADCALIBRATIONS_H

////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//  TRD calibration class for parameters which are stored pad wise (1.2M entries) //
//  2019 - Ported from various bits of AliRoot (SHTM)                             //
//  Similar CalPad                                                                //
////////////////////////////////////////////////////////////////////////////////////

#include <array>

#include "TRDBase/PadParameters.h"
#include "fairlogger/Logger.h"

class TRDGeometry;

namespace o2
{
namespace trd
{

template <class T>
class PadCalibrations
{
 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540 };
  enum { kVdrift = 0,
         kGainFactor = 1,
         kT0 = 2,
         kExB = 3,
         kLocalGainFactor = 4 };
  PadCalibrations();
  ~PadCalibrations() = default;
  //
  int getNChannels(int roc) { return mreadOutChamber[roc].getNChannels(); }
  PadParameters<T>& getChamberPads(int roc) { return mreadOutChamber[roc]; }
  T getValue(int roc, int col, int row) { return mreadOutChamber[roc].getValue(col, row); }
  T getPadValue(int roc, int col, int row) { return mreadOutChamber[roc].getValue(col,row);}
  void setPadValue(int roc, int col, int row, T value) { mreadOutChamber[roc].setValue(col,row, value);}
  
  void init();
  void dumpAllNonZeroValues(); // helps for debugging.
 protected:
  std::array<PadParameters<T>, kNdet> mreadOutChamber;
};

template <class T>
PadCalibrations<T>::PadCalibrations()
{
  //
  // TRDCalPadStatus constructor
  //
  //TRDGeometry fgeom;
  int chamberindex=0;  
 for(auto& roc : mreadOutChamber) {   // Range-for!
      LOG(debug3) << "initialising readout chamber "<< chamberindex;
      roc.init(chamberindex++);

  }
}

template <class T> 
void PadCalibrations<T>::init()
{
  //
  // TRDCalPadStatus constructor
  //
  int chamberindex=0;  
  for(auto& roc : mreadOutChamber) {   // Range-for!
      LOG(debug3) << "initialising readout chamber "<< chamberindex;
      roc.init(chamberindex++);
  }
}


template <class T>
void PadCalibrations<T>::dumpAllNonZeroValues()
{
  //
  // TRDCalPadStatus constructor
  //
  int chamberindex=0;  
  for(auto& roc : mreadOutChamber) {   // Range-for!
      roc.dumpNonZeroValues(chamberindex);
      chamberindex++;
  }
}


} // namespace trd
} // namespace o2
#endif
