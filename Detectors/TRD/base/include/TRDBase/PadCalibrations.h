// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
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
  // various functions that I *think* should be moved to higher classes, I do not understand the getmeanrms etc. as it is used in MakeHisto
  int getNChannels(int roc) { return mreadOutChamber[roc].getNChannels(); }
  PadParameters<T>& getChamberPads(int roc) { return mreadOutChamber[roc]; }
  T getValue(int roc, int col, int row) { return ((PadParameters<T>)mreadOutChamber[roc]).getValue(col, row); }
  T getPadValue(int roc, int col, int row) { return ((PadParameters<T>)mreadOutChamber[roc]).getValue(col,row);}
  void setPadValue(int roc, int col, int row, T value) { cout << "--------------------------- firt pointer is : " << &mreadOutChamber << endl; /* cout << "now in setpad "  << endl; cout << "setting value for roc : " << roc << " col : " << col << " and row : " << row << " with value of :" << value << "with current value of " << getPadValue(roc,col,row) << endl; */ cout << "roc of : " << roc << endl; cout << "col::row : " << col <<"::" << row << endl;((PadParameters<T>)mreadOutChamber[roc]).setValue(col,row, value);}
  void debug(){int count=0;for (auto & roc: mreadOutChamber) {cout << "roc : " << count++  <<" with base ptr : " << &(roc)<< endl; roc.debug();}} 
  void init();
 protected:
  std::array<PadParameters<T>, kNdet> mreadOutChamber{24};
};

template <class T>
PadCalibrations<T>::PadCalibrations()
{
  //
  // TRDCalPadStatus constructor
  //
  //TRDGeometry fgeom;
  cout << " first pad is as : " << &mreadOutChamber[0];
  int chamberindex=0;  
  for(chamberindex=0;chamberindex<540;chamberindex++) {   // Range-for!
      LOG(debug3) << "initialising readout chamber "<< chamberindex;
      cout << "initialising readout chamber "<< chamberindex;
      //((PadParameters<T>)roc).init(chamberindex++);
      mreadOutChamber[chamberindex].init(chamberindex);
      mreadOutChamber[chamberindex].setValue(1,1,42);
      mreadOutChamber[chamberindex].getValue(1,1);
  }
/*   for(const auto& roc : mreadOutChamber) {   // Range-for!
      LOG(debug3) << "initialising readout chamber "<< chamberindex;
      cout << "initialising readout chamber "<< chamberindex;
      //((PadParameters<T>)roc).init(chamberindex++);
      roc.init(chamberindex++);
      ((PadParameters<T>)roc).setValue(1,1,42);
      ((PadParameters<T>)roc).getValue(1,1);

  }*/
}

template <class T> 
void PadCalibrations<T>::init()
{
  //
  // TRDCalPadStatus constructor
  //
  //TRDGeometry fgeom;
  cout << "PADCALIBRATIONS INIT FUNCTION !!!!!!!!!!!!  first pad is as : " << &mreadOutChamber[0] <<  endl;
  int chamberindex=0;  
  for(const auto& roc : mreadOutChamber) {   // Range-for!
      LOG(debug3) << "initialising readout chamber "<< chamberindex;
      cout << "initialising readout chamber "<< chamberindex;
      ((PadParameters<T>)roc).init(chamberindex++);
      ((PadParameters<T>)roc).getValue(1,1);
  }
}




} // namespace trd
} // namespace o2
#endif
