// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_PADPARAMETERS_H
#define O2_TRD_PADPARAMETERS_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

//
#include <vector>

class TRDGeometry;

using namespace std;
namespace o2
{
namespace trd
{

template <class T>
class PadParameters
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
  PadParameters(int p, int c);
  ~PadParameters() = default;
  int init(int p, int c, std::vector<T>& data);
  //

  int getNrows() const { return mNrows; };
  int getNcols() const { return mNcols; };
  int getChannel(int c, int r) const { return r + c * mNrows; };
  int getNchannels() const { return mNchannels; };
  T getValue(int ich) const { return mData[ich]; };
  T getValue(int col, int row) { return getValue(getChannel(col, row)); };
  void setValue(int ich, T value) { mData[ich] = value; };
  void setValue(int col, int row, T value) { setValue(getChannel(col, row), value); };

  // statistic
  // Need to ponder these functions, they may be better in a higher up class, related to their own perculiarities.
  /*  double getMean(CalROC* const outlierROC = nullptr) const;
  double getMeanNotNull() const;
  double getRMS(CalROC* const outlierROC = nullptr) const;
  double getRMSNotNull() const;
  double getMedian(CalROC* const outlierROC = nullptr) const;
  double getLTM(double* sigma = nullptr, double fraction = 0.9, CalROC* const outlierROC = nullptr);
*/
  // algebra
  bool add(float c1);
  bool multiply(float c1);
  /* TODO I dont understand the definition of add, multiply which includes
 * a scaling  factor for addition, that makes it not addition but something else.
 * go back into aliroot code and figure out where its called and why.
 * bool add(const PadParam<T>* roc, double c1 = 1);
  bool multiply(const PadParam<T>* roc);
  bool divide(const PadParam<T>* roc);
  // this is used for the noise studies.
  bool unfold();
  */
 protected:
  int mPlane{0};        //  Plane number
  int mChamber{0};      //  Chamber number
  int mNrows{0};        //  Number of rows
  int mNcols{0};        //  Number of columns
  int mNchannels{0};    //  Number of channels = rows*columns
  std::vector<T> mData; // Size is mNchannels
};
} // namespace trd
} // namespace o2
#endif
