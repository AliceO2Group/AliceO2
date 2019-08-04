// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_TRDCALROC_H
#define O2_TRDCALROC_H

//////////////////////////////////////////////////
//                                              //
//  TRD calibration base class for one ROC      //
//                                              //
//////////////////////////////////////////////////

class TH1F;
class TH2F;

namespace o2
{
namespace trd
{
class CalROC
{
 public:
  CalROC() = default;
  CalROC(int, int);
  ~CalROC() = default;

  int getNrows() const { return mNrows; };
  int getNcols() const { return mNcols; };
  int getChannel(int c, int r) const { return r + c * mNrows; };
  int getNchannels() const { return mNchannels; };
  float getValue(int ich) const { return (float)mData[ich] / 10000; };
  float getValue(int col, int row) { return getValue(getChannel(col, row)); };
  void setValue(int ich, float value) { mData[ich] = (unsigned short)(value * 10000); };
  void setValue(int col, int row, float value)
  {
    setValue(getChannel(col, row), value);
  };
  void setName(std::string name) { mName = name; };
  void setTitle(std::string title) { mTitle = title; };
  std::string& getName() { return mName; };
  std::string& getTitle() { return mTitle; };
  // statistic
  double getMean(CalROC* const outlierROC = nullptr) const;
  double getMeanNotNull() const;
  double getRMS(CalROC* const outlierROC = nullptr) const;
  double getRMSNotNull() const;
  double getMedian(CalROC* const outlierROC = nullptr) const;
  double getLTM(double* sigma = nullptr, double fraction = 0.9, CalROC* const outlierROC = nullptr);

  // algebra
  bool add(float c1);
  bool multiply(float c1);
  bool add(const CalROC* roc, double c1 = 1);
  bool multiply(const CalROC* roc);
  bool divide(const CalROC* roc);

  // noise
  bool unfold();

  //Plots
  TH2F* makeHisto2D(float min, float max, int type, float mu = 1.0);
  TH1F* makeHisto1D(float min, float max, int type, float mu = 1.0);

 protected:
  int mPla{0};                       //  Plane number
  int mCha{0};                       //  Chamber number
  int mNrows{0};                     //  Number of rows
  int mNcols{0};                     //  Number of columns
  int mNchannels{0};                 //  Number of channels
  std::string mName;                 // for naming spectra, originally inherited from TNamed
  std::string mTitle;                // for prepending to spectra title spectra, originally inherited from TNamed
  std::vector<unsigned short> mData; //[mNchannels] Data
};
} // namespace trd
} // namespace o2
#endif
