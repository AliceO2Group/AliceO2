#ifndef O2_TRDCALROC_H
#define O2_TRDCALROC_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

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
class TRDCalROC
{
 public:
  TRDCalROC();
  TRDCalROC(int, int);
  virtual ~TRDCalROC();

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

  // statistic
  double getMean(TRDCalROC* const outlierROC = nullptr) const;
  double getMeanNotNull() const;
  double getRMS(TRDCalROC* const outlierROC = nullptr) const;
  double getRMSNotNull() const;
  double getMedian(TRDCalROC* const outlierROC = nullptr) const;
  double getLTM(double* sigma = nullptr, double fraction = 0.9, TRDCalROC* const outlierROC = nullptr);

  // algebra
  bool add(float c1);
  bool multiply(float c1);
  bool add(const TRDCalROC* roc, double c1 = 1);
  bool multiply(const TRDCalROC* roc);
  bool divide(const TRDCalROC* roc);

  // noise
  bool unfold();

  //Plots
  TH2F* makeHisto2D(float min, float max, int type, float mu = 1.0);
  TH1F* makeHisto1D(float min, float max, int type, float mu = 1.0);

 protected:
  int mPla{ 0 };         //  Plane number
  int mCha{ 0 };         //  Chamber number
  int mNrows{ 0 };       //  Number of rows
  int mNcols{ 0 };       //  Number of columns
  int mNchannels{ 0 };   //  Number of channels
  unsigned short* mData; //[mNchannels] Data
};
} // namespace trd
} // namespace o2
#endif
