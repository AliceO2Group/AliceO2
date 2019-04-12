// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDCALDET_H
#define O2_TRDCALDET_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved per detector        //
//  2019 - Ported from AliRoot to O2 (J. Lopez)                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

class TRDGeometry;
class TRDPadPlane;

class TH1F;
class TH2F;

namespace o2
{
namespace trd
{
class CalDet
{
 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540 };
  CalDet(std::string name = "CalDet", std::string title = "CalDet") : mName(name), mTitle(title){};
  ~CalDet() = default;
  //
  float getValue(int d) const { return mData[d]; };
  float getValue(int p, int c, int s) const { return mData[TRDGeometry::getDetector(p, c, s)]; };
  void setValue(int d, float value) { mData[d] = value; };
  void setValue(int p, int c, int s, float value) { mData[TRDGeometry::getDetector(p, c, s)] = value; };
  void setName(std::string name) { mName = name; }; // these 4 get and set methods are probably not needed (were not in the old code) but are here for completeness
  void setTitle(std::string title) { mTitle = title; };
  std::string& getName() { return mName; };
  std::string& getTitle() { return mTitle; };
  // statistic
  double getMean(CalDet* const outlierDet = nullptr) const;
  double getMeanRobust(double robust = 0.92) const;
  double getRMS(CalDet* const outlierDet = nullptr) const;
  double getRMSRobust(double robust = 0.92) const;
  double getMedian(CalDet* const outlierDet = nullptr) const;
  double getLTM(double* sigma = nullptr, double fraction = 0.9, CalDet* const outlierDet = nullptr);
  double calcMean(bool wghtPads = false);
  double calcMean(bool wghtPads, int& calib);
  double calcRMS(bool wghtPads = false);
  double calcRMS(bool wghtPads, int& calib);
  double getMeanSM(bool wghtPads, int sector) const;
  // Plot functions
  TH1F* makeHisto1Distribution(float min = 4, float max = -4, int type = 0);
  TH1F* makeHisto1DAsFunctionOfDet(float min = 4, float max = -4, int type = 0);
  TH2F* makeHisto2DCh(int ch, float min = 4, float max = -4, int type = 0);
  TH2F* makeHisto2DSmPl(int sm, int pl, float min = 4, float max = -4, int type = 0);
  // algebra functions
  void add(float c1);
  void multiply(float c1);
  void add(const CalDet* calDet, double c1 = 1);
  void multiply(const CalDet* calDet);
  void divide(const CalDet* calDet);

 protected:
  std::array<float, kNdet> mData{}; // Data
  std::string mName;                // name for spectra, carried over originally from inheritence from TNamed
  std::string mTitle;               // title prepend for spectra, carried over originally from inheritence from TNamed
};
} // namespace trd
} // namespace o2
#endif
