// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CHAMBERCALIBRATIONS_H
#define O2_TRD_CHAMBERCALIBRATIONS_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

//
class TRDGeometry;

namespace o2
{
namespace trd
{
class ChamberCalibrations
{
 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540 };
  ChamberCalibrations() = default;
  ~ChamberCalibrations() = default;
  //
  float getVDrift(int p, int c, int s) const { return mVDrift.at(o2::trd::TRDGeometry::getDetector(p, c, s)); };
  float getVDrift(int roc) const { return mVDrift.at(roc); };
  float getGainFactor(int p, int c, int s) const { return mGainFactor.at(TRDGeometry::getDetector(p, c, s)); };
  float getGainFactor(int roc) const { return mGainFactor.at(roc); };
  float getT0(int p, int c, int s) const { return mT0.at(TRDGeometry::getDetector(p, c, s)); };
  float getT0(int roc) const { return mT0.at(roc); };
  float getExB(int p, int c, int s) const { return mExB.at(TRDGeometry::getDetector(p, c, s)); };
  float getExB(int roc) const { return mExB.at(roc); };
  void setVDrift(int p, int c, int s, float vdrift) { mVDrift.at(o2::trd::TRDGeometry::getDetector(p, c, s)) = vdrift; };
  void setVDrift(int roc, float vdrift) { mVDrift.at(roc) = vdrift; };
  void setGainFactor(int p, int c, int s, float gainfactor) { mGainFactor.at(TRDGeometry::getDetector(p, c, s)) = gainfactor; };
  void setGainFactor(int roc, float gainfactor) { mGainFactor.at(roc) = gainfactor; };
  void setT0(int p, int c, int s, float t0) { mT0.at(TRDGeometry::getDetector(p, c, s)) = t0; };
  void setT0(int roc, float t0) { mT0.at(roc) = t0; };
  void setExB(int p, int c, int s, float exb) { mExB.at(TRDGeometry::getDetector(p, c, s)) = exb; };
  void setExB(int roc, float exb) { mExB.at(roc) = exb; };
  //bulk gets ?
  int loadReferenceCalibrations(int run2number);
  bool init(int run2run = 0);

 protected:
  std::array<float, kNdet> mVDrift{};
  std::array<float, kNdet> mGainFactor{};
  std::array<float, kNdet> mT0{};
  std::array<float, kNdet> mExB{}; //
  std::string mName;               // name for spectra, carried over originally from inheritence from TNamed
  std::string mTitle;              // title prepend for spectra, carried over originally from inheritence from TNamed
};
} // namespace trd
} // namespace o2
#endif
