// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_TRIGGERSTUDCS_H
#define ALICEO2_EMCAL_TRIGGERSTUDCS_H

#include <iosfwd>
#include <array>
#include <Rtypes.h>

namespace o2
{

namespace emcal
{

/// \class TriggerSTUDCS
/// \brief CCDB container for STU DCS data in EMCAL
/// \ingroup EMCALcalib
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since December 4th, 2019
///
/// based on AliEMCALTriggerSTUDCSConfig class authored by R. GUERNANE

class TriggerSTUDCS
{
 public:
  /// \brief default constructor
  TriggerSTUDCS() = default;

  /// \brief copy constructor
  TriggerSTUDCS(const TriggerSTUDCS& stu) = default;

  /// \brief Assignment operator
  TriggerSTUDCS& operator=(const TriggerSTUDCS& source) = default;

  /// \brief nomral constructor
  explicit TriggerSTUDCS(std::array<int, 3> Gammahigh,
                         std::array<int, 3> Jethigh,
                         std::array<int, 3> Gammalow,
                         std::array<int, 3> Jetlow,
                         int rawData,
                         int region,
                         int fw,
                         int patchSize,
                         int median,
                         std::array<int, 4> phosScale);

  /// \brief Destructor
  ~TriggerSTUDCS() = default;

  /// \brief Comparison of two STU data
  /// \return true if the STU data are identical, false otherwise
  ///
  /// Testing two STUs for equalness. STUs are considered identical
  /// if the contents are identical.
  bool operator==(const TriggerSTUDCS& other) const;

  /// \brief Serialize object to JSON format
  /// \return JSON-serialized STU DCS config object
  std::string toJSON() const;

  void setGammaHigh(int vzpar, int val) { mGammaHigh[vzpar] = val; }
  void setJetHigh(int vzpar, int val) { mJetHigh[vzpar] = val; }
  void setGammaLow(int vzpar, int val) { mGammaLow[vzpar] = val; }
  void setJetLow(int vzpar, int val) { mJetLow[vzpar] = val; }
  void setRawData(int rd) { mGetRawData = rd; }
  void setRegion(int rg) { mRegion = rg; }
  void setFw(int fv) { mFw = fv; }
  void setPHOSScale(int iscale, int val) { mPHOSScale[iscale] = val; }
  void setPatchSize(int size) { mPatchSize = size; }
  void setMedianMode(int mode) { mMedian = mode; }

  int getGammaHigh(int vzpar) const { return mGammaHigh[vzpar]; }
  int getJetHigh(int vzpar) const { return mJetHigh[vzpar]; }
  int getGammaLow(int vzpar) const { return mGammaLow[vzpar]; }
  int getJetLow(int vzpar) const { return mJetLow[vzpar]; }
  int getRawData() const { return mGetRawData; }
  int getRegion() const { return mRegion; }
  int getFw() const { return mFw; }
  int getPHOSScale(int iscale) const { return mPHOSScale[iscale]; }
  int getPatchSize() const { return mPatchSize; }
  int getMedianMode() const { return mMedian; }

  //void    getSegmentation(TVector2& v1, TVector2& v2, TVector2& v3, TVector2& v4) const;

  /// \brief Print STUs on a given stream
  /// \param stream Stream on which the STU is printed on
  ///
  /// Printing all the parameters of the STU on the stream.
  ///
  /// The function is called in the operator<< providing direct access
  /// to protected members. Explicit calls by the users is normally not
  /// necessary.
  void PrintStream(std::ostream& stream) const;

 private:
  std::array<int, 3> mGammaHigh; ///< Gamma high trigger
  std::array<int, 3> mJetHigh;   ///< jet low trigger
  std::array<int, 3> mGammaLow;  ///< Gamma low trigger
  std::array<int, 3> mJetLow;    ///< jet low trigger
  int mGetRawData = 1;           ///< GetRawData
  int mRegion = 0xFFFFFFFF;      ///< Region
  int mFw = 0x2A012;             ///< Firmware version
  std::array<int, 4> mPHOSScale; ///< PHOS scale factors
  int mPatchSize = 0;            ///< Jet patch size: 0 for 8x8 and 2 for 16x16
  int mMedian = 0;               ///< 1 in case of using EMCAL/DCAL for estimating the median

  ClassDefNV(TriggerSTUDCS, 1);
};

/// \brief Streaming operator
/// \param in Stream where the STU parameters are printed on
/// \param stu STU to be printed
/// \return Stream after printing the STU parameters
std::ostream& operator<<(std::ostream& in, const TriggerSTUDCS& stu);

} // namespace emcal

} // namespace o2

#endif
