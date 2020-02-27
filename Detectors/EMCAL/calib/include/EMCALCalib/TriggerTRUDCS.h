// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_TRIGGERTRUDCS_H
#define ALICEO2_EMCAL_TRIGGERTRUDCS_H

#include <iosfwd>
#include <array>
#include <Rtypes.h>

namespace o2
{

namespace emcal
{

/// \class TriggerTRUDCS
/// \brief CCDB container for TRU DCS data in EMCAL
/// \ingroup EMCALcalib
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since December 4th, 2019
///
/// based on AliEMCALTriggerTRUDCSConfig class authored by R. GUERNANE

class TriggerTRUDCS
{
 public:
  /// \brief default constructor
  TriggerTRUDCS() = default;

  /// \brief copy constructor
  TriggerTRUDCS(const TriggerTRUDCS& tru) = default;

  /// \brief Assignment operator
  TriggerTRUDCS& operator=(const TriggerTRUDCS& source) = default;

  /// \brief nomral constructor
  explicit TriggerTRUDCS(uint64_t selpf,
                         uint64_t l0sel,
                         uint64_t l0cosm,
                         uint64_t gthrl0,
                         uint64_t rlbkstu,
                         uint64_t fw,
                         std::array<uint32_t, 6> maskReg);

  /// \brief Destructor
  ~TriggerTRUDCS() = default;

  /// \brief Comparison of two TRU data
  /// \return true if the TRU data are identical, false otherwise
  ///
  /// Testing two TRUs for equalness. TRUs are considered identical
  /// if the contents are identical.
  bool operator==(const TriggerTRUDCS& other) const;

  /// \brief Serialize object to JSON format
  /// \return JSON-serialized TRU DCS config object
  std::string toJSON() const;

  void setSELPF(uint64_t pf) { mSELPF = pf; }
  void setL0SEL(uint64_t la) { mL0SEL = la; }
  void setL0COSM(uint64_t lc) { mL0COSM = lc; }
  void setGTHRL0(uint64_t lg) { mGTHRL0 = lg; }
  void setMaskReg(uint32_t msk, int pos) { mMaskReg[pos] = msk; }
  void setRLBKSTU(uint64_t rb) { mRLBKSTU = rb; }
  void setFw(uint64_t fw) { mFw = fw; }

  uint64_t getSELPF() const { return mSELPF; }
  uint64_t getL0SEL() const { return mL0SEL; }
  uint64_t getL0COSM() const { return mL0COSM; }
  uint64_t getGTHRL0() const { return mGTHRL0; }
  uint32_t getMaskReg(int pos) const { return mMaskReg[pos]; }
  uint64_t getRLBKSTU() const { return mRLBKSTU; }
  uint64_t getFw() const { return mFw; }

  //int   getSegmentation();

  /// \brief Print TRUs on a given stream
  /// \param stream Stream on which the TRU is printed on
  ///
  /// Printing all the parameters of the TRU on the stream.
  ///
  /// The function is called in the operator<< providing direct access
  /// to protected members. Explicit calls by the users is normally not
  /// necessary.
  void PrintStream(std::ostream& stream) const;

 private:
  uint64_t mSELPF = 0x1e1f;         ///< PeakFinder setup
  uint64_t mL0SEL = 0x1;            ///< L0 Algo selection
  uint64_t mL0COSM = 0;             ///< 2x2
  uint64_t mGTHRL0 = 0;             ///< 4x4
  std::array<uint32_t, 6> mMaskReg; ///< 6*16 = 96 mask bits per TRU
  uint64_t mRLBKSTU = 0;            ///< TRU circular buffer rollback
  uint64_t mFw = 0x21;              ///< TRU fw version

  ClassDefNV(TriggerTRUDCS, 1);
};

/// \brief Streaming operator
/// \param in Stream where the TRU parameters are printed on
/// \param tru TRU to be printed
/// \return Stream after printing the TRU parameters
std::ostream& operator<<(std::ostream& in, const TriggerTRUDCS& tru);

} // namespace emcal

} // namespace o2

#endif
