// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iosfwd>
#include <array>
#include <Rtypes.h>

#include "EMCALCalib/TriggerTRUDCS.h"
#include "EMCALCalib/TriggerSTUDCS.h"

namespace o2
{

namespace emcal
{

/// \class TriggerDCS
/// \brief CCDB container for the DCS data in EMCAL
/// \ingroup EMCALcalib
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since December 4th, 2019
///
/// based on AliEMCALTriggerDCSConfig class authored by R. GUERNANE

class TriggerDCS
{
 public:
  /// \brief default constructor
  TriggerDCS() = default;

  /// \brief Destructor
  ~TriggerDCS() = default;

  /// \brief Comparison of two DCS data
  /// \return true if the TRU data are identical, false otherwise
  ///
  /// Testing two DCS for equalness. DCS are considered identical
  /// if the contents are identical.
  bool operator==(const TriggerDCS& other) const;

  /// \brief Serialize object to JSON format
  /// \return JSON-serialized trigger DCS config object
  std::string toJSON() const;

  void setTRUArr(std::vector<TriggerTRUDCS>& ta) { mTRUArr = ta; }
  void setTRU(TriggerTRUDCS tru) { mTRUArr.emplace_back(tru); }

  void setSTUEMCal(TriggerSTUDCS so) { mSTUEMCal = so; }
  void setSTUDCal(TriggerSTUDCS so) { mSTUDCAL = so; }

  std::vector<TriggerTRUDCS> getTRUArr() const { return mTRUArr; }

  TriggerSTUDCS getSTUDCSEMCal() const { return mSTUEMCal; }
  TriggerSTUDCS getSTUDCSDCal() const { return mSTUDCAL; }
  TriggerTRUDCS getTRUDCS(Int_t iTRU) const { return mTRUArr.at(iTRU); }

  /// \brief Check whether TRU is enabled
  /// \param iTRU Index of the TRU
  /// Enabled-status defined via presence of the TRU in the STU region: TRU
  /// is enabled if the corresponding bit is set in the STU region
  bool isTRUEnabled(int iTRU) const;

 private:
  std::vector<TriggerTRUDCS> mTRUArr; ///< TRU array
  TriggerSTUDCS mSTUEMCal;            ///< STU of EMCal
  TriggerSTUDCS mSTUDCAL;             ///< STU of DCAL

  ClassDefNV(TriggerDCS, 1);
};

/// \brief Streaming operator
/// \param in Stream where the TRU parameters are printed on
/// \param dcs TRU to be printed
/// \return Stream after printing the TRU parameters
std::ostream& operator<<(std::ostream& in, const TriggerDCS& dcs);

} // namespace emcal

} // namespace o2
