// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_TRIGGERMAPPINGV2_H_
#define ALICEO2_EMCAL_TRIGGERMAPPINGV2_H_

#include <array>
#include <bitset>
#include <tuple>
#include "EMCALBase/GeometryBase.h"

namespace o2
{
namespace emcal
{

class Geometry;

/// \class TriggerMappingV2
/// \brief Trigger mapping starting from Run2
/// \ingroup EMCALbase
/// \author H. YOKOYAMA Tsukuba University
/// \author R. GUERNANE LPSC Grenoble CNRS/IN2P3
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
class TriggerMappingV2
{
 public:
  //********************************************
  // static constant
  //********************************************
  static constexpr unsigned int SUPERMODULES = 20;                                                    ///< Total number of supermodules in EMCAL
  static constexpr unsigned int ALLTRUS = 52;                                                         ///< Total number of TRUs in EMCAL
  static constexpr unsigned int FASTORSETATRU = 8;                                                    ///< Number of FastOR/TRU in Eta
  static constexpr unsigned int FASTORSPHITRU = 12;                                                   ///< Number of FastOR/TRU in Phi
  static constexpr unsigned int FASTORSTRU = FASTORSETATRU * FASTORSPHITRU;                           ///< Number of FastOR/TRU
  static constexpr unsigned int TRUSETASM = 3;                                                        ///< Number of TRUs/SM in Eta
  static constexpr unsigned int TRUSPHISM = 1;                                                        ///< Number of TRUs/SM in Phi
  static constexpr unsigned int TRUSSUPERMODULE = TRUSETASM * TRUSPHISM;                              ///< Number of TRUs/SM
  static constexpr unsigned int FASTORSETASM = FASTORSETATRU * TRUSETASM;                             ///< Number of FastOR/SM in Eta
  static constexpr unsigned int FASTORSPHISM = FASTORSPHITRU * TRUSPHISM;                             ///< Number of FastOR/SM in Phi
  static constexpr unsigned int FASTORSETA = 2 /*Aside,Cside*/ * FASTORSETASM;                        ///< EMCAL+DCAL region eta size
  static constexpr unsigned int FASTORSPHI = (5 * FASTORSPHISM) + (1 * FASTORSPHISM / 3)              /*EMCAL*/
                                             + (3 * FASTORSPHISM) + (1 * FASTORSPHISM / 3) /*DCAL */; ///< Number of FastOR/EMCALs in Phi
  static constexpr unsigned int ALLFASTORS = FASTORSETA * FASTORSPHI;                                 ///< Number of FastOR/EMCALs

  //********************************************
  // Index types
  //********************************************
  using IndexTRU = unsigned int;
  using IndexSTU = unsigned int;
  using IndexSupermodule = unsigned int;
  using IndexFastOR = unsigned int;
  using IndexCell = unsigned int;
  using IndexColumnEta = unsigned int;
  using IndexRowPhi = unsigned int;
  using IndexOnline = unsigned int;

  /// \enum DetType_t
  /// \brief Calorimeter type
  enum class DetType_t {
    DET_EMCAL,
    DET_DCAL
  };

  /// \brief Default constructor.
  TriggerMappingV2();

  /// \brief Default constructor.
  TriggerMappingV2(Geometry* geo);

  /// \brief Destructor
  ~TriggerMappingV2() = default;

  //********************************************
  // Get FastOR index from TRU/SM/EMCAL Geometry
  //********************************************

  /// \brief Get the absolute index of the FastOr from the index in the TRU
  /// \param truIndex Index of the TRU in the TRU numbering scheme
  /// \param positionInTRU Channel index within the TRU
  /// \return Absolute ID of the FastOR
  /// \throw TRUIndexExcepiton in case of invalid TRU index
  /// \throw FastORIndexException in case of invalid position in TRU
  ///
  /// Calculating the absolute ID of the FastOR which is a unique
  /// index of a FastOR. It is calcuated as phi + eta * NPhi, added
  /// to sum of FastORs in the TRUs with smaller TRU index.
  IndexFastOR getAbsFastORIndexFromIndexInTRU(IndexTRU truIndex, IndexFastOR fastorIndexTRU) const;

  /// \brief Get the absolute index of the FastOr from geometric position in TRU
  /// \param truIndex Index of the TRU in the TRU numbering scheme
  /// \param etaColumn Position of the channel in eta direction
  /// \param phiRow Position of the channel in phi direction
  /// \return Absolute ID of the FastOR
  /// \throw TRUIndexException in case of invalid TRU indices
  ///
  /// Calculating the absolute ID of the FastOR which is a unique
  /// index of a FastOR. It is calcuated as phi + eta * NPhi, added
  /// to sum of FastORs in the TRUs with smaller TRU index.
  IndexFastOR getAbsFastORIndexFromPositionInTRU(IndexTRU truIndex, IndexColumnEta etaColumn, IndexRowPhi phiRow) const;

  /// \brief Get the absolute index of the FastOr from the geometric position in the supermodule
  /// \param supermoduleID Supermodule index
  /// \param etaColumn Position of the channel in eta direction
  /// \param phiRow Position of the channel in phi direction
  /// \return Absoulte ID of the FastOR
  /// \throw SupermoduleIndexException in case the supermodule ID exceeds the number of supermodules
  /// \throw FastORPositionExceptionSupermodule in case the position is within the supermodule
  ///
  /// Calculating the absolute ID of the FastOR which is a unique
  /// index of a FastOR. It is calcuated as phi + eta * NPhi, added
  /// to sum of FastORs in the TRUs with smaller TRU index.
  IndexFastOR getAbsFastORIndexFromPositionInSupermodule(IndexSupermodule supermoduleID, IndexColumnEta etaColumn, IndexRowPhi phiRow) const;

  /// \brief Get the absolute index of the FastOR from the geometric position in EMCAL
  /// \param etaColumn Position of the channel in eta direction
  /// \param phiRow Position of the channel in phi direction
  /// \return Absoulte ID of the FastOR
  /// \throw FastORPositionExceptionEMCAL in case the position is not a valid index within the EMCAL
  ///
  /// Calculating the absolute ID of the FastOR which is a linear number
  /// scheme in eta and phi of the full EMCAL + DCAL surface defined as
  /// column + 48 * row. TRU index and FastOR index are based on the TRU
  /// numbering scheme.
  IndexFastOR getAbsFastORIndexFromPositionInEMCAL(IndexColumnEta etaColumn, IndexRowPhi phiRow) const;

  /// Trigger mapping method, from position in PHOS Index sub region get FastOR index
  /// \param phosRegionID: Indes of the PHOS subregion
  /// \return Virtual FastOR index for PHOS subregions
  /// \throw PHOSRegionException
  ///
  /// Calculating the absolute ID of the PHOS subregions, which are treated in the
  /// indexing as if they were regular DCAL TRUs and FastORs.
  IndexFastOR getAbsFastORIndexFromPHOSSubregion(unsigned int phosRegionID) const;

  //********************************************
  // Get TRU/SM/EMCAL Geometry from FastOR index
  //********************************************
  /// \brief Get the TRU index and FastOR index in TRU from the absolute FastOR ID
  /// \param fastOrAbsID Absoulte ID of the FastOR
  /// \return tuple with 0 - TRU index in the TRU numbering scheme, 1 - FastOR index in TRU
  /// \throw FastORIndexException in case the FastOR index is not valid
  ///
  /// Inverse mapping function calculating back the index of the FastOR inside a
  /// TRU and its corresponding TRU from the absolute FastOR ID
  std::tuple<IndexTRU, IndexFastOR> getTRUFromAbsFastORIndex(IndexFastOR fastOrAbsID) const;

  /// \brief Get the position of a FastOR inside the TRU from the absolute FastOR ID
  /// \param fastORAbsID Absolute ID of the FastOR
  /// \return tuple with 0 - TRU index in the TRU index scheme, 1 - Position in eta within TRU, 2 - position in phi within TRU
  /// \throw FastORIndexException in case the FastOR index is not valid
  ///
  /// Inverse mapping function calculating the position within a TRU of a FastOR
  /// and its corresponding TRU from the absolute FastOR ID.
  std::tuple<IndexTRU, IndexColumnEta, IndexRowPhi> getPositionInTRUFromAbsFastORIndex(IndexFastOR fastORAbsID) const;

  /// \brief Get the position inside the supermodule from the absolute FastOR ID
  /// \param fastORAbsID Absolute ID of the FastOR
  /// \return tuple with 0 - Supermodule ID, 1 - position in eta, 2 - position in phi
  /// \throw FastORIndexException in case the FastOR index is not valid
  ///
  /// Inverse mapping function calculating the postion within a supermodule
  /// of a FastOR and its corresponding supermodule ID from the absolute FastOR ID
  std::tuple<IndexSupermodule, IndexColumnEta, IndexRowPhi> getPositionInSupermoduleFromAbsFastORIndex(IndexFastOR fastORAbsID) const;

  /// \brief Get the position in the Detector from the absolute FastOR ID
  /// \param fastORAbsID Absolute ID of the FastOR
  /// \return tuple with 0 - position in eta, 1 - position in phi
  /// \throw FastORIndexException in case the FastOR index is not valid
  ///
  /// Inverse mapping frunction calculating the position of a FastOR within
  /// the EMCAL+DCAL surface from the absolute FastOR ID.
  std::tuple<IndexColumnEta, IndexRowPhi> getPositionInEMCALFromAbsFastORIndex(IndexFastOR fastORAbsID) const;

  //********************************************
  // TRU vs. STU
  //********************************************

  /// \brief Convert TRU and FastOR index in TRU from STU number scheme to TRU number scheme
  /// \param truIndexSTU TRU index in the STU number scheme
  /// \param fastOrIndexSTU FastOR index in STU number scheme
  /// \param detector Subdetector (EMCAL or DCAL)
  /// \return Tuple with 0 - TRU index in TRU number scheme, 1 - FastOR index in TRU number scheme
  /// \throw TRUIndexException in case the TRU index exceeds the max.number of TRUs in the EMCAL / DCAL STU region
  /// \throw FastORIndexException in case the FastOR index exceeds the number of FastORs in a TRU
  ///
  /// TRU and STU use different index schemes both for TRU indexing and FastOR indexing: For TRUs
  /// the TRU numbering combines the TRUs in EMCAL and DCAL, including virtual TRU indices in the PHOS
  /// retion, while the STU numbering scheme splits the TRU indicies in regions for EMCAL and DCAL
  /// excluding the PHOS regions. For what concerns the FastOR numbering scheme the STU numbering
  /// scheme uses a simple linerar indexing in eta and then in phi, while the TRU numbering scheme
  /// uses a complicated indexing starting in phi direction. Consequently a FastOR can have two different
  /// indices with the TRU depending on numbering scheme. The function converts TRU and FastOR indices
  /// from the STU scheme to the TRU scheme.
  std::tuple<IndexTRU, IndexFastOR> convertFastORIndexSTUtoTRU(IndexTRU truIndexSTU, IndexFastOR fastOrIndexSTU, DetType_t detector) const;

  /// \brief Convert TRU and FastOR index in TRU from TRU number scheme to STU number scheme
  /// \param truIndexTRU TRU index in the TRU number scheme
  /// \param fastorIndexTRU FastOR index in TRU number scheme
  /// \return Tuple with 0 - TRU index in STU number scheme, 1 - FastOR index in STU number scheme
  /// \throw TRUIndexException in case the TRU index exceeds the max.number of TRUs
  /// \throw FastORIndexException in case the FastOR index exceeds the number of FastORs in a TRU
  ///
  /// TRU and STU use different index schemes both for TRU indexing and FastOR indexing: For TRUs
  /// the TRU numbering combines the TRUs in EMCAL and DCAL, including virtual TRU indices in the PHOS
  /// retion, while the STU numbering scheme splits the TRU indicies in regions for EMCAL and DCAL
  /// excluding the PHOS regions. For what concerns the FastOR numbering scheme the STU numbering
  /// scheme uses a simple linerar indexing in eta and then in phi, while the TRU numbering scheme
  /// uses a complicated indexing starting in phi direction. Consequently a FastOR can have two different
  /// indices with the TRU depending on numbering scheme. The function converts TRU and FastOR indices
  /// from the TRU scheme to the STU scheme.
  std::tuple<IndexTRU, IndexFastOR> convertFastORIndexTRUtoSTU(IndexTRU truIndexTRU, IndexFastOR fastorIndexTRU) const;

  /// \brief Convert TRU and FastOR position in TRU from STU number scheme to TRU number scheme
  /// \param truIndexSTU TRU index in the STU number scheme
  /// \param truEtaSTU Column of the FastOR  in STU number scheme
  /// \param truPhiSTU Row of the FastOR in STU number scheme
  /// \param detector Subdetector (EMCAL or DCAL)
  /// \return Tuple with 0 - TRU index in TRU number scheme, 1 - Column of the FastOR in TRU number scheme, 2 - Row of the FastOR in TRU number scheme
  /// \throw TRUIndexException in case the TRU index exceeds the max.number of TRUs in the EMCAL / DCAL STU region
  /// \throw FastORPositionExceptionTRU in case the position in column and row exceeds the number of columns or rows of a TRU
  ///
  /// TRU and STU use different index schemes both for TRU indexing and FastOR indexing: For TRUs
  /// the TRU numbering combines the TRUs in EMCAL and DCAL, including virtual TRU indices in the PHOS
  /// retion, while the STU numbering scheme splits the TRU indicies in regions for EMCAL and DCAL
  /// excluding the PHOS regions. For what concerns the FastOR numbering scheme the STU numbering
  /// scheme uses a simple linerar indexing in eta and then in phi, while the TRU numbering scheme
  /// uses a complicated indexing starting in phi direction. Consequently a FastOR can have two different
  /// indices with the TRU depending on numbering scheme. The function converts TRU and FastOR position
  /// from the STU scheme to the TRU scheme.
  std::tuple<IndexTRU, IndexColumnEta, IndexRowPhi> convertFastORPositionSTUtoTRU(IndexTRU truIndexSTU, IndexColumnEta truEtaSTU, IndexRowPhi truPhiSTU, DetType_t detector) const;

  /// \brief Convert TRU and FastOR position in TRU from TRU number scheme to STU number scheme
  /// \param truIndexSTU TRU index in the TRU number scheme
  /// \param truEtaSTU Column of the FastOR  in TRU number scheme
  /// \param truPhiSTU Row of the FastOR in TRU number scheme
  /// \return Tuple with 0 - TRU index in STU number scheme, 1 - Column of the FastOR in STU number scheme, 2 - Row of the FastOR in STU number scheme
  /// \throw TRUIndexException in case the TRU index exceeds the max.number of TRUs
  /// \throw FastORPositionExceptionTRU in case the position in column and row exceeds the number of columns or rows of a TRU
  ///
  /// TRU and STU use different index schemes both for TRU indexing and FastOR indexing: For TRUs
  /// the TRU numbering combines the TRUs in EMCAL and DCAL, including virtual TRU indices in the PHOS
  /// retion, while the STU numbering scheme splits the TRU indicies in regions for EMCAL and DCAL
  /// excluding the PHOS regions. For what concerns the FastOR numbering scheme the STU numbering
  /// scheme uses a simple linerar indexing in eta and then in phi, while the TRU numbering scheme
  /// uses a complicated indexing starting in phi direction. Consequently a FastOR can have two different
  /// indices with the TRU depending on numbering scheme. The function converts TRU and FastOR position
  /// from the TRU scheme to the STU scheme.
  std::tuple<IndexTRU, IndexColumnEta, IndexRowPhi> convertFastORPositionTRUtoSTU(IndexTRU truIndexTRU, IndexColumnEta etaTRU, IndexRowPhi phiTRU) const;

  //********************************************
  // Cell Index
  //********************************************

  /// \brief Get the absolute FastOR index of the module containing a given cell
  /// \param cellIndex Index of the cell
  /// \return Absolute index of the FastOR
  /// \throw GeometryNotSetException in case the Geometry is not initialized
  /// \throw InvalidCellIDException in case the cell ID is outside range
  IndexFastOR getAbsFastORIndexFromCellIndex(IndexCell cellIndex) const;

  /// \brief Get the indices of the cells in the module of a given FastOR
  /// \param fastORAbsID Absolute index of the FastOR
  /// \return Cell indices of the module (order: eta, phi)
  /// \throw GeometryNotSetException in case the geometry is not initialized
  /// \throw FastORIndexException in case the FastOR ID is invalid
  std::array<IndexCell, 4> getCellIndexFromAbsFastORIndex(IndexFastOR fastORAbsID) const;

  //********************************************
  // TRU index
  //********************************************

  /// \brief Convert the TRU index from the STU numbering scheme into the TRU numbering scheme
  /// \param truIndexSTU TRU index im STU numbering scheme
  /// \param detector Subdetector type
  /// \return TRU index in TRU numbering scheme
  /// \throw TRUIndexException in case the TRU index exceeds the max. number of TRUs in the EMCAL/DCAL STU region
  ///
  /// The index scheme in the STU definition uses separate ranges for EMCAL and DCAL, where
  /// the EMCAL contains 32 TRUs and the DCAL 14 TRUs. The index is converted into the TRU
  /// index scheme adding the TRU index in DCAL + the amount of virtual TRUs in the PHOS region
  /// (2 per sector) to the TRU index in EMCAL.
  IndexTRU convertTRUIndexSTUtoTRU(IndexTRU truIndexSTU, DetType_t detector) const;

  /// Convert the TRU index from the TRU numbering scheme into the STU numbering scheme
  /// \param truIndexTRU TRU index im TRU numbering scheme
  /// \return TRU index in TRU numbering scheme
  ///
  /// The index scheme in TRU definintion consists of a linear list of TRU indices combining EMCAL
  /// and DCAL TRUs, togtehter with virtual TRUs in the PHOS region (2 per sector). The TRU index
  /// is spearated for TRUs in EMCAL and DCAL, in each side starting with 0. On the EMCAL side the
  /// TRU index is the same in both index schemes. On the DCAL side the virtual TRU index of the
  /// the PHOS region is dropped, therefore calling the function for TRU indices in the PHOS region
  /// leads to the TRU indices of the corresponding C-side TRUs in the same sector.
  IndexTRU convertTRUIndexTRUtoSTU(IndexTRU truIndexTRU) const;

  /// \brief Get the TRU index from the online index
  /// \return TRU index (= online index)
  IndexTRU getTRUIndexFromOnlineIndex(IndexOnline onlineIndex) const noexcept { return onlineIndex; };

  /// \brief Get the online index from the TRU index
  /// \return Online index (= TRU index)
  IndexOnline getOnlineIndexFromTRUIndex(IndexTRU truIndex) const { return truIndex; };

  /// \brief Get the TRU Index from the hardware address of the ALTRO channel (TRU rawdata)
  /// \param hardwareAddress hardware address
  /// \param ddlID ID of the DDL of the ALTRO channel
  /// \param supermoduleID uper-module number
  /// \return TRU global offline number from:
  /// \throw TRUIndexException in case the TRU index is out-of-range
  IndexTRU getTRUIndexFromOnlineHardareAddree(int hardwareAddress, unsigned int ddlID, unsigned int supermoduleID) const;

  //********************************************
  // L0 Index
  //********************************************

  /// Trigger mapping method, from L0 index get FastOR index
  /// \param iTRU: TRU index
  /// \param l0index:  L0? index
  /// \param idx: indeces associated to FASTOR?
  /// \param size: ?
  /// \return true if found
  std::array<unsigned int, 4> getFastORIndexFromL0Index(IndexTRU iTRU, IndexFastOR l0index, int size) const;

  /// \struct FastORInformation
  /// \brief Basic FastOR information
  struct FastORInformation {
    unsigned int mTRUID;                ///< Index of the TRU
    unsigned int mFastORIDTRU;          ///< Online FastOR index in TRU
    unsigned int mColumnEtaTRU;         ///< Column in the TRU
    unsigned int mRowPhiTRU;            ///< Row in the TRU
    unsigned int mSupermoduleID;        ///< Supermodule
    unsigned int mColumnEtaSupermodule; ///< Column in the supermodule
    unsigned int mRowPhiSupermodule;    ///< Row in the supermodule
  };

  FastORInformation getInfoFromAbsFastORIndex(IndexFastOR absFastORID) const;

  //********************************************
  // getters arrays (for debugging)
  //********************************************
  const std::array<unsigned int, ALLTRUS>& getArrayTRUFastOROffsetX() const { return mTRUFastOROffsetX; }
  const std::array<unsigned int, ALLTRUS>& getTRUFastOROffsetY() const { return mTRUFastOROffsetY; }
  const std::array<unsigned int, ALLTRUS>& getNFastORInTRUPhi() const { return mNFastORInTRUPhi; }
  const std::array<unsigned int, ALLTRUS>& getNFastORInTRUEta() const { return mNFastORInTRUEta; }
  const std::array<unsigned int, SUPERMODULES>& getArraySMFastOROffsetX() const { return mSMFastOROffsetX; }
  const std::array<unsigned int, SUPERMODULES>& getArraySMFastOROffsetY() const { return mSMFastOROffsetY; }
  const std::array<unsigned int, SUPERMODULES>& getArrayNFastORInSMPhi() const { return mNFastORInSMPhi; }
  const std::array<unsigned int, SUPERMODULES>& getArrayNFastORInSMEta() const { return mNFastORInSMEta; }
  const std::array<unsigned int, 5>& getArrayNModuleInEMCALPhi() const { return mNModuleInEMCALPhi; }

 private:
  //********************************************
  // fastOR offset parameters
  //********************************************
  o2::emcal::Geometry* mGeometry;
  std::array<unsigned int, ALLTRUS> mTRUFastOROffsetX; ///< FastOR offset per TRU in eta direction
  std::array<unsigned int, ALLTRUS> mTRUFastOROffsetY; ///< FastOR offset per TRU in phi direction
  std::array<unsigned int, ALLTRUS> mNFastORInTRUPhi;  ///< Number of FastORs per TRU in phi direction (for handling 1/3rd supermodules)
  std::array<unsigned int, ALLTRUS> mNFastORInTRUEta;  ///< Number of FastORs per TRU in eta direction (for handling 1/3rd supermodules)
  std::bitset<ALLTRUS> mTRUIsCside;                    ///< Marker for C-side supermodules (bit index := supermodule ID)

  std::array<unsigned int, SUPERMODULES> mSMFastOROffsetX; // FastOR offset[#of SM ]
  std::array<unsigned int, SUPERMODULES> mSMFastOROffsetY; //
  std::array<unsigned int, SUPERMODULES> mNFastORInSMPhi;  // SM size
  std::array<unsigned int, SUPERMODULES> mNFastORInSMEta;  //

  std::array<unsigned int, 5> mNModuleInEMCALPhi; //#FastOR/EMCAL in Phi

  //********************************************
  // Initialization of FastOR index offset of each SM/TRU
  //********************************************

  /// Initialize mapping offsets for the various TRUs
  void init_TRU_offset();
  void init_SM_offset();

  /// \brief Reset all interal arrays with 0
  void reset_arrays();

  //********************************************
  // Rotation methods from between eta and phi orientation
  //********************************************

  /// \brief Convert absolute FastOR index into increasing order with phi
  /// \param fastorIndexInEta Absolute FastOR index in eta orientation
  /// \return Absolute FastOR index in phi orientation
  ///
  /// Input is expected to have the FastOR index oriented in eta direction, meaning
  /// that the index increases as first dimension in eta direction and as second
  /// dimension in phi direction. After the conversion the index increases as first
  /// dimension with phi and as second dimension with eta. The rotation is relative
  /// to the sector (A+C side supermodules). The FastOR index is absolute, meaning
  /// that the offset of the sector is added to the relative position in the sector.
  ///
  /// The conversion is usually applied converting from the STU numbering scheme, which
  /// first increments in eta, to the TRU numbering scheme which first increments in phi.
  IndexFastOR rotateAbsFastOrIndexEtaToPhi(IndexFastOR fastorIndexInEta) const;

  /// \brief Convert absolute FastOR index into increasing order with eta
  /// \param fastorIndexInEta Absolute FastOR index in phi orientation
  /// \return Absolute FastOR index in eta orientation
  ///
  /// Input is expected to have the FastOR index oriented in phi direction, meaning
  /// that the index increases as first dimension in phi direction and as second
  /// dimension in eta direction, relative to the sector (A+C side supermodules).
  /// After the conversion the index increases as first dimension with eta and as
  /// second dimension with phi. The rotation stays relative to the sector. The FastOR
  /// index is absolute, meaning that the offset of the sector is added to the relative
  /// position in the sector.
  ///
  /// The conversion is usually applied converting from the TRU numbering scheme, which
  /// first increments in phi, to the STU numbering scheme which first increments in eta.
  IndexFastOR rotateAbsFastOrIndexPhiToEta(IndexFastOR fastorIndexInPhi) const;

  /// \brief Get the type of the supermodule
  /// \param supermoduleID Index of the supermodule
  /// \return type of the supermodule
  /// \throw SupermoduleIndexException in case the supermoduleID exceeds the number of supermodules
  EMCALSMType getSupermoduleType(IndexSupermodule supermoduleID) const
  {
    if (supermoduleID >= SUPERMODULES) {
      throw SupermoduleIndexException(supermoduleID, SUPERMODULES);
    }
    if (supermoduleID < 10) {
      return EMCAL_STANDARD;
    }
    if (supermoduleID < 12) {
      return EMCAL_THIRD;
    }
    if (supermoduleID < 18) {
      return DCAL_STANDARD;
    }
    if (supermoduleID < 20) {
      return DCAL_EXT;
    }
    // silence compiler warning (check already done before in order to avoid unnecessary filter)
    throw SupermoduleIndexException(supermoduleID, SUPERMODULES);
  }

  /// \brief Check if the supermodule is a C-side supermodule
  /// \param supermoduleID Index of the supermodule
  /// \return true if the supermodule is on the C-side (odd index), false if it is on the A-side (even index)
  bool isSupermoduleOnCSide(IndexSupermodule supermoduleID) const
  {
    return (supermoduleID % 2 == 1) ? true : false;
  }

  ClassDefNV(TriggerMappingV2, 1);
};

} // namespace emcal
} // namespace o2

#endif // ALIEMCALTRIGGERMAPPINGV2_H
