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
#include <bitset>
#include <Rtypes.h>

class TH2;

namespace o2
{

namespace emcal
{

/// \class BadChannelMap
/// \brief CCDB container for masked cells in EMCAL
/// \author Markus Fasel <>, Oak Ridge National Laboratory
/// \since June 25th, 2019
///
/// # The EMCAL Bad Channel Map
///
/// The bad channel map contains channels which are found to be
/// bad, dead or problematic during the calibration process and need
/// to be masked so they are not considered in the clusterization
/// or other analysis processes. Cells can be either bad, meaning
/// that the spectra shape is so much distorted that recovery is not
/// possible, dead, meaning that no signal is seen over a large datasample,
/// or warm, indicating that the cell might be recovered by recalibration
/// of the energy.
///
/// Bad channels can be added via
/// ~~~.{cxx}
/// o2::emcal::BadChannelMap bcm;
/// bcm.addBadChannel(1234, o2::emcal::BadChannelMap::MaskStatus_t::BAD_CELL);
/// ~~~
/// The cell container stores only bad or warm cells, good cells
/// are ignored. However the function addBadChannel also updates
/// the container with a new channel status - calling addBadChannel
///  with the status GOOD_CELL for a cell which is stored in the container
/// the cell is removed.
///
/// Reading the channel status is done via
/// ~~~.{cxx}
/// auto status = bcm.getChannelStatus(1234); // returning BAD_CELL in this case
/// ~~~
/// Calling getChannelStatus for cells not registered in the
/// container will return the status GOOD_CELL.
///
/// For visualization a histogram with the cell status as function of column and
/// row can be created on the fly from the bad channel map. As the histogram is
/// reated dynamically from the absolute cell ID an instance of the EMCAL Geometry
/// is required - otherwise an empty histogram is created.
///
/// The bad channel map can be created from multiple bad channel
/// maps using the operator +=. This allows for the combination
/// of the bad channel map from multiple time slices. For cells
/// present in both bad channel maps always the worst case condition
/// (BAD_CELL) is assumed.
class BadChannelMap
{
 public:
  /// \enum MaskType_t
  /// \brief Definition of mask types in the bad channel map
  enum class MaskType_t : char {
    GOOD_CELL, ///< GOOD cell, can be used without problems
    BAD_CELL,  ///< Bad cell, must be excluded
    WARM_CELL, ///< Warm cell, to be used with care
    DEAD_CELL  ///< Dead cell, no data obtained
  };

  /// \brief Constructor
  BadChannelMap() = default;

  /// \brief Destructor
  ~BadChannelMap() = default;

  /// \brief Add bad channel map to this bad channel map
  /// \param rhs Bad channel map to be added to this bad channel map
  /// \return Reference to the combined bad channel map
  ///
  /// Adding bad channels of another bad channel map to this
  /// bad channel map. In case the cell is present in both maps,
  /// the most severe condition is assumed.
  BadChannelMap& operator+=(const BadChannelMap& rhs);

  /// \brief Comparison of two bad channel maps
  /// \return true if the bad channel maps are identical, false otherwise
  ///
  /// Testing two bad channel maps for equalness. Bad channel maps are
  /// considered identical if the content is identical (same channels and
  /// same channel status for all cells).
  bool operator==(const BadChannelMap& other) const;

  /// \brief Add bad cell to the container
  /// \param channelID Absolute ID of the bad channel
  /// \param mask type of the bad channel
  ///
  /// Adding new bad channel to the container. In case a cell
  /// with the same ID is already present in the container,
  /// the mask status is updated. Otherwise it is added.
  ///
  /// Only bad or warm cells are added to the container. In case
  /// the mask type is GOOD_CELL, the entry is removed from the
  /// container if present before, otherwise the cell is ignored.
  void addBadChannel(unsigned short channelID, MaskType_t mask);

  /// \brief Get the status of a certain cell
  /// \param channelID channel for which to obtain the channel status
  /// \return Mask status of the cell (GOOD_CELL if not registered)
  ///
  /// Provide the mask status of a cell. In case the cell is registered
  /// in the container the mask status registered is returned, otherwise
  /// the cell is handled as GOOD_CELL
  MaskType_t getChannelStatus(unsigned short channelID) const;

  /// \brief Convert map into 2D histogram representation
  /// \return Histogram representation of the bad channel map
  ///
  /// Convert bad channel map into a 2D map with col-row dimensions.
  /// Entries in the histogram are:
  /// - 0: GOOD_CELL
  /// - 1: BAD_CELL
  /// - 2: WARM_CELL
  ///
  /// Attention: The histogram is created on the fly using the mapping
  /// in o2::emcal::Geometry. In order to get the representation, a
  /// geometry instance must be available.
  TH2* getHistogramRepresentation() const;

  /// \brief Print bad channels on a given stream
  /// \param stream Stream on which the bad channel map is printed on
  ///
  /// Printing all bad channels store in the bad channel map
  /// on the stream. Printing also the channel status (BAD_CELL/WARM_CELL).
  ///
  /// The function is called in the operator<< providing direct access
  /// to protected members. Explicit calls by the users is normally not
  /// necessary.
  void PrintStream(std::ostream& stream) const;

 private:
  std::bitset<17664> mDeadCells; ///< Container for dead cells (size corresponding to the maximum amount of cells in the EMCAL+DCAL, discarding the PHOS region)
  std::bitset<17664> mBadCells;  ///< Container for bad cells (size corresponding to the maximum amount of cells in EMCAL+DCAL discarding the PHOS region)
  std::bitset<17664> mWarmCells; ///< Contianer for warm cells (size corresponding to the maximum amount of cells in EMCAL+DCAL discarding the PHOS region)

  ClassDefNV(BadChannelMap, 1);
};

/// \brief Printing bad channel map on the stream
/// \param in Stream where the bad channel map is printed on
/// \param bcm Bad channel map to be printed
/// \return Stream after printing the bad channel map
///
/// Printing all bad channels store in the bad channel map
/// on the stream. Printing also the channel status (BAD_CELL/WARM_CELL)
std::ostream& operator<<(std::ostream& in, const BadChannelMap& bcm);

std::ostream& operator<<(std::ostream& in, const BadChannelMap::MaskType_t& masktype);

} // namespace emcal

} // namespace o2
