// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_PHOS_PHOSBADCHANNELSMAP_H_
#define O2_PHOS_PHOSBADCHANNELSMAP_H_

#include <iosfwd>
#include <bitset>
#include <Rtypes.h>
#include <CCDB/TObjectWrapper.h> // Needed to trigger dictionary build

class TH2;

namespace o2
{

namespace phos
{

/// \class BadChannelMap
/// \brief CCDB container for bad (masked) channels in PHOS
/// \author Dmitri Peresunko, adopted from EMCAL (Markus Fasel)
/// \since Aug. 1, 2019
///
/// # The PHOS Bad Channel Map
///
/// The bad channel map contains channels which are marked to be
/// bad and excluded from reonstruction: clusterization
/// or other analysis processes.
///
/// Bad channels can be added via
/// bcm.addBadChannel(1234);
/// goodness of channel can be restored with
/// bcm.setChannelGood(1234) ;
///
/// Reading the channel status is done via
/// bool status = bcm.isChannelGood(1234);
/// Calling isChannelGood for cells beyond PHOS
/// will return the status bad.
///
/// For visualization a 2D histogram with the cell status as function of x(phi) vs z
/// for each module can be created on the fly from the bad channel map. As the histogram is
/// created dynamically from the absolute cell ID an instance of the PHOS Geometry
/// is required - otherwise an empty histogram is created.
///
/// The bad channel map can be created from multiple bad channel
/// maps using the operator +=. This allows for the combination
/// of the bad channel map from multiple time slices.
class BadChannelMap
{
 public:
  /// \brief Constructor
  BadChannelMap() = default;

  /// \brief Constructur used to build test bad map
  BadChannelMap(int test);

  /// \brief Destructor
  ~BadChannelMap() = default;

  /// \brief Add bad channel map to this bad channel map
  /// \param rhs Bad channel map to be added to this bad channel map
  /// \return Reference to the combined bad channel map
  ///
  /// Adding bad channels of another bad channel map to this
  /// bad channel map.
  BadChannelMap& operator+=(const BadChannelMap& rhs)
  {
    mBadCells |= rhs.mBadCells;
    return *this;
  }

  /// \brief Comparison of two bad channel maps
  /// \return true if the bad channel maps are identical, false otherwise
  ///
  /// Testing two bad channel maps for equalness.
  bool operator==(const BadChannelMap& other) const { return mBadCells == other.mBadCells; }

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
  void addBadChannel(short channelID) { mBadCells.set(channelID - OFFSET); } //set bit to true

  /// \brief Mark channel as good
  /// \param channelID Absolute ID of the channel
  ///
  /// Setting channel as good.
  void setChannelGood(short channelID) { mBadCells.set(channelID - OFFSET, false); }

  /// \brief Get the status of a certain cell
  /// \param channelID channel for which to obtain the channel status
  /// \return true if good channel
  ///
  /// Provide the mask status of a cell.
  bool isChannelGood(short channelID) const { return !mBadCells.test(channelID - OFFSET); }

  /// \brief Convert map into 2D histogram representation
  /// \param mod Module number
  /// \param h Histogram of size 64*56 to be filled with the bad channel map.
  ///
  /// Convert bad channel map into a 2D map with phi(64) vs z(56) dimensions.
  /// Entries in the histogram are:
  /// - 0: GOOD_CELL
  /// - 1: BAD_CELL
  /// Attention: It is responsibility of user to create/delete histogram
  void getHistogramRepresentation(char mod, TH2* h) const;

  /// \brief Print bad channels on a given stream
  /// \param stream Stream on which the bad channel map is printed on
  ///
  /// Printing all bad channels store in the bad channel map
  /// on the stream.
  ///
  /// The function is called in the operator<< providing direct access
  /// to protected members. Explicit calls by the users is normally not
  /// necessary.
  void PrintStream(std::ostream& stream) const;

 private:
  static constexpr short NCHANNELS = 14337; ///< Number of channels starting from 1 (4*64*56+1
  static constexpr short OFFSET = 5377;     ///< Non-existing channels 56*64*1.5+1
  std::bitset<NCHANNELS> mBadCells;         ///< Container for bad cells, 1 means bad sell

  ClassDefNV(BadChannelMap, 1);
};

/// \brief Printing bad channel map on the stream
/// \param in Stream where the bad channel map is printed on
/// \param bcm Bad channel map to be printed
/// \return Stream after printing the bad channel map
///
/// Printing cellID of all bad channels stored in the bad channel map
/// on the stream.
std::ostream& operator<<(std::ostream& in, const BadChannelMap& bcm);

} // namespace phos

} // namespace o2

#endif /* O2_PHOS_PHOSBADCHANNELSMAP_H_ */
