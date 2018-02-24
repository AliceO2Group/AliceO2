// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @since 2016-10-28
/// @author P. Pillot
/// @brief Class defining the serialization/deserialization of preclusters

#ifndef ALICEO2_MCH_PRECLUSTERBLOCK_H_
#define ALICEO2_MCH_PRECLUSTERBLOCK_H_

#include <cstdint>
#include <ostream>
#include <vector>

#include "DigitBlock.h"

namespace o2
{
namespace mch
{

/**
 * Gives the fired channel/pad information which was considered during
 * hit reconstruction and correlated into a common cluster.
 */
struct PreClusterStruct {
  uint16_t nDigits;          // number of digits attached to this precluster
  const DigitStruct* digits; // pointer to the 1st element of the array of digits
};

/**
 * Class to handle the writing/reading of a block of preclusters
 * to/from the provided buffer of the given size.
 */
class PreClusterBlock
{
 public:
  /**
   * Default constructor that produces an object not usable yet.
   * The method reset(...) must be called to initialize it properly.
   */
  PreClusterBlock() = default;

  /**
   * Constructor that sets the internal pointer to the start of the buffer
   * space to read/write to and the total size of the buffer in bytes.
   * @param buffer  The pointer to the first byte of the memory buffer.
   * @param size    The total size of the buffer in bytes.
   * @param write   Tell wether the buffer is for writing or reading.
   */
  PreClusterBlock(void* buffer, uint32_t size, bool write = false);

  /**
   * Destructor to delete internal list of preclusters
   */
  ~PreClusterBlock() = default;

  /**
   * Do not allow copying/moving of this class
   */
  PreClusterBlock(const PreClusterBlock&) = delete;
  PreClusterBlock& operator=(const PreClusterBlock&) = delete;
  PreClusterBlock(PreClusterBlock&&) = delete;
  PreClusterBlock& operator=(PreClusterBlock&&) = delete;

  /**
   * Method that resets the internal pointer to the start of the new buffer
   * space to read/write to and the total size of this buffer in bytes.
   * @param buffer  The pointer to the first byte of the memory buffer.
   * @param size    The total size of the buffer in bytes.
   * @param write   Tell wether the buffer is for writing or reading.
   */
  int reset(void* buffer, uint32_t size, bool write = false);

  /**
   * Method to fill the buffer with a new precluster and its first digit
   * @param digit  Reference to the first digit to add to the new precluster.
   */
  int startPreCluster(const DigitStruct& digit);

  /**
   * Method to add a new digit to the current precluster
   * @param digit  Reference to the digit to add to the current precluster.
   */
  int addDigit(const DigitStruct& digit);

  /**
   * Return the number of bytes currently used for the data block in the buffer.
   */
  uint32_t getCurrentSize() const { return (mCurrentSize > mSize) ? 0 : mCurrentSize; }

  /**
   * Return the number of preclusters currently stored in the data block.
   */
  uint16_t getNPreClusters() const { return (mNPreClusters == nullptr) ? 0 : *mNPreClusters; }

  /**
   * Return the vector of preclusters currently stored in the data block.
   */
  const std::vector<PreClusterStruct>& getPreClusters() const { return mPreClusters; }

  /**
   * Return the total size of the precluster blocks.
   * @param nBlocks       Number of precluster blocks.
   * @param nPreClusters  Total number of preclusters.
   * @param nDigits       Total number of digits.
   */
  static const uint32_t sizeOfPreClusterBlocks(int nBlocks, int nPreClusters, int nDigits)
  {
    return (nBlocks > 0 && nPreClusters > 0 && nDigits > 0)
             ? (nBlocks + nPreClusters) * SSizeOfUShort + nDigits * SSizeOfDigit
             : 0;
  }

 private:
  /// read the buffer
  int readBuffer();

  static constexpr uint32_t SSizeOfUShort = sizeof(uint16_t);
  static constexpr uint32_t SSizeOfDigit = sizeof(DigitStruct);

  /// running pointer on the buffer
  /**
   indicates where the next chunk of data is written in "write" mode,
   or where it is read, in "read" mode
   **/
  void* mBuffer = nullptr;
  uint32_t mSize = 0;                ///< total size of the buffer
  bool mWriteMode = false;           ///< read/write mode
  uint32_t mSize4PreCluster = 0;     ///< total buffer size minus the one resquested to add a new precluster
  uint32_t mSize4Digit = 0;          ///< total buffer size minus the one resquested to add a new digit
  uint32_t mCurrentSize = 0;         ///< size of the used part of the buffer
  uint16_t* mNPreClusters = nullptr; ///< number of preclusters
  uint16_t* mLastNDigits = nullptr;  ///< number of digits in the last precluster (write mode)

  std::vector<PreClusterStruct> mPreClusters{}; ///< list of preclusters
};

/// stream operator for printout
std::ostream& operator<<(std::ostream& stream, const PreClusterStruct& cluster);

/// stream operator for printout
std::ostream& operator<<(std::ostream& stream, const PreClusterBlock& clusterBlock);

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PRECLUSTERBLOCK_H_
