// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DatDecoder.cxx
/// \author  Andrea Ferrero
///
/// \brief Implementation of a data processor to run the raw decoding
///

#include <functional>
#include "MCHBase/Digit.h"

#define MCH_MERGER_FEEID_MAX 63

namespace o2
{
namespace mch
{
namespace raw
{

//_________________________________________________________________
struct MergerBuffer {
  std::vector<std::pair<Digit,bool>> digits;
  uint32_t orbit = {0};
};

//_________________________________________________________________
class FeeIdMerger
{
  MergerBuffer currentBuffer, previousBuffer;
  int feeId = {0};

  std::function<void(const Digit&)> sendDigit;

 public:
  FeeIdMerger() = default;
  ~FeeIdMerger() = default;

  void setId(int id)
  {
    feeId = id;
  }

  void setDigitHandler(std::function<void(const Digit&)> h)
  {
    sendDigit = h;
  }

  void setOrbit(uint32_t orbit, bool stop);

  MergerBuffer& getCurrentBuffer()
  {
    return currentBuffer;
  }

  MergerBuffer& getPreviousBuffer()
  {
    return previousBuffer;
  }

  void mergeDigits();
};

//_________________________________________________________________
class BaseMerger
{
 public:
  BaseMerger() = default;
  ~BaseMerger() = default;

  virtual void setDigitHandler(std::function<void(const Digit&)> h) = 0;
  virtual void setOrbit(int feeId, uint32_t orbit, bool stop) = 0;
  virtual void addDigit(int feeId, int solarId, int dsAddr, int chAddr,
                        int deId, int padId, unsigned long adc, Digit::Time time, uint16_t nSamples) = 0;
  virtual void mergeDigits(int feeId) = 0;
};

class NoOpMerger : public BaseMerger
{
  std::function<void(const Digit&)> sendDigit;

 public:
  NoOpMerger() = default;
  ~NoOpMerger() = default;

  void setOrbit(int feeId, uint32_t orbit, bool stop)
  {
  }

  void setDigitHandler(std::function<void(const Digit&)> h)
  {
    sendDigit = h;
  }

  void addDigit(int feeId, int solarId, int dsAddr, int chAddr,
                int deId, int padId, unsigned long adc, Digit::Time time, uint16_t nSamples)
  {
    sendDigit(o2::mch::Digit(deId, padId, adc, time, nSamples));
  }

  void mergeDigits(int feeId) {}
};

//_________________________________________________________________
class Merger : public BaseMerger
{
  int32_t feeId = {-1};
  FeeIdMerger mergers[MCH_MERGER_FEEID_MAX + 1];

 public:
  Merger() = default;
  ~Merger() = default;

  void setOrbit(int feeId, uint32_t orbit, bool stop);

  void setDigitHandler(std::function<void(const Digit&)> h);

  void addDigit(int feeId, int solarId, int dsAddr, int chAddr,
                int deId, int padId, unsigned long adc, Digit::Time time, uint16_t nSamples);

  void mergeDigits(int feeId);
};

} // namespace raw
} // namespace mch
} // end namespace o2
