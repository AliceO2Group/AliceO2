/**
 * EPNReceiver.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#ifndef ALICEO2_DEVICES_EPNRECEIVER_H_
#define ALICEO2_DEVICES_EPNRECEIVER_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

/// Container for (sub-)timeframes

struct TFBuffer
{
  FairMQParts parts;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
};

/// Receives sub-timeframes from the flpSenders and merges these into full timeframes.

class EPNReceiver : public FairMQDevice
{
  public:
    /// Default constructor
    EPNReceiver();

    /// Default destructor
    virtual ~EPNReceiver();

    void InitTask();

    /// Prints the contents of the timeframe container
    void PrintBuffer(const std::unordered_map<uint16_t, TFBuffer> &buffer) const;

    /// Discared incomplete timeframes after \p fBufferTimeoutInMs.
    void DiscardIncompleteTimeframes();

  protected:
    /// Overloads the Run() method of FairMQDevice
    virtual void Run();

    std::unordered_map<uint16_t, TFBuffer> fTimeframeBuffer; ///< Stores (sub-)timeframes
    std::unordered_set<uint16_t> fDiscardedSet; ///< Set containing IDs of dropped timeframes

    int fNumFLPs; ///< Number of flpSenders
    int fBufferTimeoutInMs; ///< Time after which incomplete timeframes are dropped
    int fTestMode; ///< Run the device in test mode (only syncSampler+flpSender+epnReceiver)
};

} // namespace Devices
} // namespace AliceO2

#endif
