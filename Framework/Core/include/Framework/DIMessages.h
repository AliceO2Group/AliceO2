#ifndef O2_DIMESSAGES_H
#define O2_DIMESSAGES_H

#include <string>
#include <cstdint>
#include <vector>
#include <boost/optional.hpp>

namespace DIMessages
{
struct RegisterDevice
{
  struct Specs
  {
    struct Input
    {
      std::string binding;
      std::string sourceChannel;
      size_t timeslice;

      boost::optional<std::string> origin;
      boost::optional<std::string> description;
      boost::optional<uint32_t> subSpec;
    };

    struct Output
    {
      std::string binding;
      std::string channel;
      size_t timeslice;
      size_t maxTimeslices;

      std::string origin;
      std::string description;
      boost::optional<uint32_t> subSpec;
    };

    struct Forward
    {
      std::string binding;
      size_t timeslice;
      size_t maxTimeslices;
      std::string channel;

      boost::optional<std::string> origin;
      boost::optional<std::string> description;
      boost::optional<uint32_t> subSpec;
    };

    std::vector<Input> inputs;
    std::vector<Output> outputs;
    std::vector<Forward> forwards;

    size_t rank;
    size_t nSlots;
    size_t inputTimesliceId;
    size_t maxInputTimeslices;
  };

  std::string name;
  std::string runId;
  Specs specs;

  std::string toJson();
};
}

#endif // O2_DIMESSAGES_H
