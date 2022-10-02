#ifndef O2_DIMESSAGES_H
#define O2_DIMESSAGES_H

#include <string>
#include <cstdint>
#include <vector>
#include "boost/serialization/vector.hpp"

namespace DIMessages
{
struct RegisterDevice
{
  std::string name;
  std::string analysisId;

  struct Specs
  {
    struct Input
    {
      std::string binding;
      std::string sourceChannel;
      size_t timeslice;

      bool dataDescriptorMatcher;
      std::string origin;
      std::string description;
      uint32_t subSpec;

      template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
      {
        ar & binding;
        ar & sourceChannel;
        ar & timeslice;
        ar & dataDescriptorMatcher;
        ar & origin;
        ar & description;
        ar & subSpec;
      }
    };

    struct Output
    {
      std::string binding;
      std::string channel;
      size_t timeslice;
      size_t maxTimeslices;

      std::string origin;
      std::string description;
      uint32_t subSpec;

      template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
      {
        ar & binding;
        ar & channel;
        ar & timeslice;
        ar & maxTimeslices;
        ar & origin;
        ar & description;
        ar & subSpec;
      }
    };

    struct Forward
    {
      std::string binding;
      size_t timeslice;
      size_t maxTimeslices;
      std::string channel;

      bool dataDescriptorMatcher;
      std::string origin;
      std::string description;
      uint32_t subSpec;

      template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
      {
        ar & binding;
        ar & timeslice;
        ar & maxTimeslices;
        ar & channel;
        ar & dataDescriptorMatcher;
        ar & origin;
        ar & description;
        ar & subSpec;
      }
    };

    std::vector<Input> inputs;
    std::vector<Output> outputs;
    std::vector<Forward> forwards;

    size_t rank;
    size_t nSlots;
    size_t inputTimesliceId;
    size_t maxInputTimeslices;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & inputs;
      ar & outputs;
      ar & forwards;
      ar & rank;
      ar & nSlots;
      ar & inputTimesliceId;
      ar & maxInputTimeslices;
    }
  };

  Specs specs;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & name;
    ar & analysisId;
    ar & specs;
  }
};
}

#endif // O2_DIMESSAGES_H
