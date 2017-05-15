#ifndef O2_TPC_RAWREADERARRAY_H_ 
#define O2_TPC_RAWREADERARRAY_H_ 

/// \file RawReaderArray.h
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <map>
#include <string>
#include <utility>

#include "TPCReconstruction/RawReader.h"

namespace o2 {
namespace TPC {

/// \class RawReaderArray
/// \brief Class to hold and manage severall RawReaders
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)
class RawReaderArray {
  public:

    /// Default constructor
    RawReaderArray();

    /// Copy constructor
    RawReaderArray(const RawReaderArray& other) = default;

    /// Default destructor
    ~RawReaderArray() = default;

    /// Add input file for decoding
    /// @param infile Input file string in the format "path_to_file:#region:#fec", where #region/#fec is a number
    /// @return True if string has correct format and file can be opened
    bool addInputFile(std::string infile);

    /// Add several input files for decoding
    /// @param infiles vector of input file strings, each formated as "path_to_file:#region:#fec"
    /// @return True if at least one string has correct format and file can be opened
    bool addInputFile(const std::vector<std::string>* infiles);

    /// Add input file for decoding
    /// @param region Region of the data
    /// @param link FEC of the data
    /// @param path Path to data
    /// @return True file can be opened
    bool addInputFile(int region, int link, std::string path);

  private:

    std::map<std::pair<int,int>, RawReader> mReaders;
};
}
}
#endif
