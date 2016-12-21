#ifndef ALICE_O2_OBJECTHANDLER_H_
#define ALICE_O2_OBJECTHANDLER_H_

#include <string>
#include <vector>

using namespace std;

namespace AliceO2 {
namespace CDB {

class ObjectHandler {
public:
  ObjectHandler();
  virtual ~ObjectHandler();

  /// Returns the binary payload of a ROOT file as an std::string
  void GetObject(const std::string& path, std::string& object);

  /// Compresses uncompressed_string in to compressed_string using zlib
  void Compress(const std::string& uncompressed_string, std::string& compressed_string);
  void Decompress(std::string& uncompressed_string, const std::string& compressed_string);
};
}
}
#endif
