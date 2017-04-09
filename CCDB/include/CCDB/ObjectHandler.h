#ifndef ALICE_O2_OBJECTHANDLER_H_
#define ALICE_O2_OBJECTHANDLER_H_

#include <string>
#include <vector>

namespace o2 {
namespace CDB {

class ObjectHandler {
public:
  ObjectHandler();
  virtual ~ObjectHandler();

  /// Returns the binary payload of a ROOT file as an std::string
  static void GetObject(const std::string& path, std::string& object);

};
}
}
#endif
