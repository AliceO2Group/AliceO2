#ifndef O2_TRK_VDSENSORREGISTRY_H
#define O2_TRK_VDSENSORREGISTRY_H

#include <string>
#include <vector>

namespace o2::trk
{

struct VDSensorDesc {
  enum class Kind { Barrel,
                    Disk };
  std::string name; // sensor volume name
  int petal = -1;
  Kind kind = Kind::Barrel;
  int idx = -1; // layer or disk index
};

// Accessor (defined in VDGeometryBuilder.cxx)
std::vector<VDSensorDesc>& vdSensorRegistry();

// Utilities (defined in VDGeometryBuilder.cxx)
void clearVDSensorRegistry();
void registerSensor(const std::string& volName, int petal, VDSensorDesc::Kind kind, int idx);

} // namespace o2::trk
#endif
