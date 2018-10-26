#include "Framework/DebugGUI.h"
#include "DebugGUI/Sokol3DUtils.h"

using namespace o2::framework;
using namespace o2::framework;

int main(int argc, char** argv)
{
  auto context = initGUI("Foo");
  sokol::init3DContext(context);

  auto callback = []() -> void {
    sokol::render3D();
  };
  while (pollGUI(context, callback)) {
  }
  disposeGUI();
}
