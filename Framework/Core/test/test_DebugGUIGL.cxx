#include "Framework/DebugGUI.h"
#include "DebugGUI/GL3DUtils.h"

using namespace o2::framework;
using namespace o2::framework;

int main(int argc, char** argv)
{
  auto context = initGUI("Foo");
  gl::init3DContext(context);

  auto callback = []() -> void {
    gl::render3D();
  };
  while (pollGUI(context, callback)) {
  }
  disposeGUI();
}
