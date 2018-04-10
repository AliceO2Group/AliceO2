#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>

extern const int init_width, init_height;
extern volatile int needUpdate;
extern volatile int exitButton;
extern volatile int sendKey;
extern float mouseDnX, mouseDnY;
extern float mouseMvX, mouseMvY;
extern bool mouseDn, mouseDnR;
extern int mouseWheel;
extern bool keys[256];
extern bool keysShift[256];

extern int InitGL();
extern void HandleKeyRelease(int wParam);
extern int DrawGLScene(bool doAnimation = false);
void OpenGLPrint(const char* s);
extern void ReSizeGLScene(int width, int height);
void HandleSendKey();
void SwitchFullscreen();

#define GL_WINDOW_NAME "Alice HLT TPC CA Event Display"
