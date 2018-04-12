#ifndef OPENGL_BACKEND_H
#define OPENGL_BACKEND_H

#include <vector>
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
void SetVSync(bool enable);

#define GL_WINDOW_NAME "Alice HLT TPC CA Event Display"

#define KEY_UP 1
#define KEY_DOWN 2
#define KEY_LEFT 3
#define KEY_RIGHT 4
#define KEY_PAGEUP 5
#define KEY_PAGEDOWN 6
#define KEY_SPACE 13
#define KEY_SHIFT 16
#define KEY_ALT 17
#define KEY_CTRL 18

class opengl_spline
{
public:
	opengl_spline() : fa(), fb(), fc(), fd(), fx() {}
	void create(const std::vector<float>& x, const std::vector<float>& y);
	float evaluate(float x);
	
private:
	std::vector<float> fa, fb, fc, fd, fx;
};

#endif
