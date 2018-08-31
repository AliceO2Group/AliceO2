#ifndef OPENGL_BACKEND_H
#define OPENGL_BACKEND_H

#include "../cmodules/vecpod.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>

#if !defined(GL_VERSION_4_6) || GL_VERSION_4_6 != 1
#error Unsupported OpenGL version < 4.6
#endif

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
extern int maxFPSRate;

extern int InitGL();
extern void ExitGL();
extern void HandleKeyRelease(int wParam, char character);
extern int DrawGLScene(bool mixAnimation = false, float animateTime = -1.f);
void OpenGLPrint(const char* s);
extern void ReSizeGLScene(int width, int height, bool init = false);
void HandleSendKey();
void SwitchFullscreen();
void ToggleMaximized(bool set = false);
void SetVSync(bool enable);
void createQuaternionFromMatrix(float* v, const float* mat);
void DoScreenshot(char *filename, float animateTime = -1.f);

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
	void create(const vecpod<float>& x, const vecpod<float>& y);
	float evaluate(float x);
	void setVerbose() {verbose = true;}
	
private:
	vecpod<float> fa, fb, fc, fd, fx;
	bool verbose = false;
};

struct OpenGLConfig
{
	int animationMode = 0;
	
	bool smoothPoints = true;
	bool smoothLines = false;
	bool depthBuffer = false;

	int drawClusters = true;
	int drawLinks = false;
	int drawSeeds = false;
	int drawInitLinks = false;
	int drawTracklets = false;
	int drawTracks = false;
	int drawGlobalTracks = false;
	int drawFinal = false;
	int excludeClusters = 0;
	int propagateTracks = 0;

	int colorClusters = 1;
	int drawSlice = -1;
	int drawRelatedSlices = 0;
	int drawGrid = 0;
	int colorCollisions = 0;
	int showCollision = -1;

	float pointSize = 2.0;
	float lineWidth = 1.4;
};
extern OpenGLConfig cfg;

#endif
