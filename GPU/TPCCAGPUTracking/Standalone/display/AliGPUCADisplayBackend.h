#ifndef ALIGPUCADISPLAYBACKEND_H
#define ALIGPUCADISPLAYBACKEND_H

class AliGPUReconstruction;
class AliGPUCADisplay;

class AliGPUCADisplayBackend
{
	friend class AliGPUCADisplay;
public:
	AliGPUCADisplayBackend() = default;
	~AliGPUCADisplayBackend() = default;
	
	virtual void StartDisplay() = 0; //Start the display. This function returns, and should spawn a thread that runs the display, and calls InitGL
	virtual void DisplayExit() = 0; //Stop the display. Display thread should call ExitGL and the function returns after the thread has terminated
	virtual void SwitchFullscreen() = 0; //Toggle full-screen mode
	virtual void ToggleMaximized(bool set = false) = 0; //Maximize window
	virtual void SetVSync(bool enable) = 0; //Enable / disable vsync
	
	virtual void OpenGLPrint(const char* s) = 0; //Print text on the display (needs the backend to build the font)

	//volatile variables to exchange control informations between display and backend
	volatile int displayControl = 0; //Control for next event (=1) or quit (=2)
	volatile int sendKey = 0; //Key sent by external entity (usually console), may be ignored by backend.
	volatile int needUpdate = 0; //flag that backend shall update the GL content, and call DrawGLScene

protected:
	virtual void* OpenGLMain() = 0;
	static void* OpenGLWrapper(void*);
	
	static constexpr int init_width = 1024, init_height = 768; //Initial window size, before maximizing
	
	static constexpr const char* GL_WINDOW_NAME = "Alice HLT TPC CA Event Display";
	static constexpr int KEY_UP = 1;
	static constexpr int KEY_DOWN = 2;
	static constexpr int KEY_LEFT = 3;
	static constexpr int KEY_RIGHT = 4;
	static constexpr int KEY_PAGEUP = 5;
	static constexpr int KEY_PAGEDOWN = 6;
	static constexpr int KEY_SPACE = 13;
	static constexpr int KEY_SHIFT = 16;
	static constexpr int KEY_ALT = 17;
	static constexpr int KEY_CTRL = 18;

	//Keyboard / Mouse actions
	float mouseDnX, mouseDnY;
	float mouseMvX, mouseMvY;
	bool mouseDn = false;
	bool mouseDnR = false;
	int mouseWheel = 0;
	bool keys[256] = {false}; //Array of keys currently pressed
	bool keysShift[256] = {false}; //Array whether shift was held during key-press
	
	int maxFPSRate; //run at highest possible frame rate, do not sleep in between frames
	
	AliGPUCADisplay* mDisplay; //Ptr to display, not owning, set by display when it connects to backend
	
	void HandleKeyRelease(int wParam, char key); //Callback for handling key presses
	int DrawGLScene(bool mixAnimation = false, float animateTime = -1.f); //Callback to draw the GL scene
	void HandleSendKey(); //Optional callback to handle sendKey variable
	void ReSizeGLScene(int width, int height); //Callback when GL window is resized
	int InitGL(); //Callback to initialize the GL Display (to be called in StartDisplay)
	void ExitGL(); //Callback to clean up the GL Display
};

#endif
