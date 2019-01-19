#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "AliGPUCADisplayBackendWindows.h"
#include <windows.h>
#include <winbase.h>
#include <windowsx.h>

HDC hDC = NULL;                                       // Private GDI Device Context
HGLRC hRC = NULL;                                     // Permanent Rendering Context
HWND hWnd = NULL;                                     // Holds Our Window Handle
HINSTANCE hInstance;                                  // Holds The Instance Of The Application
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM); // Declaration For WndProc

bool active = TRUE;     // Window Active Flag Set To TRUE By Default
bool fullscreen = TRUE; // Fullscreen Flag Set To Fullscreen Mode By Default

POINT mouseCursorPos;

volatile int mouseReset = false;

void KillGLWindow() // Properly Kill The Window
{
	if (fullscreen) // Are We In Fullscreen Mode?
	{
		ChangeDisplaySettings(NULL, 0); // If So Switch Back To The Desktop
		ShowCursor(TRUE);               // Show Mouse Pointer
	}

	if (hRC) // Do We Have A Rendering Context?
	{
		if (!wglMakeCurrent(NULL, NULL)) // Are We Able To Release The DC And RC Contexts?
		{
			MessageBox(NULL, "Release Of DC And RC Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		}

		if (!wglDeleteContext(hRC)) // Are We Able To Delete The RC?
		{
			MessageBox(NULL, "Release Rendering Context Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		}
		hRC = NULL; // Set RC To NULL
	}

	if (hDC && !ReleaseDC(hWnd, hDC)) // Are We Able To Release The DC
	{
		MessageBox(NULL, "Release Device Context Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hDC = NULL; // Set DC To NULL
	}

	if (hWnd && !DestroyWindow(hWnd)) // Are We Able To Destroy The Window?
	{
		MessageBox(NULL, "Could Not Release hWnd.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hWnd = NULL; // Set hWnd To NULL
	}

	if (!UnregisterClass("OpenGL", hInstance)) // Are We Able To Unregister Class
	{
		MessageBox(NULL, "Could Not Unregister Class.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hInstance = NULL; // Set hInstance To NULL
	}
}

BOOL CreateGLWindow(char *title, int width, int height, int bits, bool fullscreenflag)
{
	GLuint PixelFormat;                // Holds The Results After Searching For A Match
	WNDCLASS wc;                       // Windows Class Structure
	DWORD dwExStyle;                   // Window Extended Style
	DWORD dwStyle;                     // Window Style
	RECT WindowRect;                   // Grabs Rectangle Upper Left / Lower Right Values
	WindowRect.left = (long) 0;        // Set Left Value To 0
	WindowRect.right = (long) width;   // Set Right Value To Requested Width
	WindowRect.top = (long) 0;         // Set Top Value To 0
	WindowRect.bottom = (long) height; // Set Bottom Value To Requested Height

	fullscreen = fullscreenflag; // Set The Global Fullscreen Flag

	hInstance = GetModuleHandle(NULL);             // Grab An Instance For Our Window
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; // Redraw On Size, And Own DC For Window.
	wc.lpfnWndProc = (WNDPROC) WndProc;            // WndProc Handles Messages
	wc.cbClsExtra = 0;                             // No Extra Window Data
	wc.cbWndExtra = 0;                             // No Extra Window Data
	wc.hInstance = hInstance;                      // Set The Instance
	wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);        // Load The Default Icon
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);      // Load The Arrow Pointer
	wc.hbrBackground = NULL;                       // No Background Required For GL
	wc.lpszMenuName = NULL;                        // We Don't Want A Menu
	wc.lpszClassName = "OpenGL";                   // Set The Class Name

	if (!RegisterClass(&wc)) // Attempt To Register The Window Class
	{
		MessageBox(NULL, "Failed To Register The Window Class.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (fullscreen) // Attempt Fullscreen Mode?
	{
		DEVMODE dmScreenSettings;                               // Device Mode
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings)); // Makes Sure Memory's Cleared
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);     // Size Of The Devmode Structure
		dmScreenSettings.dmPelsWidth = width;                   // Selected Screen Width
		dmScreenSettings.dmPelsHeight = height;                 // Selected Screen Height
		dmScreenSettings.dmBitsPerPel = bits;                   // Selected Bits Per Pixel
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL)
		{
			printf("The Requested Fullscreen Mode Is Not Supported By Your Video Card.\n");
			return(FALSE);
		}

		dwExStyle = WS_EX_APPWINDOW;
		dwStyle = WS_POPUP;
		ShowCursor(FALSE);
	}
	else
	{
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE; // Window Extended Style
		dwStyle = WS_OVERLAPPEDWINDOW;                  // Windows Style
	}

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle); // Adjust Window To True Requested Size

	// Create The Window
	if (!(hWnd = CreateWindowEx(dwExStyle, "OpenGL", title, dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0, 0, WindowRect.right - WindowRect.left, WindowRect.bottom - WindowRect.top, NULL, NULL, hInstance, NULL)))
	{
		KillGLWindow();
		MessageBox(NULL, "Window Creation Error.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}

	static PIXELFORMATDESCRIPTOR pfd = // pfd Tells Windows How We Want Things To Be
	    {
	        sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
	        1,                             // Version Number
	        PFD_DRAW_TO_WINDOW |           // Format Must Support Window
	            PFD_SUPPORT_OPENGL |       // Format Must Support OpenGL
	            PFD_DOUBLEBUFFER,          // Must Support Double Buffering
	        PFD_TYPE_RGBA,                 // Request An RGBA Format
	        (unsigned char) bits,          // Select Our Color Depth
	        0,
	        0, 0, 0, 0, 0,  // Color Bits Ignored
	        0,              // No Alpha Buffer
	        0,              // Shift Bit Ignored
	        0,              // No Accumulation Buffer
	        0, 0, 0, 0,     // Accumulation Bits Ignored
	        16,             // 16Bit Z-Buffer (Depth Buffer)
	        0,              // No Stencil Buffer
	        0,              // No Auxiliary Buffer
	        PFD_MAIN_PLANE, // Main Drawing Layer
	        0,              // Reserved
	        0, 0, 0         // Layer Masks Ignored
	    };

	if (!(hDC = GetDC(hWnd))) // Did We Get A Device Context?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Create A GL Device Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!(PixelFormat = ChoosePixelFormat(hDC, &pfd))) // Did Windows Find A Matching Pixel Format?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Find A Suitable PixelFormat.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!SetPixelFormat(hDC, PixelFormat, &pfd)) // Are We Able To Set The Pixel Format?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Set The PixelFormat.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!(hRC = wglCreateContext(hDC))) // Are We Able To Get A Rendering Context?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Create A GL Rendering Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!wglMakeCurrent(hDC, hRC)) // Try To Activate The Rendering Context
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Activate The GL Rendering Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	ShowWindow(hWnd, SW_SHOW);    // Show The Window
	SetForegroundWindow(hWnd);    // Slightly Higher Priority
	SetFocus(hWnd);               // Sets Keyboard Focus To The Window
	ReSizeGLScene(width, height); // Set Up Our Perspective GL Screen

	if (InitGL()) // Initialize Our Newly Created GL Window
	{
		KillGLWindow(); // Reset The Display
		printf("Initialization Failed.\n");
		return FALSE; // Return FALSE
	}

	return TRUE; // Success
}

int GetKey(int key)
{
	if (key == 107 || key == 187) return('+');
	if (key == 109 || key == 189) return('-');
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	
	return(key);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) // Check For Windows Messages
	{
		case WM_ACTIVATE: // Watch For Window Activate Message
		{
			if (!HIWORD(wParam)) // Check Minimization State
			{
				active = TRUE; // Program Is Active
			}
			else
			{
				active = FALSE; // Program Is No Longer Active
			}

			return 0; // Return To The Message Loop
		}

		case WM_SYSCOMMAND: // Intercept System Commands
		{
			switch (wParam) // Check System Calls
			{
			case SC_SCREENSAVE:   // Screensaver Trying To Start?
			case SC_MONITORPOWER: // Monitor Trying To Enter Powersave?
				return 0;         // Prevent From Happening
			}
			break; // Exit
		}

		case WM_CLOSE: // Did We Receive A Close Message?
		{
			PostQuitMessage(0); // Send A Quit Message
			return 0;           // Jump Back
		}

		case WM_KEYDOWN: // Is A Key Being Held Down?
		{
			wParam = GetKey(wParam);
			keys[wParam] = TRUE; // If So, Mark It As TRUE
			keysShift[wParam] = keys[KEY_SHIFT];
			return 0;            // Jump Back
		}

		case WM_KEYUP: // Has A Key Been Released?
		{
			wParam = GetKey(wParam);
			HandleKeyRelease(wParam);
			keysShift[wParam] = false;

			printf("Key: %d\n", wParam);
			return 0; // Jump Back
		}

		case WM_SIZE: // Resize The OpenGL Window
		{
			ReSizeGLScene(LOWORD(lParam), HIWORD(lParam)); // LoWord=Width, HiWord=Height
			return 0;                                      // Jump Back
		}

		case WM_LBUTTONDOWN:
		{
			mouseDnX = GET_X_LPARAM(lParam);
			mouseDnY = GET_Y_LPARAM(lParam);
			mouseDn = true;
			GetCursorPos(&mouseCursorPos);
			return 0;
		}

		case WM_LBUTTONUP:
		{
			mouseDn = false;
			return 0;
		}

		case WM_RBUTTONDOWN:
		{
			mouseDnX = GET_X_LPARAM(lParam);
			mouseDnY = GET_Y_LPARAM(lParam);
			mouseDnR = true;
			GetCursorPos(&mouseCursorPos);
			return 0;
		}

		case WM_RBUTTONUP:
		{
			mouseDnR = false;
			return 0;
		}

		case WM_MOUSEMOVE:
		{
			if (mouseReset)
			{
				mouseDnX = GET_X_LPARAM(lParam);
				mouseDnY = GET_Y_LPARAM(lParam);
				mouseReset = 0;
			}
			mouseMvX = GET_X_LPARAM(lParam);
			mouseMvY = GET_Y_LPARAM(lParam);
			return 0;
		}

		case WM_MOUSEWHEEL:
		{
			mouseWheel += GET_WHEEL_DELTA_WPARAM(wParam);
			return 0;
		}
	}

	// Pass All Unhandled Messages To DefWindowProc
	return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

DWORD WINAPI OpenGLMain()
{
	MSG msg;           // Windows Message Structure
	BOOL done = FALSE; // Bool Variable To Exit Loop

	// Ask The User Which Screen Mode They Prefer
	fullscreen = FALSE; // Windowed Mode

	// Create Our OpenGL Window
	if (!CreateGLWindow(GL_WINDOW_NAME, init_width, init_height, 32, fullscreen))
	{
		return 0; // Quit If Window Was Not Created
	}

	while (!done) // Loop That Runs While done=FALSE
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) // Is There A Message Waiting?
		{
			if (msg.message == WM_QUIT) // Have We Received A Quit Message?
			{
				done = TRUE; // If So done=TRUE
			}
			else // If Not, Deal With Window Messages
			{
				TranslateMessage(&msg); // Translate The Message
				DispatchMessage(&msg);  // Dispatch The Message
			}
		}
		else // If There Are No Messages
		{
			// Draw The Scene.  Watch For ESC Key And Quit Messages From DrawGLScene()
			if (active) // Program Active?
			{
				if (keys[VK_ESCAPE]) // Was ESC Pressed?
				{
					done = TRUE; // ESC Signalled A Quit
				}
				else // Not Time To Quit, Update Screen
				{
					DrawGLScene(); // Draw The Scene
					SwapBuffers(hDC); // Swap Buffers (Double Buffering)
				}
			}
		}
	}

	// Shutdown
	KillGLWindow();      // Kill The Window
	return (msg.wParam); // Exit The Program
}

void DisplayExit() {}
void OpenGLPrint(const char* s) {}
void SwitchFullscreen(bool set) {}
void ToggleMaximized(bool set) {}
void SetVSync(bool enable) {}

void AliGPUCADisplayBackendWindows::StartDisplay()
{
	HANDLE hThread;
	if ((hThread = CreateThread(NULL, NULL, &OpenGLWrapper, this, NULL, NULL)) == NULL)
	{
		printf("Coult not Create GL Thread...\nExiting...\n");
	}
}
