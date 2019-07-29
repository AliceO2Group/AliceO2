// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendWindows.cxx
/// \author David Rohr

// GLEW must be the first header
#include <GL/glew.h>

// Now the other headers
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "GPUDisplayBackendWindows.h"
#include <windows.h>
#include <winbase.h>
#include <windowsx.h>

using namespace GPUCA_NAMESPACE::gpu;

HDC hDC = nullptr;                                    // Private GDI Device Context
HGLRC hRC = nullptr;                                  // Permanent Rendering Context
HWND hWnd = nullptr;                                  // Holds Our Window Handle
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
    ChangeDisplaySettings(nullptr, 0); // If So Switch Back To The Desktop
    ShowCursor(TRUE);                  // Show Mouse Pointer
  }

  if (hRC) // Do We Have A Rendering Context?
  {
    if (!wglMakeCurrent(nullptr, nullptr)) { // Are We Able To Release The DC And RC Contexts?
      MessageBox(nullptr, "Release Of DC And RC Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
    }

    if (!wglDeleteContext(hRC)) { // Are We Able To Delete The RC?
      MessageBox(nullptr, "Release Rendering Context Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
    }
    hRC = nullptr;
  }

  if (hDC && !ReleaseDC(hWnd, hDC)) // Are We Able To Release The DC
  {
    MessageBox(nullptr, "Release Device Context Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
    hDC = nullptr;
  }

  if (hWnd && !DestroyWindow(hWnd)) // Are We Able To Destroy The Window?
  {
    MessageBox(nullptr, "Could Not Release hWnd.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
    hWnd = nullptr;
  }

  if (!UnregisterClass("OpenGL", hInstance)) // Are We Able To Unregister Class
  {
    MessageBox(nullptr, "Could Not Unregister Class.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
    hInstance = nullptr;
  }
}

BOOL CreateGLWindow(char* title, int width, int height, int bits, bool fullscreenflag)
{
  GLuint PixelFormat;               // Holds The Results After Searching For A Match
  WNDCLASS wc;                      // Windows Class Structure
  DWORD dwExStyle;                  // Window Extended Style
  DWORD dwStyle;                    // Window Style
  RECT WindowRect;                  // Grabs Rectangle Upper Left / Lower Right Values
  WindowRect.left = (long)0;        // Set Left Value To 0
  WindowRect.right = (long)width;   // Set Right Value To Requested Width
  WindowRect.top = (long)0;         // Set Top Value To 0
  WindowRect.bottom = (long)height; // Set Bottom Value To Requested Height

  fullscreen = fullscreenflag; // Set The Global Fullscreen Flag

  hInstance = GetModuleHandle(nullptr);          // Grab An Instance For Our Window
  wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; // Redraw On Size, And Own DC For Window.
  wc.lpfnWndProc = (WNDPROC)WndProc;             // WndProc Handles Messages
  wc.cbClsExtra = 0;                             // No Extra Window Data
  wc.cbWndExtra = 0;                             // No Extra Window Data
  wc.hInstance = hInstance;                      // Set The Instance
  wc.hIcon = LoadIcon(nullptr, IDI_WINLOGO);     // Load The Default Icon
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);   // Load The Arrow Pointer
  wc.hbrBackground = nullptr;                    // No Background Required For GL
  wc.lpszMenuName = nullptr;                     // We Don't Want A Menu
  wc.lpszClassName = "OpenGL";                   // Set The Class Name

  if (!RegisterClass(&wc)) // Attempt To Register The Window Class
  {
    MessageBox(nullptr, "Failed To Register The Window Class.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
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

    if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL) {
      printf("The Requested Fullscreen Mode Is Not Supported By Your Video Card.\n");
      return (FALSE);
    }

    dwExStyle = WS_EX_APPWINDOW;
    dwStyle = WS_POPUP;
    ShowCursor(FALSE);
  } else {
    dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE; // Window Extended Style
    dwStyle = WS_OVERLAPPEDWINDOW;                  // Windows Style
  }

  AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);

  // Create The Window
  if (!(hWnd = CreateWindowEx(dwExStyle, "OpenGL", title, dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0, 0, WindowRect.right - WindowRect.left, WindowRect.bottom - WindowRect.top, nullptr, nullptr, hInstance, nullptr))) {
    KillGLWindow();
    MessageBox(nullptr, "Window Creation Error.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    return FALSE;
  }

  static PIXELFORMATDESCRIPTOR pfd = // pfd Tells Windows How We Want Things To Be
    {
      sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
      1,                             // Version Number
      PFD_DRAW_TO_WINDOW |           // Format Must Support Window
        PFD_SUPPORT_OPENGL |         // Format Must Support OpenGL
        PFD_DOUBLEBUFFER,            // Must Support Double Buffering
      PFD_TYPE_RGBA,                 // Request An RGBA Format
      (unsigned char)bits,           // Select Our Color Depth
      0,
      0,
      0,
      0,
      0,
      0, // Color Bits Ignored
      0, // No Alpha Buffer
      0, // Shift Bit Ignored
      0, // No Accumulation Buffer
      0,
      0,
      0,
      0,              // Accumulation Bits Ignored
      16,             // 16Bit Z-Buffer (Depth Buffer)
      0,              // No Stencil Buffer
      0,              // No Auxiliary Buffer
      PFD_MAIN_PLANE, // Main Drawing Layer
      0,              // Reserved
      0,
      0,
      0 // Layer Masks Ignored
    };

  if (!(hDC = GetDC(hWnd))) // Did We Get A Device Context?
  {
    KillGLWindow();
    MessageBox(nullptr, "Can't Create A GL Device Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    return FALSE;
  }

  if (!(PixelFormat = ChoosePixelFormat(hDC, &pfd))) // Did Windows Find A Matching Pixel Format?
  {
    KillGLWindow();
    MessageBox(nullptr, "Can't Find A Suitable PixelFormat.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    return FALSE;
  }

  if (!SetPixelFormat(hDC, PixelFormat, &pfd)) // Are We Able To Set The Pixel Format?
  {
    KillGLWindow();
    MessageBox(nullptr, "Can't Set The PixelFormat.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    return FALSE;
  }

  if (!(hRC = wglCreateContext(hDC))) // Are We Able To Get A Rendering Context?
  {
    KillGLWindow();
    MessageBox(nullptr, "Can't Create A GL Rendering Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    return FALSE;
  }

  if (!wglMakeCurrent(hDC, hRC)) // Try To Activate The Rendering Context
  {
    KillGLWindow();
    MessageBox(nullptr, "Can't Activate The GL Rendering Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    return FALSE;
  }

  ShowWindow(hWnd, SW_SHOW);
  SetForegroundWindow(hWnd);
  SetFocus(hWnd);
  ReSizeGLScene(width, height);

  return TRUE;
}

int GetKey(int key)
{
  if (key == 107 || key == 187) {
    return ('+');
  }
  if (key == 109 || key == 189) {
    return ('-');
  }
  if (key >= 'a' && key <= 'z') {
    key += 'A' - 'a';
  }

  return (key);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  switch (uMsg) // Check For Windows Messages
  {
    case WM_ACTIVATE: // Watch For Window Activate Message, check minimization state
      if (!HIWORD(wParam)) {
        active = TRUE;
      } else {
        active = FALSE;
      }
      return 0;

    case WM_SYSCOMMAND: // Intercept System Commands
      switch (wParam)   // Check System Calls
      {
        case SC_SCREENSAVE:   // Screensaver Trying To Start?
        case SC_MONITORPOWER: // Monitor Trying To Enter Powersave?
          return 0;           // Prevent From Happening
      }
      break; // Exit

    case WM_CLOSE: // Did We Receive A Close Message?
      PostQuitMessage(0);
      return 0;

    case WM_KEYDOWN: // Is A Key Being Held Down?
      wParam = GetKey(wParam);
      mKeys[wParam] = TRUE;
      mKeysShift[wParam] = mKeys[KEY_SHIFT];
      return 0;

    case WM_KEYUP:
      wParam = GetKey(wParam);
      HandleKeyRelease(wParam);
      mKeysShift[wParam] = false;

      printf("Key: %d\n", wParam);
      return 0;

    case WM_SIZE:
      ReSizeGLScene(LOWORD(lParam), HIWORD(lParam)); // LoWord=Width, HiWord=Height
      return 0;

    case WM_LBUTTONDOWN:
      mMouseDnX = GET_X_LPARAM(lParam);
      mMouseDnY = GET_Y_LPARAM(lParam);
      mMouseDn = true;
      GetCursorPos(&mouseCursorPos);
      return 0;

    case WM_LBUTTONUP:
      mMouseDn = false;
      return 0;

    case WM_RBUTTONDOWN:
      mMouseDnX = GET_X_LPARAM(lParam);
      mMouseDnY = GET_Y_LPARAM(lParam);
      mMouseDnR = true;
      GetCursorPos(&mouseCursorPos);
      return 0;

    case WM_RBUTTONUP:
      mMouseDnR = false;
      return 0;

    case WM_MOUSEMOVE:
      if (mouseReset) {
        mMouseDnX = GET_X_LPARAM(lParam);
        mMouseDnY = GET_Y_LPARAM(lParam);
        mouseReset = 0;
      }
      mouseMvX = GET_X_LPARAM(lParam);
      mouseMvY = GET_Y_LPARAM(lParam);
      return 0;

    case WM_MOUSEWHEEL:
      mMouseWheel += GET_WHEEL_DELTA_WPARAM(wParam);
      return 0;
  }

  // Pass All Unhandled Messages To DefWindowProc
  return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

int GPUDisplayBackendWindows::OpenGLMain()
{
  MSG msg;
  BOOL done = FALSE;
  fullscreen = FALSE;

  if (glewInit()) {
    return (-1);
  }

  if (!CreateGLWindow(GL_WINDOW_NAME, INIT_WIDTH, INIT_HEIGHT, 32, fullscreen)) {
    return -1;
  }

  if (InitGL()) {
    KillGLWindow();
    printf("Initialization Failed.\n");
    return 1;
  }

  while (!done) {
    if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) // Is There A Message Waiting?
    {
      if (msg.message == WM_QUIT) {
        done = TRUE;
      } else {
        TranslateMessage(&msg); // Translate The Message
        DispatchMessage(&msg);  // Dispatch The Message
      }
    } else {
      if (active) // Program Active?
      {
        if (mKeys[VK_ESCAPE]) {
          done = TRUE;
        } else {
          DrawGLScene();    // Draw The Scene
          SwapBuffers(hDC); // Swap Buffers (Double Buffering)
        }
      }
    }
  }

  // Shutdown
  KillGLWindow();
  return (0);
}

void DisplayExit() {}
void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton) {}
void SwitchFullscreen(bool set) {}
void ToggleMaximized(bool set) {}
void SetVSync(bool enable) {}

int GPUDisplayBackendWindows::StartDisplay()
{
  HANDLE hThread;
  if ((hThread = CreateThread(nullptr, nullptr, &OpenGLWrapper, this, nullptr, nullptr)) == nullptr) {
    printf("Coult not Create GL Thread...\n");
    return (1);
  }
  return (0);
}
