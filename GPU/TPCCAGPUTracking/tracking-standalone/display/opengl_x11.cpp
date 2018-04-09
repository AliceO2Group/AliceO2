#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opengl_backend.h"
#include <GL/glx.h> // This includes the necessary X headers
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t semLockDisplay = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t semLockExit = PTHREAD_MUTEX_INITIALIZER;
static volatile bool displayRunning = false;

int GetKey(int key)
{
	if (key == 65453 || key == 45) return('-');
	if (key == 65451 || key == 43) return('+');
	if (key == 65505) return(16); //Shift
	if (key == 65307) return('Q'); //ESC
	if (key == 32) return(13); //Space
	if (key > 255) return(0);
	
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	
	return(key);
}

void *OpenGLMain(void *ptr)
{
	Display *g_pDisplay = NULL;
	Window g_window;
	
	XSetWindowAttributes windowAttributes;
	XVisualInfo *visualInfo = NULL;
	XEvent event;
	Colormap colorMap;
	GLXContext glxContext;
	int errorBase;
	int eventBase;

	// Open a connection to the X server
	g_pDisplay = XOpenDisplay(NULL);

	if (g_pDisplay == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "could not open display");
		exit(1);
	}

	// Make sure OpenGL's GLX extension supported
	if (!glXQueryExtension(g_pDisplay, &errorBase, &eventBase))
	{
		fprintf(stderr, "glxsimple: %s\n", "X server has no OpenGL GLX extension");
		exit(1);
	}

	// Find an appropriate visual

	int doubleBufferVisual[] =
	    {
	        GLX_RGBA,           // Needs to support OpenGL
	        GLX_DEPTH_SIZE, 16, // Needs to support a 16 bit depth buffer
	        GLX_DOUBLEBUFFER,   // Needs to support double-buffering
	        None                // end of list
	    };

	// Try for the double-bufferd visual first
	visualInfo = glXChooseVisual(g_pDisplay, DefaultScreen(g_pDisplay), doubleBufferVisual);
	if (visualInfo == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "no RGB visual with depth buffer");
		exit(1);
	}

	// Create an OpenGL rendering context
	glxContext = glXCreateContext(g_pDisplay,
	                              visualInfo,
	                              NULL,     // No sharing of display lists
	                              GL_TRUE); // Direct rendering if possible

	if (glxContext == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "could not create rendering context");
		exit(1);
	}
	
	Window win = RootWindow(g_pDisplay, visualInfo->screen);

	// Create an X colormap since we're probably not using the default visual
	colorMap = XCreateColormap(g_pDisplay,
	                           win,
	                           visualInfo->visual,
	                           AllocNone);

	windowAttributes.colormap = colorMap;
	windowAttributes.border_pixel = 0;
	windowAttributes.event_mask = ExposureMask |
	                              VisibilityChangeMask |
	                              KeyPressMask |
	                              KeyReleaseMask |
	                              ButtonPressMask |
	                              ButtonReleaseMask |
	                              PointerMotionMask |
	                              StructureNotifyMask |
	                              SubstructureNotifyMask |
	                              FocusChangeMask;

	// Create an X window with the selected visual
	g_window = XCreateWindow(g_pDisplay,
	                         win,
	                         50, 50,     // x/y position of top-left outside corner of the window
	                         init_width, init_height, // Width and height of window
	                         0,        // Border width
	                         visualInfo->depth,
	                         InputOutput,
	                         visualInfo->visual,
	                         CWBorderPixel | CWColormap | CWEventMask,
	                         &windowAttributes);

	XSetStandardProperties(g_pDisplay,
	                       g_window,
	                       "AliHLTTPCCA Online Event Display",
	                       "AliHLTTPCCA Online Event Display",
	                       None,
	                       NULL,
	                       0,
	                       NULL);

	// Bind the rendering context to the window
	glXMakeCurrent(g_pDisplay, g_window, glxContext);

	// Request the X window to be displayed on the screen
	XMapWindow(g_pDisplay, g_window);
	
	XEvent xev;
	Atom wm_state  =  XInternAtom(g_pDisplay, "_NET_WM_STATE", False);
	Atom max_horz  =  XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
	Atom max_vert  =  XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_VERT", False);
	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = g_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = 1; //_NET_WM_STATE_ADD
	xev.xclient.data.l[1] = max_horz;
	xev.xclient.data.l[2] = max_vert;
	XSendEvent(g_pDisplay, DefaultRootWindow(g_pDisplay), False, SubstructureNotifyMask, &xev);
	
	Atom WM_DELETE_WINDOW = XInternAtom(g_pDisplay, "WM_DELETE_WINDOW", False); 
    XSetWMProtocols(g_pDisplay, g_window, &WM_DELETE_WINDOW, 1);

	// Init OpenGL...
	InitGL();

	// Enter the render loop and don't forget to dispatch X events as they occur.
	
	XMapWindow(g_pDisplay, g_window);
	XFlush(g_pDisplay);
	int x11_fd = ConnectionNumber(g_pDisplay);
	
	pthread_mutex_lock(&semLockExit);
	displayRunning = true;
	pthread_mutex_unlock(&semLockExit);

	while (1)
	{
		int num_ready_fds;
		struct timeval tv;
		fd_set in_fds;
		int waitCount = 0;
		do
		{
			FD_ZERO(&in_fds);
			FD_SET(x11_fd, &in_fds);
			tv.tv_usec = 10000;
			tv.tv_sec = 0;
			num_ready_fds = XPending(g_pDisplay) || select(x11_fd + 1, &in_fds, NULL, NULL, &tv);
			if (num_ready_fds < 0)
			{
				printf("Error\n");
			}
			else if (num_ready_fds > 0) needUpdate = 0;
			if (exitButton == 2) break;
			if (sendKey) needUpdate = 1;
			if (waitCount++ != 100) needUpdate = 1;
		} while (!(num_ready_fds || needUpdate));
		
		do
		{
			//XNextEvent(g_pDisplay, &event);
			if (exitButton == 2) break;
			if (needUpdate)
			{
				needUpdate = 0;
				event.type = Expose;
			}
			else
			{
				XNextEvent(g_pDisplay, &event);
			}
			
			switch (event.type)
			{
				case ButtonPress:
				{
					if (event.xbutton.button == 4)
					{
						mouseWheel += 100;
					}
					else if (event.xbutton.button == 5)
					{
						mouseWheel -= 100;
					}
					else if (event.xbutton.button == 1)
					{
						mouseDn = true;
					}
					else if (event.xbutton.button != 1)
					{
						mouseDnR = true;
					}
					mouseDnX = event.xmotion.x;
					mouseDnY = event.xmotion.y;
				}
				break;

				case ButtonRelease:
				{
					if (event.xbutton.button == 1)
					{
						mouseDn = false;
					}
					if (event.xbutton.button != 1)
					{
						mouseDnR = false;
					}
				}
				break;

				case KeyPress:
				{
					KeySym sym = XLookupKeysym(&event.xkey, 0);
					int wParam = GetKey(sym);
					//fprintf(stderr, "KeyPress event %d --> %d (%c) -> %d (%c), %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam, (char) wParam, (int) keys[16]);
					keys[wParam] = true;
					keysShift[wParam] = keys[16];
				}
				break;

				case KeyRelease:
				{
					KeySym sym = XLookupKeysym(&event.xkey, 0);
					int wParam = GetKey(sym);
					//fprintf(stderr, "KeyRelease event %d -> %d (%c) -> %d (%c), %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam, (char) wParam, (int) keysShift[wParam]);
					HandleKeyRelease(wParam);
					keysShift[wParam] = false;
					//if (updateDLList) printf("Need update!!\n");
				}
				break;

				case MotionNotify:
				{
					mouseMvX = event.xmotion.x;
					mouseMvY = event.xmotion.y;
				}
				break;

				case Expose:
				{
				}
				break;

				case ConfigureNotify:
				{
					glViewport(0, 0, event.xconfigure.width, event.xconfigure.height);
					ReSizeGLScene(event.xconfigure.width, event.xconfigure.height);
				}
				break;
				
				case ClientMessage:
				{
					exitButton = 2;
				}
				break;
			}
			
			HandleSendKey();
		} while (XPending(g_pDisplay)); // Loop to compress events
		if (exitButton == 2) break;

		DrawGLScene();
		glXSwapBuffers(g_pDisplay, g_window); // Buffer swap does implicit glFlush
	}
	
	glXDestroyContext(g_pDisplay, glxContext);
	XDestroyWindow(g_pDisplay, g_window);
	
	pthread_mutex_lock(&semLockExit);
	displayRunning = false;
	pthread_mutex_unlock(&semLockExit);
		
	return(NULL);
}

void DisplayExit()
{
	pthread_mutex_lock(&semLockExit);
	if (displayRunning) exitButton = 2;
	pthread_mutex_unlock(&semLockExit);
	while (displayRunning) usleep(10000);
}
