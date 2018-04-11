#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opengl_backend.h"
#include <GL/glx.h>
#include <pthread.h>
#include <unistd.h>
#include <GL/glxext.h>
pthread_mutex_t semLockDisplay = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t semLockExit = PTHREAD_MUTEX_INITIALIZER;
static volatile bool displayRunning = false;

static GLuint font_base;

Display *g_pDisplay = NULL;
Window g_window;

PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = NULL;

int GetKey(int key)
{
	if (key == 65453 || key == 45) return('-');
	if (key == 65451 || key == 43) return('+');
	if (key == 65505) return(KEY_SHIFT); //Shift
	if (key == 65513) return(KEY_ALT); //ALT
	if (key == 65027) return(KEY_ALT); //R ALT
	if (key == 65507) return(KEY_CTRL); //L CTRL
	if (key == 65508) return(KEY_CTRL); //R CTRL
	if (key == 65362) return(KEY_UP); //UP
	if (key == 65364) return(KEY_DOWN); //DOWN
	if (key == 65361) return(KEY_LEFT); //LEFT
	if (key == 65363) return(KEY_RIGHT); //RIGHT
	if (key == 65365) return(KEY_PAGEUP); //LEFT
	if (key == 65366) return(KEY_PAGEDOWN); //RIGHT
	if (key == 65307) return('Q'); //ESC
	if (key == 32) return(KEY_SPACE); //Space
	if (key > 255) return(0);
	
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	
	return(key);
}

void OpenGLPrint(const char* s)
{
   if (!glIsList(font_base))
   {
      fprintf(stderr, "print string: Bad display list.\n");
      exit (1);
   }
   else if (s && strlen(s))
   {
      glPushAttrib(GL_LIST_BIT);
      glListBase(font_base);
      glCallLists(strlen(s), GL_UNSIGNED_BYTE, (GLubyte*) s);
      glPopAttrib();
   }
}

void *OpenGLMain(void *ptr)
{
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
	
	const char* glxExt = glXQueryExtensionsString(g_pDisplay, DefaultScreen(g_pDisplay));
	if (strstr(glxExt, "GLX_EXT_swap_control") == NULL)
	{
		fprintf(stderr, "No vsync support!\n");
		exit(1);
	}

	int attribs[] =
	{
		GLX_X_RENDERABLE    , True,
		GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
		GLX_RENDER_TYPE     , GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
		GLX_RED_SIZE        , 8,
		GLX_GREEN_SIZE      , 8,
		GLX_BLUE_SIZE       , 8,
		GLX_ALPHA_SIZE      , 8,
		GLX_DEPTH_SIZE      , 24,
		GLX_STENCIL_SIZE    , 8,
		GLX_DOUBLEBUFFER    , True,
		GLX_SAMPLE_BUFFERS  , 1,
		GLX_SAMPLES         , 4,
		None
	};

	GLXFBConfig fbconfig = 0;
	int fbcount;
	GLXFBConfig *fbc = glXChooseFBConfig(g_pDisplay, DefaultScreen(g_pDisplay), attribs, &fbcount);
	if (fbc == NULL || fbcount == 0)
	{
		fprintf(stderr, "Failed to get MSAA GLXFBConfig\n");
		exit(1);
	}
	fbconfig = fbc[0];
	XFree(fbc);
	visualInfo = glXGetVisualFromFBConfig(g_pDisplay, fbconfig);
	
	if (visualInfo == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "no RGB visual with depth buffer");
		exit(1);
	}

	// Create an OpenGL rendering context
	glxContext = glXCreateContext(g_pDisplay, visualInfo, NULL, GL_TRUE);
	if (glxContext == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "could not create rendering context");
		exit(1);
	}
	
	Window win = RootWindow(g_pDisplay, visualInfo->screen);
	colorMap = XCreateColormap(g_pDisplay, win, visualInfo->visual, AllocNone);
	windowAttributes.colormap = colorMap;
	windowAttributes.border_pixel = 0;
	windowAttributes.event_mask = ExposureMask | VisibilityChangeMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask | SubstructureNotifyMask | FocusChangeMask;

	// Create an X window with the selected visual
	g_window = XCreateWindow(g_pDisplay, win,
	                         50, 50, init_width, init_height, // Position / Width and height of window
	                         0, visualInfo->depth, InputOutput, visualInfo->visual, CWBorderPixel | CWColormap | CWEventMask, &windowAttributes);
	XSetStandardProperties(g_pDisplay, g_window, GL_WINDOW_NAME, GL_WINDOW_NAME, None, NULL, 0, NULL);
	glXMakeCurrent(g_pDisplay, g_window, glxContext);
	XMapWindow(g_pDisplay, g_window);
	
	//Maximize window
	XEvent xev;
	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = g_window;
	xev.xclient.message_type = XInternAtom(g_pDisplay, "_NET_WM_STATE", False);
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = 1; //_NET_WM_STATE_ADD
	xev.xclient.data.l[1] = XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
	xev.xclient.data.l[2] = XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_VERT", False);
	XSendEvent(g_pDisplay, DefaultRootWindow(g_pDisplay), False, SubstructureNotifyMask, &xev);
	
	//Receive signal when window closed
	Atom WM_DELETE_WINDOW = XInternAtom(g_pDisplay, "WM_DELETE_WINDOW", False); 
    XSetWMProtocols(g_pDisplay, g_window, &WM_DELETE_WINDOW, 1);
	
	//Prepare fonts
	font_base = glGenLists(256);
    if (!glIsList(font_base))
	{
       fprintf(stderr, "Out of display lists.\n");
       exit(1);
    }
	const char* f = "fixed";
	XFontStruct* font_info = XLoadQueryFont(g_pDisplay, f);
	if (!font_info)
	{
		fprintf(stderr, "XLoadQueryFont failed - Exiting.\n");
		exit(1);
	}
	else
	{
		int first = font_info->min_char_or_byte2;
		int last = font_info->max_char_or_byte2;
		glXUseXFont(font_info->fid, first, last-first+1, font_base+first);
	}

	// Init OpenGL...
	InitGL();
	
	XMapWindow(g_pDisplay, g_window);
	XFlush(g_pDisplay);
	int x11_fd = ConnectionNumber(g_pDisplay);
	
	//Enable vsync
	glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC) glXGetProcAddressARB((const GLubyte*) "glXSwapIntervalEXT");
	if (glXSwapIntervalEXT == NULL)
	{
		fprintf(stderr, "Cannot enable vsync\n");
		exit(1);
	}
	glXSwapIntervalEXT(g_pDisplay, glXGetCurrentDrawable(), 0);
	
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
				fprintf(stderr, "Error\n");
			}
			else if (num_ready_fds > 0) needUpdate = 0;
			if (exitButton == 2) break;
			if (sendKey) needUpdate = 1;
			if (waitCount++ != 100) needUpdate = 1;
		} while (!(num_ready_fds || needUpdate));
		
		do
		{
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
					else
					{
						if (event.xbutton.button == 1)
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
				}
				break;

				case ButtonRelease:
				{
					if (event.xbutton.button != 4 && event.xbutton.button != 5)
					{
						if (event.xbutton.button == 1)
						{
							mouseDn = false;
						}
						else if (event.xbutton.button != 1)
						{
							mouseDnR = false;
						}
					}
				}
				break;

				case KeyPress:
				{
					KeySym sym = XLookupKeysym(&event.xkey, 0);
					int wParam = GetKey(sym);
					//fprintf(stderr, "KeyPress event %d --> %d (%c) -> %d (%c), %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam, (char) wParam, (int) keys[16]);
					keys[wParam] = true;
					keysShift[wParam] = keys[KEY_SHIFT];
				}
				break;

				case KeyRelease:
				{
					KeySym sym = XLookupKeysym(&event.xkey, 0);
					int wParam = GetKey(sym);
					//fprintf(stderr, "KeyRelease event %d -> %d (%c) -> %d (%c), %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam, (char) wParam, (int) keysShift[wParam]);
					HandleKeyRelease(wParam);
					keysShift[wParam] = false;
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
					if (event.xclient.message_type == XInternAtom(g_pDisplay, "_NET_WM_STATE", False))
					{
						XFlush(g_pDisplay);
					}
					else
					{
						exitButton = 2;
					}
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
	XCloseDisplay(g_pDisplay);
	
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

void SwitchFullscreen()
{
	XEvent xev;
	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = g_window;
	xev.xclient.message_type = XInternAtom(g_pDisplay, "_NET_WM_STATE", False);
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = 2; // _NET_WM_STATE_TOGGLE
	xev.xclient.data.l[1] = XInternAtom(g_pDisplay, "_NET_WM_STATE_FULLSCREEN", True);
	xev.xclient.data.l[2] = 0;
	XSendEvent(g_pDisplay, DefaultRootWindow(g_pDisplay), False, SubstructureNotifyMask, &xev);
}

void SetVSync(bool enable)
{
	glXSwapIntervalEXT(g_pDisplay, glXGetCurrentDrawable(), (int) enable);
}
