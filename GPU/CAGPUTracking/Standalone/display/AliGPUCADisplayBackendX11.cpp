#include <GL/glew.h>
#include "AliGPUCADisplayBackendX11.h"

int AliGPUCADisplayBackendX11::GetKey(int key)
{
	if (key == 65453) return('-');
	if (key == 65451) return('+');
	if (key == 65505 || key == 65506) return(KEY_SHIFT);
	if (key == 65513 || key == 65027) return(KEY_ALT);
	if (key == 65507 || key == 65508) return(KEY_CTRL);
	if (key == 65362) return(KEY_UP);
	if (key == 65364) return(KEY_DOWN);
	if (key == 65361) return(KEY_LEFT);
	if (key == 65363) return(KEY_RIGHT);
	if (key == 65365) return(KEY_PAGEUP);
	if (key == 65366) return(KEY_PAGEDOWN);
	if (key == 65307) return(KEY_ESCAPE);
	if (key == 65293) return(KEY_ENTER);
	if (key == 65367) return(KEY_END);
	if (key == 65360) return(KEY_HOME);
	if (key == 65379) return(KEY_INSERT);
	if (key == 65470) return(KEY_F1);
	if (key == 65471) return(KEY_F2);
	if (key == 65472) return(KEY_F3);
	if (key == 65473) return(KEY_F4);
	if (key == 65474) return(KEY_F5);
	if (key == 65475) return(KEY_F6);
	if (key == 65476) return(KEY_F7);
	if (key == 65477) return(KEY_F8);
	if (key == 65478) return(KEY_F9);
	if (key == 65479) return(KEY_F10);
	if (key == 65480) return(KEY_F11);
	if (key == 65481) return(KEY_F12);
	if (key == 32) return(KEY_SPACE);
	if (key > 255) return(0);
	return 0;
}

void AliGPUCADisplayBackendX11::GetKey(XEvent& event, int& keyOut, int& keyPressOut)
{
	char tmpString[9];
	KeySym sym;
	if (XLookupString(&event.xkey, tmpString, 8, &sym, NULL) == 0) tmpString[0] = 0;
	int specialKey = GetKey(sym);
	int localeKey = tmpString[0];
	//printf("Key: keycode %d -> sym %d (%c) key %d (%c) special %d (%c)\n", event.xkey.keycode, (int) sym, (char) sym, (int) localeKey, localeKey, specialKey, (char) specialKey);

	if (specialKey)
	{
		keyOut = keyPressOut = specialKey;
	}
	else
	{
		keyOut = keyPressOut = localeKey;
		if (keyPressOut >= 'a' && keyPressOut <= 'z') keyPressOut += 'A' - 'a';
	}
}

void AliGPUCADisplayBackendX11::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a)
{
	glColor4f(r, g, b, a);
	glRasterPos2f(x, y);
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

int AliGPUCADisplayBackendX11::OpenGLMain()
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
		return(-1);
	}

	// Make sure OpenGL's GLX extension supported
	if (!glXQueryExtension(g_pDisplay, &errorBase, &eventBase))
	{
		fprintf(stderr, "glxsimple: %s\n", "X server has no OpenGL GLX extension");
		return(-1);
	}
	
	const char* glxExt = glXQueryExtensionsString(g_pDisplay, DefaultScreen(g_pDisplay));
	if (strstr(glxExt, "GLX_EXT_swap_control") == NULL)
	{
		fprintf(stderr, "No vsync support!\n");
		return(-1);
	}

	//Require MSAA, double buffering, and Depth buffer
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
//		GLX_SAMPLE_BUFFERS  , 1, //Disable MSAA here, we do it by rendering to offscreenbuffer
//		GLX_SAMPLES         , MSAA_SAMPLES,
		None
	};

	GLXFBConfig fbconfig = 0;
	int fbcount;
	GLXFBConfig *fbc = glXChooseFBConfig(g_pDisplay, DefaultScreen(g_pDisplay), attribs, &fbcount);
	if (fbc == NULL || fbcount == 0)
	{
		fprintf(stderr, "Failed to get MSAA GLXFBConfig\n");
		return(-1);
	}
	fbconfig = fbc[0];
	XFree(fbc);
	visualInfo = glXGetVisualFromFBConfig(g_pDisplay, fbconfig);
	
	if (visualInfo == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "no RGB visual with depth buffer");
		return(-1);
	}

	// Create an OpenGL rendering context
	glxContext = glXCreateContext(g_pDisplay, visualInfo, NULL, GL_TRUE);
	if (glxContext == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "could not create rendering context");
		return(-1);
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
	ToggleMaximized(true);
	
	//Receive signal when window closed
	Atom WM_DELETE_WINDOW = XInternAtom(g_pDisplay, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(g_pDisplay, g_window, &WM_DELETE_WINDOW, 1);
	
	//Prepare fonts
	font_base = glGenLists(256);
    if (!glIsList(font_base))
	{
       fprintf(stderr, "Out of display lists.\n");
       return(-1);
    }
	const char* f = "fixed";
	XFontStruct* font_info = XLoadQueryFont(g_pDisplay, f);
	if (!font_info)
	{
		fprintf(stderr, "XLoadQueryFont failed.\n");
		return(-1);
	}
	else
	{
		int first = font_info->min_char_or_byte2;
		int last = font_info->max_char_or_byte2;
		glXUseXFont(font_info->fid, first, last-first+1, font_base+first);
	}

	// Init OpenGL...
	if (glewInit()) return(-1);
	
	XMapWindow(g_pDisplay, g_window);
	XFlush(g_pDisplay);
	int x11_fd = ConnectionNumber(g_pDisplay);
	
	//Enable vsync
	glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC) glXGetProcAddressARB((const GLubyte*) "glXSwapIntervalEXT");
	if (glXSwapIntervalEXT == NULL)
	{
		fprintf(stderr, "Cannot enable vsync\n");
		return(-1);
	}
	glXSwapIntervalEXT(g_pDisplay, glXGetCurrentDrawable(), 0);
	
	if (InitGL()) return(1);
	
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
			num_ready_fds = maxFPSRate || XPending(g_pDisplay) || select(x11_fd + 1, &in_fds, NULL, NULL, &tv);
			if (num_ready_fds < 0)
			{
				fprintf(stderr, "Error\n");
			}
			if (displayControl == 2) break;
			if (sendKey) needUpdate = 1;
			if (waitCount++ != 100) needUpdate = 1;
		} while (!(num_ready_fds || needUpdate));
		needUpdate = 0;
		
		do
		{
			if (displayControl == 2) break;
			HandleSendKey();
			if (!XPending(g_pDisplay))
			{
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
					int handleKey = 0, keyPress = 0;
					GetKey(event, handleKey, keyPress);
					keysShift[keyPress] = keys[KEY_SHIFT];
					keys[keyPress] = true;
				}
				break;

				case KeyRelease:
				{
					int handleKey = 0, keyPress = 0;
					GetKey(event, handleKey, keyPress);
					HandleKeyRelease(handleKey);
					keys[keyPress] = false;
					keysShift[keyPress] = false;
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
						displayControl = 2;
					}
				}
				break;
			}
		} while (XPending(g_pDisplay)); // Loop to compress events
		if (displayControl == 2) break;

		DrawGLScene();
		glXSwapBuffers(g_pDisplay, g_window); // Buffer swap does implicit glFlush
	}
	
	glDeleteLists(font_base, 256);
	ExitGL();
	glXDestroyContext(g_pDisplay, glxContext);
	XUnloadFont(g_pDisplay, font_info->fid);
	XFree(visualInfo);
	XDestroyWindow(g_pDisplay, g_window);
	XCloseDisplay(g_pDisplay);
	
	pthread_mutex_lock(&semLockExit);
	displayRunning = false;
	pthread_mutex_unlock(&semLockExit);
		
	return(0);
}

void AliGPUCADisplayBackendX11::DisplayExit()
{
	pthread_mutex_lock(&semLockExit);
	if (displayRunning) displayControl = 2;
	pthread_mutex_unlock(&semLockExit);
	while (displayRunning) usleep(10000);
}

void AliGPUCADisplayBackendX11::SwitchFullscreen(bool set)
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

void AliGPUCADisplayBackendX11::ToggleMaximized(bool set)
{
	XEvent xev;
	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = g_window;
	xev.xclient.message_type = XInternAtom(g_pDisplay, "_NET_WM_STATE", False);
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = set ? 1 : 2; //_NET_WM_STATE_ADD
	xev.xclient.data.l[1] = XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
	xev.xclient.data.l[2] = XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_VERT", False);
	XSendEvent(g_pDisplay, DefaultRootWindow(g_pDisplay), False, SubstructureNotifyMask, &xev);
}

void AliGPUCADisplayBackendX11::SetVSync(bool enable)
{
	glXSwapIntervalEXT(g_pDisplay, glXGetCurrentDrawable(), (int) enable);
}

int AliGPUCADisplayBackendX11::StartDisplay()
{
	static pthread_t hThread;
	if (pthread_create(&hThread, NULL, OpenGLWrapper, this))
	{
		printf("Coult not Create GL Thread...\n");
		return(1);
	}
	return(0);
}
