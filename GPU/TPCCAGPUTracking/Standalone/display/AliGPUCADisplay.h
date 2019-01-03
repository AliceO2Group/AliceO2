#ifndef ALIGPUCADISPLAY_H
#define ALIGPUCADISPLAY_H

#include "AliGPUCADisplayBackend.h"
#include "AliGPUReconstruction.h"
#include "../cmodules/vecpod.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>

class AliHLTTPCCATracker;
class AliGPUCAParam;
class AliGPUReconstruction;
class AliGPUCAQA;

#if !defined(GL_VERSION_4_6) || GL_VERSION_4_6 != 1
#error Unsupported OpenGL version < 4.6
#endif

class AliGPUCADisplay
{
public:
	static constexpr int fgkNSlices = AliGPUReconstruction::NSLICES;
	
	AliGPUCADisplay(AliGPUCADisplayBackend* backend, AliGPUReconstruction* rec, AliGPUCAQA* qa);
	~AliGPUCADisplay() = default;
	AliGPUCADisplay(const AliGPUCADisplay&) = delete;
	
	void ShowNextEvent();
	
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
	
	void HandleKeyRelease(int wParam, char key);
	int DrawGLScene(bool mixAnimation = false, float animateTime = -1.f);
	void HandleSendKey();
	int InitGL();
	void ExitGL();
	void ReSizeGLScene(int width, int height, bool init = false);

private:
	static constexpr const int N_POINTS_TYPE = 9;
	static constexpr const int N_LINES_TYPE = 6;
	static constexpr const int N_FINAL_TYPE = 4;
	static constexpr int TRACK_TYPE_ID_LIMIT = 100;
	
	typedef std::tuple<GLsizei, GLsizei, int> vboList;
	struct GLvertex {GLfloat x, y, z; GLvertex(GLfloat a, GLfloat b, GLfloat c) : x(a), y(b), z(c) {}};

	struct DrawArraysIndirectCommand
	{
		DrawArraysIndirectCommand(unsigned int a = 0, unsigned int b = 0, unsigned int c = 0, unsigned int d = 0) : count(a), instanceCount(b), first(c), baseInstance(d) {}
		unsigned int  count;
		unsigned int  instanceCount;
		unsigned int  first;
		unsigned int  baseInstance;
	};
	
	struct GLfb
	{
		GLuint fb_id = 0, fbCol_id = 0, fbDepth_id = 0;
		bool tex = false;
		bool msaa = false;
		bool depth = false;
		bool created = false;
	};
	
	struct threadVertexBuffer
	{
		vecpod<GLvertex> buffer;
		vecpod<GLint> start[N_FINAL_TYPE];
		vecpod<GLsizei> count[N_FINAL_TYPE];
		std::pair<vecpod<GLint>*, vecpod<GLsizei>*> vBuf[N_FINAL_TYPE];
		threadVertexBuffer() : buffer()
		{
			for (int i = 0;i < N_FINAL_TYPE;i++) {vBuf[i].first = start + i; vBuf[i].second = count + i;}
		}
		void clear()
		{
			for (int i = 0;i < N_FINAL_TYPE;i++) {start[i].clear(); count[i].clear();}
		}
	};
	
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
	
	const AliGPUCAParam& param();
	const AliHLTTPCCATracker& sliceTracker(int iSlice);
	const AliGPUReconstruction::InOutPointers ioptrs();
	void drawVertices(const vboList& v, const GLenum t);
	void insertVertexList(std::pair<vecpod<GLint>*, vecpod<GLsizei>*>& vBuf, size_t first, size_t last);
	void insertVertexList(int iSlice, size_t first, size_t last);
	template <typename... Args> void SetInfo(Args... args);
	void calcXYZ();
	void SetCollisionFirstCluster(unsigned int collision, int slice, int cluster);
	void animationCloseAngle(float& newangle, float lastAngle);
	void animateCloseQuaternion(float* v, float lastx, float lasty, float lastz, float lastw);
	void setAnimationPoint();
	void resetAnimation();
	void removeAnimationPoint();
	void startAnimation();
	void showInfo(const char* info);
	void SetColorClusters();
	void SetColorInitLinks();
	void SetColorLinks();
	void SetColorSeeds();
	void SetColorTracklets();
	void SetColorTracks();
	void SetColorGlobalTracks();
	void SetColorFinal();
	void SetColorGrid();
	void SetColorMarked();
	void SetCollisionColor(int col);
	void setQuality();
	void setDepthBuffer();
	void createFB_texture(GLuint& id, bool msaa, GLenum storage, GLenum attachment);
	void createFB_renderbuffer(GLuint& id, bool msaa, GLenum storage, GLenum attachment);
	void createFB(GLfb& fb, bool tex, bool withDepth, bool msaa);
	void deleteFB(GLfb& fb);
	void setFrameBuffer(int updateCurrent = -1, GLuint newID = 0);
	void UpdateOffscreenBuffers(bool clean = false);
	void updateConfig();
	void drawPointLinestrip(int iSlice, int cid, int id, int id_limit = TRACK_TYPE_ID_LIMIT);
	vboList DrawClusters(const AliHLTTPCCATracker &tracker, int select, int iCol);
	vboList DrawLinks(const AliHLTTPCCATracker &tracker, int id, bool dodown = false);
	vboList DrawSeeds(const AliHLTTPCCATracker &tracker);
	vboList DrawTracklets(const AliHLTTPCCATracker &tracker);
	vboList DrawTracks(const AliHLTTPCCATracker &tracker, int global);
	void DrawFinal(int iSlice, int /*iCol*/, AliHLTTPCGMPropagator* prop, std::array<vecpod<int>, 2>& trackList, threadVertexBuffer& threadBuffer);
	vboList DrawGrid(const AliHLTTPCCATracker &tracker);
	void DoScreenshot(char *filename, float animateTime = -1.f);
	void PrintHelp();
	void createQuaternionFromMatrix(float* v, const float* mat);
	
	AliGPUCADisplayBackend* mBackend;
	OpenGLConfig cfg;
	AliGPUReconstruction* mRec;
	AliGPUCAQA* mQA;
	const AliHLTTPCGMMerger& merger;

	GLfb mixBuffer;
	
	GLuint vbo_id[fgkNSlices], indirect_id;
	int indirectSliceOffset[fgkNSlices];
	vecpod<GLvertex> vertexBuffer[fgkNSlices];
	vecpod<GLint> vertexBufferStart[fgkNSlices];
	vecpod<GLsizei> vertexBufferCount[fgkNSlices];
	vecpod<GLuint> mainBufferStack{0};

	int drawCalls = 0;
	bool useGLIndirectDraw = true;
	bool useMultiVBO = false;
	
	bool invertColors = false;
	const int drawQualityPoint = 0;
	const int drawQualityLine = 0;
	const int drawQualityPerspective = 0;
	const int drawQualityRenderToTexture = 1;
	int drawQualityMSAA = 0;
	int drawQualityDownsampleFSAA = 0;
	bool drawQualityVSync = 0;

	int testSetting = 0;

	bool camLookOrigin = false;
	bool camYUp = false;
	int cameraMode = 0;

	float angleRollOrigin = -1e9;
	float maxClusterZ = 0;

	int screenshot_scale = 1;

	int screen_width = AliGPUCADisplayBackend::init_width, screen_height = AliGPUCADisplayBackend::init_height;
	int render_width = AliGPUCADisplayBackend::init_width, render_height = AliGPUCADisplayBackend::init_height;

	bool separateGlobalTracks = 0;
	bool propagateLoopers = 0;

	GLfloat currentMatrix[16];
	float xyz[3];
	float angle[3];
	float rphitheta[3];
	float quat[4];
	
	int projectxy = 0;

	int markClusters = 0;
	int hideRejectedClusters = 1;
	int hideUnmatchedClusters = 0;
	int hideRejectedTracks = 1;
	int markAdjacentClusters = 0;

	vecpod<std::array<int,37>> collisionClusters;
	int nCollisions = 1;
	
	float Xadd = 0;
	float Zadd = 0;

	std::unique_ptr<float4[]> globalPosPtr = nullptr;
	float4* globalPos;
	int maxClusters = 0;
	int currentClusters = 0;

	int glDLrecent = 0;
	int updateDLList = 0;

	int animate = 0;
	HighResTimer animationTimer;
	int animationFrame = 0;
	int animationLastBase = 0;
	int animateScreenshot = 0;
	int animationExport = 0;
	bool animationChangeConfig = true;
	float animationDelay = 2.f;
	vecpod<float> animateVectors[9];
	vecpod<OpenGLConfig> animateConfig;
	opengl_spline animationSplines[8];
	
	volatile int resetScene = 0;

	int printInfoText = 1;
	char infoText2[1024];
	HighResTimer infoText2Timer, infoHelpTimer;
	
	GLfb offscreenBuffer, offscreenBufferNoMSAA;
	std::vector<threadVertexBuffer> threadBuffers;
	std::vector<std::vector<std::array<std::array<vecpod<int>, 2>, fgkNSlices>>> threadTracks;
};

#endif
