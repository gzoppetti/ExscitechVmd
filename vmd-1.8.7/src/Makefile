
.SILENT:

# Makefile for vmd
# VMD for WIN32, version 1.8.7 (April 28, 2011)

# Exscitech: Switch between these!
PLATFORM =LINUXAMD64
#PLATFORM =WIN32
BUILDTARGET = DEBUG

CD          = cd
COPY        = cp
COPYDIR     = cp -r
MAKEDIR     = mkdir -p
MOVE        = mv -f
DELETE      = rm -f
DELETEDIR   = rm -rf
ECHO        = echo
TAR         = tar
COMPRESS    = compress
GNUCOMPRESS = /usr/local/bin/gzip
LATEX       = latex

# makefile configuration
VPATH                   = .:../$(PLATFORM)
SHELL                   = /bin/sh
.SUFFIXES: .C .c .cu ..c .i .o


# C++ source files
#OpenGLDisplayDevice.C 
VMD_CCPP    =	\
		OpenGLExtensions.C \
		OpenGLRenderer.C \
		OpenGLShader.C \
		OpenGLCache.C \
		Hershey.C \
		Animation.C \
		ArtDisplayDevice.C \
		AtomColor.C \
		AtomParser.C \
		AtomLexer.C \
		AtomRep.C \
		AtomSel.C \
		Axes.C \
		BaseMolecule.C \
		Benchmark.C \
		BondSearch.C \
		CmdAnimate.C \
		CmdColor.C \
		CmdDisplay.C \
		CmdLabel.C \
		CmdMaterial.C \
		CmdMenu.C \
		CmdMol.C \
		CmdRender.C \
		CmdTrans.C \
		CommandQueue.C \
		CoorPluginData.C \
		CUDAAccel.C \
		DisplayDevice.C \
		Displayable.C \
		DisplayRocker.C \
		DispCmds.C \
		DrawMolecule.C \
		DrawMolItem.C \
		DrawMolItem2.C \
		DrawMolItemRibbons.C \
		DrawMolItemMSMS.C \
		DrawMolItemRings.C \
		DrawMolItemOrbital.C \
		DrawMolItemSurface.C \
		DrawMolItemVolume.C \
		DrawForce.C \
		DrawRingsUtils.C \
		FileRenderList.C \
		FileRenderer.C \
		FltkOpenGLDisplayDevice.C \
		FPS.C \
		GelatoDisplayDevice.C \
		GeometryAngle.C \
		GeometryAtom.C \
		GeometryBond.C \
		GeometryDihedral.C \
		GeometryList.C \
		GeometryMol.C \
		GeometrySpring.C \
		Inform.C \
		ImageIO.C \
		Isosurface.C \
		JRegex.C \
		JString.C \
		macosxvmdstart.C \
		MaterialList.C \
		Matrix4.C \
		Measure.C \
		MeasurePBC.C \
		MeasureSurface.C \
		MeasureSymmetry.C \
		Molecule.C \
		MoleculeList.C \
		MoleculeGraphics.C \
		MolFilePlugin.C \
		Mouse.C \
		MSMSInterface.C \
		Orbital.C \
		PeriodicTable.C \
		P_JoystickTool.C \
		P_TugTool.C \
		P_GrabTool.C \
		P_PrintTool.C \
		P_PinchTool.C \
		P_UIVR.C \
		P_Buttons.C \
		P_Tracker.C \
		P_Tool.C \
		P_CmdTool.C \
		P_SensorConfig.C \
		ParseTree.C \
		PickList.C \
		PickModeAddBond.C \
		PickModeCenter.C \
		PickModeForce.C \
		PickModeList.C \
		PickModeMolLabel.C \
		PickModeMove.C \
		PickModeUser.C \
		PlainTextInterp.C \
		PluginMgr.C \
		POV3DisplayDevice.C \
		PSDisplayDevice.C \
		QMData.C \
		QMTimestep.C \
		RadianceDisplayDevice.C \
		RayShadeDisplayDevice.C \
		R3dDisplayDevice.C \
		RenderManDisplayDevice.C \
		Scene.C \
		SnapshotDisplayDevice.C \
		Spaceball.C \
		SpaceballButtons.C \
		SpaceballTracker.C \
		SpatialSearch.C \
		SpringTool.C \
		Stage.C \
		STLDisplayDevice.C \
		Stride.C \
		Surf.C \
		SymbolTable.C \
		TachyonDisplayDevice.C \
		Timestep.C \
		UIObject.C \
		UIText.C \
		VMDApp.C \
		VMDCollab.C \
		VMDDir.C \
		VMDDisplayList.C \
		VMDMenu.C \
		VMDQuat.C \
		VMDTitle.C \
		VMDThreads.C \
		VolCPotential.C \
		VolMapCreate.C \
		VolMapCreateILS.C \
		VolumetricData.C \
		VolumeTexture.C \
		VrmlDisplayDevice.C \
		Vrml2DisplayDevice.C \
		WavefrontDisplayDevice.C \
		utilities.C \
		vmd.C \
		vmdmain.C \
		ColorInfo.C \
		TclCommands.C \
		TclMeasure.C \
		TclMolInfo.C \
		TclTextInterp.C \
		TclVec.C \
		TclGraphics.C \
		TclVolMap.C \
		cmd_animate.C \
		cmd_collab.C \
		cmd_color.C \
		cmd_display.C \
		cmd_imd.C \
		cmd_label.C \
		cmd_material.C \
		cmd_menu.C \
		cmd_mol.C \
		cmd_mouse.C \
		cmd_plugin.C \
		cmd_render.C \
		cmd_spaceball.C \
		cmd_tool.C \
		cmd_trans.C \
		cmd_user.C \
		cmd_util.C \
		cmd_vmdbench.C \
		tcl_commands.C \
		VMDTkMenu.C \
		ColorFltkMenu.C \
		DisplayFltkMenu.C \
		FileChooserFltkMenu.C \
		GraphicsFltkMenu.C \
		SelectionBuilder.C \
		GeometryFltkMenu.C \
		MaterialFltkMenu.C \
		MolBrowser.C \
		RenderFltkMenu.C \
		SaveTrajectoryFltkMenu.C \
		VMDFltkMenu.C \
		MainFltkMenu.C \
		ToolFltkMenu.C \
		frame_selector.C

EXSCITECH_GAMES_CPP = \
		Exscitech/Games/GameController.C \
		Exscitech/Games/GameInfoManager.C \
	        Exscitech/Games/Game.C \
		\
		Exscitech/Games/JobSubmitGame/JobSubmitGame.C \
		\
		Exscitech/Games/IdentificationGame/IdentificationGame.C \
	        \
	        Exscitech/Games/LindseyGame/LindseyGame.C \

EXSCITECH_GRAPHICS_CPP = \
		Exscitech/Graphics/Scene.C \
		Exscitech/Graphics/Transformable.C \
		Exscitech/Graphics/Drawable.C \
		Exscitech/Graphics/MoleculeLoader.C \
		Exscitech/Graphics/Skybox.C \
		Exscitech/Graphics/VolmapExscitech.C \
		Exscitech/Graphics/SpaceFillMolecule.C \
		Exscitech/Graphics/SSAO.C \
		Exscitech/Graphics/FullQuad.C \
		\
		Exscitech/Graphics/Animation/KeyFrame.C \
		\
		Exscitech/Graphics/Lighting/Material.C \
		Exscitech/Graphics/Lighting/Light.C \
		Exscitech/Graphics/Lighting/PointLight.C \
		Exscitech/Graphics/Lighting/DirectionalLight.C \
		Exscitech/Graphics/Lighting/SpotLight.C \
		Exscitech/Graphics/Lighting/MaterialLibrary.C \
		Exscitech/Graphics/Lighting/Texture.C \
		Exscitech/Graphics/Lighting/CubeMap.C \
		\
		Exscitech/Graphics/Mesh/Mesh.C \
		Exscitech/Graphics/Mesh/MeshPart.C \
		Exscitech/Graphics/Mesh/Vertex.C \
		Exscitech/Graphics/Mesh/VertexAttribute.C \
		Exscitech/Graphics/Mesh/VertexBuffer.C \
		Exscitech/Graphics/Mesh/IndexBuffer.C \
		Exscitech/Graphics/Mesh/VertexDescriptor.C \
		Exscitech/Graphics/Mesh/ObjMeshLoader.C \
		\
		Exscitech/Graphics/Shaders/Shader.C \
		Exscitech/Graphics/Shaders/ShaderProgram.C \
		Exscitech/Graphics/Shaders/ShaderUtility.C \
		Exscitech/Graphics/Shaders/LightUniformManager.C

EXSCITECH_MATH_CPP = \
		Exscitech/Math/Quaternion.C

EXSCITECH_PHYSICS_CPP = 

EXSCITECH_DISPLAY_CPP = \
		Exscitech/Display/QtOpenGLDisplayDevice.C \
		Exscitech/Display/QtWindow.C \
		Exscitech/Display/VmdGlWidget.C \
		Exscitech/Display/GameSelectionWindow.C \
		Exscitech/Display/GameChoiceWidget.C \
		Exscitech/Display/WebLinkWidget.C \
		Exscitech/Display/GameWidgetScrollArea.C \
		Exscitech/Display/Camera.C \
		Exscitech/Display/LoginWindow.C \
		Exscitech/Display/ImageSelectionWidget.C \
		Exscitech/Display/ToggleWidget.C \
		Exscitech/Display/ToggleButton.C

EXSCITECH_UTILITIES_CPP = \
		Exscitech/Utilities/TransformUtility.C \
		Exscitech/Utilities/CameraUtility.C \
		Exscitech/Utilities/KeyboardBuffer.C \
		Exscitech/Utilities/BulletUtility.C \
		Exscitech/Utilities/ServerCommunicationManager.C \
		Exscitech/Utilities/TextUtility.C \
		Exscitech/Utilities/HostIdManager.C \
		Exscitech/Utilities/WorkunitId.C \
		Exscitech/Utilities/ErrorLog.C \
		Exscitech/Utilities/ProteinServerData.C \
		Exscitech/Utilities/ConformationServerData.C \
		Exscitech/Utilities/LigandServerData.C \

EXSCITECH_CPP = $(EXSCITECH_GAMES_CPP) \
				$(EXSCITECH_GRAPHICS_CPP) \
				$(EXSCITECH_MATH_CPP) \
				$(EXSCITECH_PHYSICS_CPP) \
				$(EXSCITECH_DISPLAY_CPP) \
				$(EXSCITECH_UTILITIES_CPP)

EXSCITECH_MOC_HEADERS = \
		Exscitech/Display/GameSelectionWindow.hpp \
		Exscitech/Display/GameChoiceWidget.hpp \
		Exscitech/Games/Game.hpp \
		Exscitech/Games/LindseyGame/LindseyGame.hpp \
		Exscitech/Games/IdentificationGame/IdentificationGame.hpp \
		Exscitech/Display/LoginWindow.hpp \
		Exscitech/Display/ImageSelectionWidget.hpp \
		Exscitech/Games/JobSubmitGame/JobSubmitGame.hpp \
		Exscitech/Display/ToggleWidget.hpp

# C source files
VMD_CC      = 	hash.c \
		inthash.c \
		intstack.c \
		msmpot.c \
		msmpot_compute.c \
		msmpot_cubic.c \
		msmpot_setup.c \
		vmdsock.c \
		vmddlopen.c \
		pcre.c \
		fitrms.c

# CUDA source files
VMD_CU      = 	

#
#

# Header files
#OpenGLDisplayDevice.h 
VMD_H       = 	\
		OpenGLExtensions.h \
		OpenGLRenderer.h \
		OpenGLShader.h \
		OpenGLCache.h \
		Hershey.h \
		imd.h \
		IMDMgr.h \
		IMDSim.h \
		IMDSimThread.h \
		CmdIMD.h \
		Animation.h \
		ArtDisplayDevice.h \
		Atom.h \
		AtomColor.h \
		AtomParser.h \
		AtomRep.h \
		AtomSel.h \
		Axes.h \
		BaseMolecule.h \
		Benchmark.h \
		BondSearch.h \
		CmdAnimate.h \
		CmdColor.h \
		CmdDisplay.h \
		CmdLabel.h \
		CmdMaterial.h \
		CmdMenu.h \
		CmdMol.h \
		CmdRender.h \
		CmdTrans.h \
		Command.h \
		CommandQueue.h \
		CoorData.h \
		CUDAAccel.h \
		CoorPluginData.h \
		DepthSortObj.h \
		DispCmds.h \
		DisplayDevice.h \
		Displayable.h \
		DisplayRocker.h \
		DrawMolecule.h \
		DrawMolItem.h \
		DrawMolItemSolventPoints.data \
		DrawForce.h \
		GelatoDisplayDevice.h \
		FPS.h \
		FileRenderList.h \
		FileRenderer.h \
		Fragment.h \
		GeometryAngle.h \
		GeometryAtom.h \
		GeometryBond.h \
		GeometryDihedral.h \
		GeometryList.h \
		GeometryMol.h \
		GeometrySpring.h \
		Inform.h \
		ImageIO.h \
		Isosurface.h \
		JRegex.h \
		JString.h \
		macosxvmdstart.h \
		MaterialList.h \
		Matrix4.h \
		Measure.h \
		MeasureSymmetry.h \
		Molecule.h \
		MoleculeGraphics.h \
		MoleculeList.h \
		MolFilePlugin.h \
		Mouse.h \
		MSMSInterface.h \
		NameList.h \
		PeriodicTable.h \
		Orbital.h \
		P_JoystickTool.h \
		P_TugTool.h \
		P_PinchToo.h \
		P_GrabTool.h \
		P_PrintTool.h \
		P_Feedback.h \
		P_UIVR.h \
		P_Buttons.h \
		P_Tracker.h \
		P_CmdTool.h \
		P_SensorConfig.h \
		P_Tool.h \
		ParseTree.h \
		PickList.h \
		PickMode.h \
		PickModeAddBond.h \
		PickModeCenter.h \
		PickModeForce.h \
		PickModeList.h \
		PickModeMolLabel.h \
		PickModeMove.h \
		Pickable.h \
		PlainTextInterp.h \
		PluginMgr.h \
		PointerTool.h \
		POV3DisplayDevice.h \
		PSDisplayDevice.h \
		QMData.h \
		QMTimestep.h \
		RadianceDisplayDevice.h \
		RayShadeDisplayDevice.h \
		R3dDisplayDevice.h \
		ResizeArray.h \
		RenderManDisplayDevice.h \
		Residue.h \
		Scene.h \
		SnapshotDisplayDevice.h \
		SortableArray.h \
		Spaceball.h \
		SpaceballButtons.h \
		SpaceballTracker.h \
		SpatialSearch.h \
		SpringTool.h \
		Stack.h \
		Stage.h \
		STLDisplayDevice.h \
		Stride.h \
		Surf.h \
		SymbolTable.h \
		TachyonDisplayDevice.h \
		TextEvent.h \
		TextInterp.h \
		Timestep.h \
		UIObject.h \
		UIText.h \
		VMDApp.h \
		VMDDir.h \
		VMDDisplayList.h \
		VMDMenu.h \
		VMDQuat.h \
		VMDTitle.h \
		VMDThreads.h \
		VolCPotential.h \
		VolMapCreate.h \
		VolumetricData.h \
		VolumeTexture.h \
		VrmlDisplayDevice.h \
		Vrml2DisplayDevice.h \
		WavefrontDisplayDevice.h \
		utilities.h \
		pcre.h \
		pcreinternal.h \
		pcretables.h \
		vmdsock.h \
		fitrms.h \
		TclCommands.h \
		TclTextInterp.h \
		tcl_commands.h \
		VMDTkMenu.h \
		plugin.h \
		molfile_plugin.h \
		libmolfile_plugin.h \
		ColorFltkMenu.h \
		DisplayFltkMenu.h \
		FileChooserFltkMenu.h \
		GeometryFltkMenu.h \
		GraphicsFltkMenu.h \
		GraphicsFltkReps.h \
		SelectionBuilder.h \
		MaterialFltkMenu.h \
		MolBrowser.h \
		RenderFltkMenu.h \
		SaveTrajectoryFltkMenu.h \
		VMDFltkMenu.h \
		MainFltkMenu.h \
		ToolFltkMenu.h \
		frame_selector.h 

# yacc and lex files
VMD_YACC    = 	AtomParser.y

VMD_LEX     = 	AtomLexer.l

# Misc. data file
VMD_DATA    = 	.vmdsensors .vmdrc

VMD_EXTRA          = 	

VMD_OTHER_EXE      = 	../lib/stride/stride_$(PLATFORM) ../lib/surf/surf_$(PLATFORM) ../lib/tachyon/tachyon_$(PLATFORM)

VMD_OTHER_NAMES    = 	stride_$(PLATFORM)

VMD_MAIN_DIR_FILES = 	Announcement FEEDBACK LICENSE README configure

# Turn things into objects

# Exscitech: Transform moc headers to src files with "Moc_" prefix
#            Dir/Filename.hpp => Dir/Moc_Filename.C 
mocHeaderBasenames = $(notdir $(EXSCITECH_MOC_HEADERS))
mocSrcBasenames = $(addprefix  Moc_, $(mocHeaderBasenames:.hpp=.C))

EXSCITECH_MOC_CPP = $(join $(dir $(EXSCITECH_MOC_HEADERS)), $(mocSrcBasenames))
EXSCITECH_OBJS = $(EXSCITECH_CPP:.C=.o) $(EXSCITECH_MOC_CPP:.C=.o)

VMD_OBJS    =   $(VMD_CCPP:.C=.o) $(EXSCITECH_OBJS) $(VMD_CC:.c=.o) $(VMD_CU:.cu=.o)

ifeq ($(PLATFORM), LINUXAMD64)
  #Exscitech: Linux
  EXSCITECH_INCDIRS = -I/usr/include/bullet -I/usr/include/bullet/HACD -I/usr/include/qt4
  INCDIRS           = $(EXSCITECH_INCDIRS) -I../lib/tk/include -I../plugins/include -I../plugins/LINUXAMD64/molfile -I../lib/netcdf/include -I../lib/fltk/include -I.
  EXSCITECH_LIBS    = -lcurl -lfltk_gl -lGLEW -lQtGui -lQtOpenGL -lQtCore -lQtWebKit -lQtXml -lboost_filesystem -lboost_system -lBulletDynamics -lBulletCollision -lLinearMath -lConvexDecomposition -lHACD -lIL -lILU -lfmodex64 #-lILUT #
  LIBS              = -L/usr/X11R6/lib64 -lGL -lGLU -lglut -lX11  -lpthread -ltk -lX11 -ltcl  -lmolfile_plugin -lnetcdf -lfltk -lX11 -lm -ldl $(VMDEXTRALIBS) $(EXSCITECH_LIBS)
  EXSCITECH_LIBDIRS = 
  LIBDIRS           = -L../lib/tcl/lib_LINXAMD64 -L../lib/tk/lib_LINUXAMD64  -L../plugins/LINUXAMD64/molfile -L../lib/netcdf/lib_LINUXAMD64 -L../lib/fltk/LINUXAMD64 $(EXSCITECH_LIBDIRS)
  OTHER_OPTIONS     = 
  DEFINES           = -DVMD_EXSCITECH -DVMDOPENGL -DVMDGRAPHICS -DVMDTHREADS -D_REENTRANT -DVMDORBITALS -DVMDWITHORBITALS -DVMDWITHCARBS -DVMDPOLYHEDRA -DVMDSURF -DVMDMSMS -DVMDFIELDLINES -DVMDPBCSMOOTH -DVMDTCL -DVMDTK  -DVMDSTATICPLUGINS  -DVMDGUI -DVMDFLTK 
  CFLAGS            = -m64 -Wall -O6 -ffast-math -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
  ifeq ($(BUILDTARGET), RELEASE)
    # Exscitech: Release Flags
    CFLAGS	   = -m64 -Wall -O2 -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
    CPPFLAGS   = -m64 -O2 -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
  else 
    # Exscitech: Not release, so assume debug Flags
    CFLAGS     = -m64 -Wall -ggdb -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
    #CPPFLAGS   = -m64 -fno-for-scope -Wno-deprecated -Wall -O6 -ffast-math  -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
    CPPFLAGS   = -std=c++0x -m64 -Wno-write-strings -Wno-unused-but-set-variable -Wall -ggdb -DARCH_WIN32 $(DEFINES) $(INCDIRS) -DUSE_INTERP_RESULT
  endif
endif
#End Exscitech: Linux

ifeq ($(PLATFORM), WIN32)
  #Exscitech: MINGW
  EXSCITECH_INCDIRS = -I/d/Qt/include -I/d/MinGW/include/bullet
  INCDIRS           = $(EXSCITECH_INCDIRS) -I../lib/tk/include -I../plugins/include -I../plugins/WIN32/molfile -I../lib/netcdf/include -I../lib/fltk/include -I.
  EXSCITECH_LIBDIRS = -L/d/Qt/lib
  LIBDIRS	        = -L../plugins/WIN32/molfile $(EXSCITECH_LIBDIRS)
  SYSTEM_LIBS       = -lkernel32 -luser32 -lgdi32 -lwinspool -lcomdlg32 -lcomctl32 -lwinmm -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lwsock32 -lodbc32 -lodbccp32
  EXSCITECH_LIBS    = -lcurl -lfltk_gl -lglew32 -lfreeglut_static -lglu32 -lopengl32 -lQtGui4 -lQtOpenGL4 -lQtCore4 -lQtXml4 -lQtWebKit4 -lboost_filesystem -lboost_system -lBulletDynamics -lBulletCollision -lLinearMath -lDevIL -lILU -lILUT
  LIBS	            = -ltk -ltcl -lmolfile_plugin -lfltk $(VMDEXTRALIBS) $(EXSCITECH_LIBS) $(SYSTEM_LIBS)
  OTHER_OPTIONS     =
  DEFINES           = -DFREEGLUT_STATIC -DMINGW -DVMD_EXSCITECH -DVMDOPENGL -DVMDGRAPHICS -DVMDTHREADS -D_REENTRANT -DVMDORBITALS -DVMDWITHORBITALS -DVMDWITHCARBS -DVMDPOLYHEDRA -DVMDSURF -DVMDMSMS -DVMDFIELDLINES -DVMDPBCSMOOTH -DVMDTCL -DVMDTK  -DVMDSTATICPLUGINS  -DVMDGUI -DVMDFLTK 
  ifeq ($(BUILDTARGET), RELEASE)
    # Exscitech: Release Flags
    CFLAGS   = -m32 -Wall -O2 -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
    CPPFLAGS = -m32 -O2 -DARCH_WIN32 $(DEFINES) $(INCDIRS)
  else
    # Exscitech: Debug Flags
    CFLAGS      = -m32 -Wall -ggdb -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
    CPPFLAGS    = -m32 -Wno-write-strings -Wall -ggdb -DARCH_WIN32 $(DEFINES) $(INCDIRS) 
  endif
endif
# End Exscitech: MINGW

# compiler and compiler directives 
CC          = gcc
CCPP	    = g++

NVCC        = /usr/local/encap/cuda-2.2/bin/nvcc
NVCCFLAGS   = --ptxas-options=-v --host-compilation=c -DARCH_WIN32 $(DEFINES) $(INCDIRS)

COMPILERC   = echo
RCFLAGS     = No resource compiler required on this platform.

DEPEND      = -MM
DEPENDFILE  = Makedata.depend

LOADLIBES   = $(LIBDIRS) $(LIBS)

LEX 	    = flex
YACC	    = yacc
YFLAGS      = -d

############################### 
# default rules 
###############################

Moc_%.C : %.hpp
	$(ECHO) "Moc'ing $< --> $@ ..."
	moc-qt4 $(DEFINES) $(INCDIRS) $< -o $@

.C.o: 
	$(ECHO) "Compiling " $< " --> " $*.o " ..."; \
	$(CCPP) $(CPPFLAGS) -c $< -o ../$(PLATFORM)/$@

.c.o:
	$(ECHO) "Compiling " $< " --> " $*.o " ..."; \
	$(CC) $(CFLAGS) -c $< -o ../$(PLATFORM)/$@

.cu.o:
	$(ECHO) "Compiling " $< " --> " $*.o " ..."; \
	$(NVCC) $(NVCCFLAGS) -c $< -o ../$(PLATFORM)/$@

.y.o:

.l.o:

########## Targets

### Source targets
all: vmd_$(PLATFORM)
default:   vmd_$(PLATFORM)

vmd_$(PLATFORM): y.tab.h $(EXSCITECH_MOC_CPP) $(VMD_OBJS)
	$(ECHO) "Linking " $@ "..."; \
	$(CD) ../$(PLATFORM) ; \
	if [ ! -r ./foobar ]; then \
	  ln -s ../src/foobar ./foobar ; \
	fi; \
	$(CCPP) $(CPPFLAGS) -I../src -o $@ $(VMD_OBJS) $(LOADLIBES) $(OTHER_OPTIONS); 
	$(COMPILERC) $(RCFLAGS)

install:
	if [ ! -d "/usr/local/lib/vmd" ]; then \
		$(MAKEDIR) "/usr/local/lib/vmd" ; \
	fi ; \
	if [ ! -d "/usr/local/bin" ]; then \
		$(MAKEDIR) "/usr/local/bin" ; \
	fi ; \
	if [ ! -d "/usr/local/lib/vmd"/doc ]; then \
		$(MAKEDIR) "/usr/local/lib/vmd"/doc; \
	fi
	-$(COPY) ../Announcement  "/usr/local/lib/vmd"
	-$(COPY) ../README        "/usr/local/lib/vmd"
	-$(COPY) ../LICENSE       "/usr/local/lib/vmd"
	-$(COPY) ../doc/ug.pdf        "/usr/local/lib/vmd"/doc
	if [ -f /usr/local/lib/vmd/vmd_WIN32 ]; then \
           $(MOVE) "/usr/local/lib/vmd/vmd_$(PLATFORM)" "/usr/local/lib/vmd/OLD_vmd_$(PLATFORM)" ; $(DELETE) "/usr/local/lib/vmd/OLD_vmd_WIN32" ; fi
	-$(COPY) ../$(PLATFORM)/vmd_$(PLATFORM) "/usr/local/lib/vmd"
	if [ -d "../lib/redistrib/lib_$(PLATFORM)" ]; then \
		$(CD) ../lib/redistrib/lib_$(PLATFORM); $(TAR) -cf - ./* | \
		(cd "/usr/local/lib/vmd" ; $(TAR) -xf -) \
	fi ;
	-$(COPY) ../lib/stride/stride_$(PLATFORM) ../lib/surf/surf_$(PLATFORM) ../lib/tachyon/tachyon_$(PLATFORM) "/usr/local/lib/vmd"
	-$(CD) ..; $(TAR) -cf - scripts | \
	(cd "/usr/local/lib/vmd" ; $(TAR) -xf -)
	-$(CD) ../lib; $(TAR) -cf - scripts | \
	(cd "/usr/local/lib/vmd" ; $(TAR) -xf -)
	-$(CD) ..; $(TAR) -cf - python | \
	(cd "/usr/local/lib/vmd"/scripts ; $(TAR) -xf -)
	-$(CD) ..; $(TAR) -cf - plugins | \
	(cd "/usr/local/lib/vmd" ; $(TAR) -xf -)
	-$(CD) ..; $(TAR) -cf - shaders | \
	(cd "/usr/local/lib/vmd" ; $(TAR) -xf -)
	-$(COPY) ../data/.vmdrc ../data/.vmdsensors ../data/vmd_completion.dat "/usr/local/lib/vmd"
	$(CD) ../bin ; \
	if [ -f run_vmd_tmp ]; then $(DELETE) run_vmd_tmp; fi ; \
	if [ ! -x "/bin/csh" ]; then \
		$(ECHO) "Info: /bin/csh shell not found, installing Bourne shell startup script instead" ; \
		$(ECHO) '#!/bin/sh' >> run_vmd_tmp ; \
		$(ECHO) 'defaultvmddir="/usr/local/lib/vmd"' >> run_vmd_tmp ; \
		$(ECHO) 'vmdbasename=vmd' >> run_vmd_tmp ; \
		cat vmd.sh >> run_vmd_tmp ; \
	else \
		$(ECHO) '#!/bin/csh' >> run_vmd_tmp ; \
		$(ECHO) 'set defaultvmddir="/usr/local/lib/vmd"' >> run_vmd_tmp ; \
		$(ECHO) 'set vmdbasename=vmd' >> run_vmd_tmp ; \
		cat vmd.csh >> run_vmd_tmp ; \
	fi ; \
	chmod +x run_vmd_tmp ; \
	$(COPY) run_vmd_tmp "/usr/local/bin"/vmd ; \
	$(DELETE) run_vmd_tmp
	$(ECHO) Make sure "/usr/local/bin"/vmd is in your path.
	$(ECHO) "VMD installation complete.  Enjoy!"

##### remove most of the cruft
clean:
	$(CD) ../$(PLATFORM) ;\
	$(DELETE) *.o *.warnings *.depend.old core		
	find ../$(PLATFORM) -name '*.o' -exec rm {} \;

veryclean: clean
	$(CD) ../$(PLATFORM) ; \
	  $(DELETEDIR) foobar; \
	  $(DELETE) vmd_$(PLATFORM)
	$(CD) ../src ; \
	  $(DELETE) *.o *.a *~ core; \
	  $(DELETEDIR) foobar; \
	  $(DELETE) vmd_$(PLATFORM)
	$(CD) ../doc ; \
	  $(DELETE) *~ core

# The '/usr/include' stuff is to reduce checking /usr/include dates
depend: y.tab.h
	if [ "$(DEPEND)" != "" ]; then \
	echo "Building Makefile dependencies"; \
	  $(ECHO) "Creating " $(DEPENDFILE) " ..."; \
	  if [ -f $(DEPENDFILE) ]; then \
	    $(MOVE) -f $(DEPENDFILE) $(DEPENDFILE).old; \
	fi; \
	  touch $(DEPENDFILE); \
	for i in ZZZ $(VMD_CCPP) ; do \
	  if [ "$$i" != "ZZZ" ]; then \
	        $(ECHO) checking dependencies for $$i ...; \
	        $(CCPP) $(DEPEND) $(CPPFLAGS) $$i |  \
			sed -e 's/\/usr\/include\/[^ ]*\..//g' \
			    -e 's/\.\.\/lib\/.*\/[^ ]*\..//g' | \
			grep -v '^ *\\$$' >> $(DEPENDFILE) ; \
	  fi; \
	done; \
	\
	for i in ZZZ $(EXSCITECH_CPP) $(EXSCITECH_MOC_CPP) ; do \
	  if [ "$$i" != "ZZZ" ]; then \
	        $(ECHO) checking dependencies for $$i ...; \
	        $(CCPP) -MM -MT $${i/.C/.o} $(CPPFLAGS) $$i |  \
			sed -e 's/\/usr\/include\/[^ ]*\..//g' \
			    -e 's/\.\.\/lib\/.*\/[^ ]*\..//g' | \
			grep -v '^ *\\$$' >> $(DEPENDFILE) ; \
	  fi; \
	done; \
	\
	for i in ZZZ $(VMD_CC) ; do \
	  if [ "$$i" != "ZZZ" ]; then \
	        $(ECHO) checking dependencies for $$i ...; \
	        $(CC) $(DEPEND) $(CFLAGS) $$i |  \
			sed -e 's/\/usr\/include\/[^ ]*\..//g' \
			    -e 's/\.\.\/lib\/.*\/[^ ]*\..//g' | \
			grep -v '^ *\\$$' >> $(DEPENDFILE) ; \
	  fi; \
	done; \
	$(ECHO) ParseTree.o AtomLexer.o AtomParser.o: y.tab.h \
                >> $(DEPENDFILE); \
	$(COPY) $(DEPENDFILE) $(DEPENDFILE).$(PLATFORM) ; \
	else \
	  $(ECHO) "Cannot find dependencies; your compiler does not support dependency checking."; \
        fi

mocs:
	for i in $(EXSCITECH_MOC_HEADERS) ; do \
	  $(ECHO) Processing $$i... ; \
	  moc-qt4 $(DEFINES) $(INCDIRS) $$i -o `dirname $$i`/Moc_`basename $${i/.hpp/.C}` ; \
	done

# to bootstrap without a Makedata.depend file, either
#   touch Makedata.depend
# or change the following line to 'sinclude'
include Makedata.depend
