
echo "Be sure you've installed tcl8.5-dev"
echo "  and tk8.5-dev"
echo "  and libfltk-dev"

export TCL_INCLUDE_DIR="/usr/local/include/"

currentDirectory=`pwd`
export PLUGINDIR="${currentDirectory}/../vmd-1.8.7/plugins"

cd ../plugins
make WIN32MINGW TCLINC=-I${TCL_INCLUDE_DIR}
make distrib

cd ../vmd-1.8.7
./configure WIN32 OPENGL TK FLTK SILENT TCL

mkdir ../vmd-1.8.7/WIN32
mkdir ../vmd-1.8.7/WIN32/Exscitech
mkdir ../vmd-1.8.7/WIN32/Exscitech/Display
mkdir ../vmd-1.8.7/WIN32/Exscitech/Games
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/JobSubmitGame
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/IdentificationGame
mkdir ../vmd-1.8.7/WIN32/Exscitech/Games/LindseyGame
mkdir ../vmd-1.8.7/WIN32/Exscitech/Graphics
mkdir ../vmd-1.8.7/WIN32/Exscitech/Graphics/Animation
mkdir ../vmd-1.8.7/WIN32/Exscitech/Graphics/Instancing
mkdir ../vmd-1.8.7/WIN32/Exscitech/Graphics/Lighting
mkdir ../vmd-1.8.7/WIN32/Exscitech/Graphics/Mesh
mkdir ../vmd-1.8.7/WIN32/Exscitech/Graphics/Shaders
mkdir ../vmd-1.8.7/WIN32/Exscitech/Math
mkdir ../vmd-1.8.7/WIN32/Exscitech/Physics
mkdir ../vmd-1.8.7/WIN32/Exscitech/Physics/Bullet
mkdir ../vmd-1.8.7/WIN32/Exscitech/Utilities

touch ../vmd-1.8.7/src/Makedata.depend

echo "The original Makefile in vmd-<version> has been replaced by configure."
echo "Make sure you checkout the lastest Makefile from the repo after running this script."

