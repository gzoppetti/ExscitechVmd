
echo "Be sure you've installed tcl8.5-dev"
echo "  and tk8.5-dev"
echo "  and libnetcdf-dev"
echo "  and libfltk-dev"

export TCL_INCLUDE_DIR="/usr/include/"

currentDirectory=`pwd`
export PLUGINDIR="${currentDirectory}/../vmd-1.8.7/plugins"

cd ../plugins
make LINUXAMD64 TCLINC=-I${TCL_INCLUDE_DIR}
make distrib

cd ../vmd-1.8.7
./configure LINUXAMD64 OPENGL TK FLTK IMD SILENT TCL PTHREADS NETCDF

mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Display
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/EddieGame
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/EddieGame2
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/LindseyGame
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/Dragndock
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/TestGames
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Animation
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Instancing
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Lighting
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Mesh
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Shaders
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Math
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Physics
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Physics/Bullet
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Utilities

touch ../vmd-1.8.7/src/Makedata.depend

echo "The original Makefile in vmd-<version> has been replaced by configure."
echo "Make sure you checkout the lastest Makefile from the repo after running this script."

