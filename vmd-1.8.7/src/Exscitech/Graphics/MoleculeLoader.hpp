#ifndef MOLECULELOADER_HPP_
#define MOLECULELOADER_HPP_

#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <istream>
#include <vector>

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Graphics/AtomicName.hpp"

namespace Exscitech
{
class GameController;
class BallAndStickMolecule;
class SpaceFillMolecule;
class LabeledMolecule;
class MoleculeLoader
{

public:

  static BallAndStickMolecule*
  loadMolecule (const std::string& pdbFile);

  static LabeledMolecule*
  loadLabeledMolecule(const std::string& pdbFile);

  static SpaceFillMolecule*
  loadSpaceFillMolecule (const std::string& pdbFile);

  static Vector3f getAtomicDetailFromName(const AtomicName& name);

private:

  static std::map<const AtomicName, Vector3f> ms_atomColorMap;
  static GameController* ms_gameControllerInstance;

private:

  static void
  centerMolecule (std::vector<Vector3f>& vertices);
};
}
#endif

