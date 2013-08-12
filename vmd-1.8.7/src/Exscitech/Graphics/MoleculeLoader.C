#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <iostream>
#include <istream>
#include <algorithm>

#include "Exscitech/Constants.hpp"
#include "Exscitech/Graphics/MoleculeLoader.hpp"
#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Graphics/BallAndStickMolecule.hpp"
#include "Exscitech/Graphics/SpaceFillMolecule.hpp"
#include "Exscitech/Graphics/LabeledMolecule.hpp"

#include "Exscitech/Graphics/Atoms.hpp"
#include "Exscitech/Graphics/Bonds.hpp"
#include "Exscitech/Graphics/LabeledAtoms.hpp"
#include "Exscitech/Graphics/AtomicName.hpp"

#include "Exscitech/Games/GameController.hpp"

#include "VMDApp.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "Scene.h"

namespace Exscitech
{
  struct A
  {
    static std::map<const AtomicName, Vector3f>
    load_color_map ()
    {
      std::map<const AtomicName, Vector3f> map;

      map["H\0\0"] = Vector3f (255, 255, 255);
      map["HE\0"] = Vector3f (217, 255, 255);
      map["LI\0"] = Vector3f (204, 128, 255);
      map["BE\0"] = Vector3f (194, 255, 0);
      map["B\0\0"] = Vector3f (255, 181, 181);
      map["C\0\0"] = Vector3f (144, 144, 144);
      map["N\0\0"] = Vector3f (48, 80, 248);
      map["O\0\0"] = Vector3f (255, 13, 13);
      map["F\0\0"] = Vector3f (144, 224, 80);
      map["NE\0"] = Vector3f (179, 227, 245);
      map["NA\0"] = Vector3f (171, 92, 242);
      map["MG\0"] = Vector3f (138, 255, 0);
      map["AL\0"] = Vector3f (191, 166, 166);
      map["SI\0"] = Vector3f (240, 200, 160);
      map["P\0\0"] = Vector3f (255, 128, 0);
      map["S\0\0"] = Vector3f (255, 255, 48);
      map["CL\0\0"] = Vector3f (31, 240, 31);
      map["AR\0\0"] = Vector3f (128, 209, 227);
      map["K\0\0"] = Vector3f (143, 64, 212);
      map["CA\0"] = Vector3f (61, 255, 0);
      map["SC\0"] = Vector3f (230, 230, 23);
      map["TI\0"] = Vector3f (191, 194, 199);
      map["V\0\0"] = Vector3f (166, 166, 171);
      map["CR\0"] = Vector3f (138, 153, 199);
      map["MN\0"] = Vector3f (156, 122, 199);
      map["FE\0"] = Vector3f (224, 102, 51);
      map["CO\0"] = Vector3f (240, 144, 1);
      map["NI\0"] = Vector3f (80, 208, 80);
      map["CU\0"] = Vector3f (200, 128, 51);
      map["ZN\0"] = Vector3f (125, 128, 176);
      map["GA\0"] = Vector3f (194, 143, 143);
      map["GE\0"] = Vector3f (102, 143, 143);
      map["AS\0"] = Vector3f (189, 128, 227);
      map["SE\0"] = Vector3f (255, 161, 0);
      map["BR\0"] = Vector3f (166, 41, 41);
      map["KR\0"] = Vector3f (92, 184, 209);
      map["RB\0"] = Vector3f (112, 46, 176);
      map["SR\0"] = Vector3f (0, 255, 0);
      map["Y\0\0"] = Vector3f (148, 255, 255);
      map["ZR\0"] = Vector3f (148, 224, 224);
      map["NB\0"] = Vector3f (115, 194, 201);
      map["MO\0"] = Vector3f (84, 181, 181);
      map["TC\0"] = Vector3f (59, 158, 158);
      map["RU\0"] = Vector3f (36, 143, 143);
      map["RH\0"] = Vector3f (10, 125, 140);
      map["PD\0"] = Vector3f (0, 105, 133);
      map["AG\0"] = Vector3f (192, 192, 192);
      map["CD\0"] = Vector3f (255, 217, 143);
      map["IN\0"] = Vector3f (166, 117, 115);
      map["SN\0"] = Vector3f (102, 128, 128);
      map["SB\0"] = Vector3f (158, 99, 181);
      map["TR\0"] = Vector3f (212, 122, 0);
      map["I\0\0"] = Vector3f (148, 0, 148);
      map["XE\0"] = Vector3f (66, 158, 176);
      map["CS\0"] = Vector3f (87, 23, 143);
      map["BA\0"] = Vector3f (0, 201, 0);
      map["LA\0"] = Vector3f (112, 212, 255);
      map["CE\0"] = Vector3f (255, 255, 199);
      map["PR\0"] = Vector3f (217, 255, 199);
      map["ND\0"] = Vector3f (199, 255, 199);
      map["PM\0"] = Vector3f (163, 255, 199);
      map["SM\0"] = Vector3f (143, 255, 199);
      map["EU\0"] = Vector3f (97, 255, 199);
      map["GD\0"] = Vector3f (69, 255, 199);
      map["TB\0"] = Vector3f (48, 255, 199);
      map["DY\0"] = Vector3f (31, 255, 199);
      map["HO\0"] = Vector3f (0, 255, 156);
      map["ER\0"] = Vector3f (0, 230, 117);
      map["TM\0"] = Vector3f (0, 212, 82);
      map["YB\0"] = Vector3f (0, 191, 56);
      map["LU\0"] = Vector3f (0, 171, 36);
      map["HF\0"] = Vector3f (77, 194, 255);
      map["TA\0"] = Vector3f (77, 166, 255);
      map["W\0\0"] = Vector3f (33, 148, 214);
      map["RE\0"] = Vector3f (38, 125, 171);
      map["OS\0"] = Vector3f (38, 102, 150);
      map["IR\0"] = Vector3f (23, 84, 135);
      map["PT\0"] = Vector3f (208, 208, 224);
      map["AU\0"] = Vector3f (255, 209, 35);
      map["HG\0"] = Vector3f (184, 184, 208);
      map["TL\0"] = Vector3f (166, 84, 77);
      map["PB\0"] = Vector3f (87, 89, 97);
      map["BI\0"] = Vector3f (158, 79, 181);
      map["PO\0"] = Vector3f (171, 92, 0);
      map["AT\0"] = Vector3f (117, 79, 69);
      map["RN\0"] = Vector3f (66, 130, 150);
      map["FR\0"] = Vector3f (66, 0, 102);
      map["RA\0"] = Vector3f (0, 125, 0);
      map["AC\0"] = Vector3f (112, 171, 250);
      map["TH\0"] = Vector3f (0, 186, 255);
      map["PA\0"] = Vector3f (0, 161, 255);
      map["U\0\0"] = Vector3f (0, 143, 255);
      map["NP\0"] = Vector3f (0, 128, 255);
      map["PU\0"] = Vector3f (0, 107, 255);
      map["AM\0"] = Vector3f (84, 92, 242);
      map["CM\0"] = Vector3f (120, 92, 227);
      map["BK\0"] = Vector3f (138, 79, 227);
      map["CF\0"] = Vector3f (161, 54, 212);
      map["ES\0"] = Vector3f (179, 31, 212);
      map["FM\0"] = Vector3f (179, 31, 186);
      map["MD\0"] = Vector3f (179, 13, 166);
      map["NO\0"] = Vector3f (189, 13, 135);
      map["LR\0"] = Vector3f (199, 0, 102);
      map["RF\0"] = Vector3f (204, 0, 89);
      map["DB\0"] = Vector3f (209, 0, 79);
      map["SG\0"] = Vector3f (217, 0, 69);
      map["BH\0"] = Vector3f (224, 0, 56);
      map["HS\0"] = Vector3f (230, 0, 46);
      map["MT\0"] = Vector3f (235, 0, 38);

      // Stupid USA, having fake elements...
      // This is the usa "alternative" to NB...
      map["CB\0"] = Vector3f (115, 194, 201);

      for(std::pair<const AtomicName, Vector3f>& pair : map)
      {
        pair.second /= 255.f;
      }
      return map;
    }

  };
  std::map<const AtomicName, Vector3f> MoleculeLoader::ms_atomColorMap =
      A::load_color_map ();

  GameController* MoleculeLoader::ms_gameControllerInstance = GameController::acquire();

  BallAndStickMolecule*
  MoleculeLoader::loadMolecule (const std::string& pdbFile)
  {
    FileSpec spec;
    int molId = ms_gameControllerInstance->m_vmdApp->molecule_load (-1, pdbFile.c_str (),
        NULL, &spec);
    Molecule* molecule = ms_gameControllerInstance->m_vmdApp->moleculeList->mol_from_id (
        molId);
    int numAtoms = molecule->nAtoms;

    std::vector<AtomicName> atomNames;
    atomNames.reserve (numAtoms);

    std::vector<unsigned int> bondIndices;

    Vector3f* positionPointer =
        reinterpret_cast<Vector3f*> (molecule->current ()->pos);
    std::vector<Vector3f> atomPositions (positionPointer,
        &positionPointer[numAtoms]);

    centerMolecule (atomPositions);

    for (int i = 0; i < numAtoms; ++i)
    {
      MolAtom* atom = molecule->atom (i);
      int atomNameIndex = atom->nameindex;
      atomNames.push_back (AtomicName(molecule->atomNames.name (atomNameIndex)));

      for (int j = 0; j < atom->bonds; ++j)
      {
        int bondTo = atom->bondTo[j];
        fprintf (stderr, "Checking bond between %d and %d\n", i, bondTo);

        if (i > bondTo)
        {
          fprintf (stderr, "new bond: %d %d\n", i, atom->bondTo[j]);
          bondIndices.push_back (i);
          bondIndices.push_back (bondTo);
        }
      }
    }

    ms_gameControllerInstance->m_vmdApp->scene->root.remove_child (molecule);
    ms_gameControllerInstance->m_vmdApp->molecule_delete (molId);

    Atoms* atoms = new Atoms (atomNames, atomPositions, Constants::BALL_AND_STICK_RADIUS_SCALE);
    Bonds* bonds = new Bonds (atomPositions, bondIndices,
        Constants::BOND_DETAILS);
    return new BallAndStickMolecule (atoms, bonds);
  }

  LabeledMolecule*
  MoleculeLoader::loadLabeledMolecule (const std::string& pdbFile)
  {
    FileSpec spec;
    int molId = ms_gameControllerInstance->m_vmdApp->molecule_load (-1, pdbFile.c_str (),
        NULL, &spec);
    Molecule* molecule = ms_gameControllerInstance->m_vmdApp->moleculeList->mol_from_id (
        molId);
    int numAtoms = molecule->nAtoms;

    std::vector<AtomicName> atomNames;
    atomNames.reserve (numAtoms);

    std::vector<unsigned int> bondIndices;

    Vector3f* positionPointer =
        reinterpret_cast<Vector3f*> (molecule->current ()->pos);
    std::vector<Vector3f> atomPositions (positionPointer,
        &positionPointer[numAtoms]);

    centerMolecule (atomPositions);

    for (int i = 0; i < numAtoms; ++i)
    {
      MolAtom* atom = molecule->atom (i);
      int atomNameIndex = atom->nameindex;
      atomNames.push_back (
          AtomicName (molecule->atomNames.name (atomNameIndex)));

      for (int j = 0; j < atom->bonds; ++j)
      {
        int bondTo = atom->bondTo[j];
        fprintf (stderr, "Checking bond between %d and %d\n", i, bondTo);

        if (i > bondTo)
        {
          fprintf (stderr, "new bond: %d %d\n", i, atom->bondTo[j]);
          bondIndices.push_back (i);
          bondIndices.push_back (bondTo);
        }
      }
    }

    ms_gameControllerInstance->m_vmdApp->scene->root.remove_child (molecule);
    ms_gameControllerInstance->m_vmdApp->molecule_delete (molId);

    LabeledAtoms* atoms = new LabeledAtoms (atomNames, atomPositions, Constants::BALL_AND_STICK_RADIUS_SCALE);
    Bonds* bonds = new Bonds (atomPositions, bondIndices,
        Constants::BOND_DETAILS);
    return new LabeledMolecule (atoms, bonds);
  }

  SpaceFillMolecule*
  MoleculeLoader::loadSpaceFillMolecule (const std::string& pdbFile)
  {
    FileSpec spec;
    int molId = ms_gameControllerInstance->m_vmdApp->molecule_load (-1, pdbFile.c_str (),
        NULL, &spec);
    Molecule* molecule = ms_gameControllerInstance->m_vmdApp->moleculeList->mol_from_id (
        molId);
    ms_gameControllerInstance->m_vmdApp->VMDupdate (1);
    int numAtoms = molecule->nAtoms;

    std::vector<AtomicName> atomNames;
    atomNames.reserve (numAtoms);

    Vector3f* positionPointer =
        reinterpret_cast<Vector3f*> (molecule->current ()->pos);
    std::vector<Vector3f> atomPositions (positionPointer,
        &positionPointer[numAtoms]);

    for (int i = 0; i < numAtoms; ++i)
    {
      MolAtom* atom = molecule->atom (i);
      int atomNameIndex = atom->nameindex;
      atomNames.push_back (AtomicName(molecule->atomNames.name (atomNameIndex)));
    }

    ms_gameControllerInstance->m_vmdApp->scene->root.remove_child (molecule);
    ms_gameControllerInstance->m_vmdApp->molecule_delete (molId);

    return new SpaceFillMolecule (atomPositions, atomNames);
  }

  void
  MoleculeLoader::centerMolecule (std::vector<Vector3f>& vertices)
  {
    Vector3f positionSum;
    size_t numVertices = vertices.size ();
    for (size_t i = 0; i < numVertices; ++i)
    {
      positionSum += vertices[i];
    }

    positionSum /= numVertices;

    for (size_t i = 0; i < numVertices; ++i)
    {
      vertices[i] -= positionSum;
    }
  }

  Vector3f
  MoleculeLoader::getAtomicDetailFromName (const AtomicName& name)
  {
    Vector3f color = ms_atomColorMap[name];
    if (color.length() < 0.01f)
    {
      color.set(rand() % 255, rand() % 255, rand() % 255);
      ms_atomColorMap[name] = color;
    }
    return color;
  }
}
