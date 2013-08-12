#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

#include "Exscitech/Types.hpp"

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"
namespace Exscitech
{
  namespace Constants
  {
    static const Vector3f WORLD_X (1, 0, 0);
    static const Vector3f WORLD_Y (0, 1, 0);
    static const Vector3f WORLD_Z (0, 0, 1);
    static const Vector3f NEG_WORLD_Y (0, -1, 0);
    static const Vector3f ZERO (0, 0, 0);

    static const Vector4f BOND_DETAILS (0.2f, 0.2f, 0.2f, 0.05f);

    static const Vector3f CARBON_COLOR (.5, .5, .5);
    static const Vector3f OXYGEN_COLOR (0, 0, .7);
    static const Vector3f HYDROGEN_COLOR (.7, 0, 0);
    static const Vector3f NITROGEN_COLOR (0, .7, .5);
    static const Vector3f DEFAULT_COLOR (0, .7, 0);

    static const float DEFAULT_RADIUS = 0.25f;
    static const float BALL_AND_STICK_RADIUS_SCALE = 2.0f;
    static const float SPACE_FILL_RADIUS_SCALE = 3.0f;
  }

#define BIT(x) (1<<(x))
  enum BulletCollisionGroups
  {
    COL_NOTHING = 0,
    COL_CHARACTER = BIT(1),
    COL_WALL = BIT(2),
    COL_OBJECT = BIT(3),
    COL_ALL = BIT(1) | BIT(2) | BIT(3)
  };

  namespace StipplePatterns
  {

    //    static const GLubyte HALFTONE[] =
    //      {
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55,
    //          0xAA, 0xAA, 0xAA, 0xAA,
    //          0x55, 0x55, 0x55, 0x55 };

    // 0000|0000|0000|0000|0000|0000|0000|0100 00 00 00 04
    // 0111|1111|1111|1111|1111|1111|1110|0010 7F FF FF E2
    // 0100|0000|0000|0000|0000|0000|0001|0001 40 00 00 11
    // 0100|1111|1111|1111|1111|1111|1100|1001 4F FF FF C9
    // 0100|1000|0000|0000|0000|0000|0010|0101 48 00 00 25
    // 0101|0001|1111|1111|1111|1111|1001|0101 51 FF FF 95
    // 0101|0010|0000|0000|0000|0000|1001|0101 52 00 00 95
    // 0101|0010|0111|1111|1111|1111|0101|0101 52 7F FF 55
    // 0101|0100|1000|0000|0000|0001|0101|0101 54 80 01 55
    // 0101|0100|1001|1111|1111|1001|0101|0101 54 9F F9 55
    // 0101|0101|0010|0000|0000|0101|0101|0101 55 20 05 55
    // 0101|0101|0100|1111|1110|0101|0101|0101 55 4F E5 55
    // 0101|0101|0101|0000|0001|0101|0101|0101 55 50 15 55
    // 0101|0101|0101|0011|1001|0101|0101|0101 55 53 95 55
    // 0101|0101|0101|0100|0101|0101|0101|0101 55 54 55 55
    // 0101|0101|0101|0101|0101|0101|0101|0101 55 55 55 55
    // 0101|0101|0101|0100|1001|0101|0101|0101 55 54 95 55
    // 0101|0101|0101|0100|0010|0101|0101|0101 55 54 25 55
    // 0101|0101|0101|0011|1110|0101|0101|0101 55 53 E5 55
    // 0101|0101|0100|1000|0000|1001|0101|0101 55 48 09 55
    // 0101|0101|0010|0111|1111|0010|0101|0101 55 27 F2 55
    // 0101|0101|0001|0000|0000|0100|1001|0101 55 10 04 95
    // 0101|0100|1000|1111|1111|1001|0010|0101 54 8F F9 25
    // 0101|0100|1000|0000|0000|0100|0100|0101 54 80 04 45
    // 0101|0001|0011|1111|1111|1100|0100|1001 51 3F FC 49
    // 0100|1000|1000|0000|0000|0000|1001|0001 48 80 00 91
    // 0100|0100|0111|1111|1111|1111|0010|0010 55 7F FF 22
    // 0010|0010|0000|0000|0000|0000|0100|0100 22 00 00 44
    // 0001|0001|1111|1111|1111|1111|1000|1000 11 FF FF 88
    // 0000|1000|0000|0000|0000|0000|0001|0000 08 00 00 10
    // 0000|0111|1111|1111|1111|1111|1110|0000 07 FF FF E0
    // 0000|0000|0000|0000|0000|0000|0000|0000 00 00 00 00

    static const unsigned char SPIRAL[] =
      {
          0x00,
          0x00,
          0x00,
          0x04,
          0x7F,
          0xFF,
          0xFF,
          0xE2,
          0x40,
          0x00,
          0x00,
          0x11,
          0x4F,
          0xFF,
          0xFF,
          0xC9,
          0x48,
          0x00,
          0x00,
          0x25,
          0x51,
          0xFF,
          0xFF,
          0x95,
          0x52,
          0x00,
          0x00,
          0x95,
          0x52,
          0x7F,
          0xFF,
          0x55,
          0x54,
          0x80,
          0x01,
          0x55,
          0x54,
          0x9F,
          0xF9,
          0x55,
          0x55,
          0x20,
          0x05,
          0x55,
          0x55,
          0x4F,
          0xE5,
          0x55,
          0x55,
          0x50,
          0x15,
          0x55,
          0x55,
          0x53,
          0x95,
          0x55,
          0x55,
          0x54,
          0x55,
          0x55,
          0x55,
          0x55,
          0x55,
          0x55,
          0x55,
          0x54,
          0x95,
          0x55,
          0x55,
          0x54,
          0x25,
          0x55,
          0x55,
          0x53,
          0xE5,
          0x55,
          0x55,
          0x48,
          0x09,
          0x55,
          0x55,
          0x27,
          0xF2,
          0x55,
          0x55,
          0x10,
          0x04,
          0x95,
          0x54,
          0x8F,
          0xF9,
          0x25,
          0x54,
          0x80,
          0x04,
          0x45,
          0x51,
          0x3F,
          0xFC,
          0x49,
          0x48,
          0x80,
          0x00,
          0x91,
          0x55,
          0x7F,
          0xFF,
          0x22,
          0x22,
          0x00,
          0x00,
          0x44,
          0x11,
          0xFF,
          0xFF,
          0x88,
          0x08,
          0x00,
          0x00,
          0x10,
          0x07,
          0xFF,
          0xFF,
          0xE0,
          0x00,
          0x00,
          0x00,
          0x00 };

    static const unsigned char HALFTONE[] =
      {
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55,
          0xAA,
          0xAA,
          0xAA,
          0xAA,
          0x55,
          0x55,
          0x55,
          0x55 };

  }
}

#endif
