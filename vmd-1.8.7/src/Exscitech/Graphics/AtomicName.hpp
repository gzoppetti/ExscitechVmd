#ifndef ATOMICNAME_HPP_
#define ATOMICNAME_HPP_

#include <ctype.h>

namespace Exscitech
{
struct AtomicName
{
  AtomicName(const char* const characters)
  {
    name[0] = characters[0];
    if (std::isalpha(characters[1]))
    {
      name[1] = characters[1];

      if (std::isalpha(characters[2]))
      {
        name[2] = characters[2];
      }
      else
      {
        name[2] = '\0';
      }
    }
    else
    {
      name[1] = '\0';
      name[2] = '\0';
    }
 }

  char&
  operator[](int index)
  {
    return name[index];
  }

  const char&
  operator[](int index) const
  {
    return name[index];
  }

  bool
  operator<(const AtomicName& other) const
  {
    if (name[0] != other[0])
    {
      return name[0] < other[0];
    }

    if (name[1] != other[1])
    {
      return name[1] < other[1];
    }

    return name[2] < other[2];
  }

  bool
  operator=(const AtomicName& other)
  {
    return (name[0] == other[0] && name[1] == other[1] && name[2] == other[2]);
  }

  char name[3];
};

}
#endif
