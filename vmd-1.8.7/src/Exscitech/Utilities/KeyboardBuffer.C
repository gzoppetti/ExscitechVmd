#include <algorithm>

#include "Exscitech/Utilities/KeyboardBuffer.hpp"

namespace Exscitech
{
  KeyboardBuffer::KeyboardBuffer ()
  {
    std::fill (&m_keys[0], &m_keys[BUFFER_SIZE], false);
  }

  bool
  KeyboardBuffer::isKeyDown (int key)
  {
    return (key < BUFFER_SIZE) ? m_keys[key] : false;
  }

  void
  KeyboardBuffer::press (int key)
  {
    if (key < BUFFER_SIZE)
      m_keys[key] = true;
  }

  void
  KeyboardBuffer::release (int key)
  {
    if (key < BUFFER_SIZE)
      m_keys[key] = false;
  }

  void
  KeyboardBuffer::clear ()
  {
    std::fill (&m_keys[0], &m_keys[BUFFER_SIZE], false);
  }
}
