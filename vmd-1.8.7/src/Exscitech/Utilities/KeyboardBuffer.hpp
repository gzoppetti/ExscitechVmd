#ifndef KEYBOARDBUFFER_HPP_
#define KEYBOARDBUFFER_HPP_

namespace Exscitech
{
  class KeyboardBuffer
  {
  public:

    KeyboardBuffer();

    bool
    isKeyDown(int key);

    void
    press(int key);

    void
    release(int key);

    void
    clear();

  private:

    static const int BUFFER_SIZE = 256;

  private:

    bool m_keys[BUFFER_SIZE];

  };
}
#endif
