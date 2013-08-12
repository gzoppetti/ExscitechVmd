#ifndef GAME_LOOP_HPP_
#define GAME_LOOP_HPP_

#include <QtCore/QTime>

#define CALL_MEMBER_FUNCTION(object, ptrToMember)  ((object).*(ptrToMember))

namespace Exscitech
{
  template<typename GameType>
  class GameLoop
  {
    typedef void (GameType::*UpdateCallbackType) (int timeInMs);

  public:

    GameLoop (int updatesPerSec, int maxFramesToSkip) :
      m_maxFramesToSkip (maxFramesToSkip)
    {
      m_timePerUpdateInMs = 1000 / updatesPerSec;
    }

    void
    setCallback (GameType* game, UpdateCallbackType updateCallback)
    {
      m_game = game;
      m_updateCallback = updateCallback;
    }

    void
    start ()
    {
      m_timer.start ();
      m_lastUpdateTimeInMs = m_timer.elapsed ();
      m_nextUpdateTimeInMs = m_lastUpdateTimeInMs;
    }

    void
    update ()
    {
      int numFramesSkipped = 0;
      int currentTimeInMs = m_timer.elapsed ();
      // Update the game at UPDATES_PER_SEC
      while (currentTimeInMs > m_nextUpdateTimeInMs && numFramesSkipped
          < m_maxFramesToSkip)
      {
        // Due for game update and haven't skipped too many frames (draw calls)
        // Call back to game to update itself
        int deltaTime = currentTimeInMs - m_lastUpdateTimeInMs;
        CALL_MEMBER_FUNCTION (*m_game, m_updateCallback) (deltaTime);

        m_lastUpdateTimeInMs = currentTimeInMs;
        m_nextUpdateTimeInMs += m_timePerUpdateInMs;
        ++numFramesSkipped;
        currentTimeInMs = m_timer.elapsed ();
      }
    }

  private:

    int m_timePerUpdateInMs;
    int m_maxFramesToSkip;
    int m_lastUpdateTimeInMs;
    int m_nextUpdateTimeInMs;

    GameType* m_game;
    UpdateCallbackType m_updateCallback;

    QTime m_timer;
  };
}

#endif 
