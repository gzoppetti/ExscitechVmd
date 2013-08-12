#ifndef SIGNAL_HANDLER_HPP_
#define SIGNAL_HANDLER_HPP_

#include <boost/signals2.hpp>
#include <QtCore/QObject>

#include <cstdio>

namespace Exscitech
{
  class SignalHandler : public QObject
  {
  Q_OBJECT

    typedef boost::signals2::signal<void(const QObject*)> SignalType;
    typedef boost::signals2::signal<void(const QObject*)>::slot_type SlotType;

  public:

    SignalHandler (const SlotType& callback, QObject* signalSource, QObject* parent = NULL) :
       QObject (parent), m_slotFunction (callback), m_signalSource (signalSource)
    {
    }

  public:

    static void
    connect (QObject* signalSrcObject, const char* signal, const SlotType& callback)
    {
      SignalHandler* handler = new SignalHandler (callback, signalSrcObject);
      QObject::connect (signalSrcObject, signal, handler, SLOT (emitSignal ()));
    }

  public slots:

    void
    emitSignal () const
    {
      m_slotFunction (m_signalSource);
    }

  private:

    SlotType m_slotFunction;
    QObject* m_signalSource;

  };
}

#endif 
