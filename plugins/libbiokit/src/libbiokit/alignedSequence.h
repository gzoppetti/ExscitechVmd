/*
 * University of Illinois Open Source License
 * Copyright 2007 Luthey-Schulten Group, 
 * All rights reserved.
 * 
 * Developed by: Luthey-Schulten Group
 * 			     University of Illinois at Urbana-Champaign
 * 			     http://www.scs.uiuc.edu/~schulten
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the Software), to deal with 
 * the Software without restriction, including without limitation the rights to 
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 * of the Software, and to permit persons to whom the Software is furnished to 
 * do so, subject to the following conditions:
 * 
 * - Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimers.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimers in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * - Neither the names of the Luthey-Schulten Group, University of Illinois at
 * Urbana-Champaign, nor the names of its contributors may be used to endorse or
 * promote products derived from this Software without specific prior written
 * permission.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL 
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 * OTHER DEALINGS WITH THE SOFTWARE.
 *
 * Author(s): John Eargle, Elijah Roberts
 */

#ifndef ALIGNEDSEQUENCE_H
#define ALIGNEDSEQUENCE_H

#include "sequence.h"
#include "ShortIntList.h"

class ShortIntList;
class Symbol;
class Alphabet;

class AlignedSequence : public Sequence {
    
    public:
    AlignedSequence(Alphabet* alphabet, const char* name=0) : Sequence(alphabet, name) {}
    virtual ~AlignedSequence() {}

    virtual int getSize();
    virtual int getNumberPositions();
    virtual int getNumberResidues();
    virtual Symbol& get(int positionIndex);
    virtual Symbol& getPosition(int positionIndex);
    virtual Symbol& getResidue(int residueIndex);
    virtual int getResidueForPosition(int positionIndex);
    virtual int getPositionForResidue(int residueIndex);
    virtual void addGap();

    protected:
    virtual void addSymbolIndex(int symbolIndex);
    

    private:
    ShortIntList residueToPosition;
    ShortIntList positionToResidue;
    
    friend class AlignedSequenceTest;
};

#endif
