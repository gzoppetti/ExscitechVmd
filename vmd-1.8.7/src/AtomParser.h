/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: AtomParser.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.23 $	$Date: 2009/05/05 19:38:39 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Basic data types for the yacc/lex interface and for the parse tree
 *
 ***************************************************************************/
#ifndef ATOMPARSER_H
#define ATOMPARSER_H

#include "JString.h"

// the idea about strings is that some can be regexed.
//  "This" is a double-quoted string  -- can do regex
//  'this' is a single-quoted string  -- don't do regex
//   this  is a raw, or unquoted, string -- don't do regex
enum  atomparser_stringtype {DQ_STRING, SQ_STRING, RAW_STRING};

/// stores an atom parser string with its string type enumeration
typedef struct atomparser_string {
  atomparser_stringtype st;
  JString s;
} atomparser_string;

/// Each node of the parse tree contains all data needed for that description
typedef struct atomparser_node {
   int node_type;   ///< these are token types, e.g. 'AND', 'WITHIN', 
                    ///< defined in AtomParser.y/y.tab.h
   int extra_type;  ///< for weird things like distinguishing
                    ///< 'index 5 to 7' from 'index 5 7'
   double dval;     ///< floating point value (if any)
   int ival;        ///< integer value (if any)
   atomparser_string sele;  ///< if this is a string, what kind of string?
   atomparser_node *left;   ///< link to left branch of parse tree
   atomparser_node *right;  ///< link to right branch of parse tree

   /// constructor
   atomparser_node(int nnode_t, int nextra_t = -1) {
      node_type = nnode_t;
      extra_type = nextra_t;
      left = NULL;
      right = NULL;
   }

   /// destructor
   /// XXX This is recursive and can fail on massive selection strings,
   ///     e.g. a list of 500,000 residue names can blow the stack.
   ///     This needs to be redesigned to avoid recursion
   ~atomparser_node(void) {  // destructor
      if (left) delete left;
      if (right) delete right;
   }
} atomparser_node;

/// contains the final parse tree, or NULL if there was an error
extern atomparser_node *atomparser_result;

/// given the string and its length, return the index in the symbol table
/// or -1 if it isn't there
int atomparser_yylookup(const char *s, int len);

/// contains the location of the string to parse
extern char *atomparser_yystring;

/// contains the list of the functions and keywords
extern class SymbolTable *atomparser_symbols;

#endif

