
# line 2 "AtomParser.y"
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
 *	$RCSfile: AtomParser.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.22 $	$Date: 2009/05/04 21:42:16 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  a parser for atom selections
 *
 ***************************************************************************/



#include <stdio.h>
#include <string.h>
#include "AtomParser.h"
#include "Inform.h"

#if !defined(_MSC_VER) && !defined(MINGW)
extern "C" int yyparse();
#endif
extern "C" void yyerror(const char *s);
extern "C" int yylex();

atomparser_node *atomparser_result;

# line 39 "AtomParser.y"
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
	int ival;
	double dval;
	atomparser_node *node;
} YYSTYPE;
# define KEY 257
# define WITHIN 258
# define EXWITHIN 259
# define PBWITHIN 260
# define WITHINBONDS 261
# define MAXRINGSIZE 262
# define RINGSIZE 263
# define WHERE 264
# define FUNC 265
# define STRFCTN 266
# define SAME 267
# define SINGLE 268
# define FROM 269
# define OF 270
# define AS 271
# define THROUGH 272
# define PARSEERROR 273
# define RANGE 274
# define FLOATVAL 275
# define INTVAL 276
# define STRWORD 277
# define COMPARE 278
# define OR 279
# define AND 280
# define LT 281
# define LE 282
# define EQ 283
# define GE 284
# define GT 285
# define NE 286
# define NLT 287
# define NLE 288
# define NEQ 289
# define NGE 290
# define NGT 291
# define NNE 292
# define SLT 293
# define SLE 294
# define SEQ 295
# define SGE 296
# define SGT 297
# define SNE 298
# define MATCH 299
# define ADD 300
# define SUB 301
# define MULT 302
# define DIV 303
# define MOD 304
# define EXP 305
# define nonassoc 306
# define NOT 307
# define UMINUS 308

#if !defined(_MSC_VER) && !defined(MINGW) && !defined(ARCH_TRU64)
#include <inttypes.h>
#endif

#if 1
//#ifdef __STDC__
#include <stdlib.h>
#include <string.h>
#define	YYCONST	const
#else
#include <malloc.h>
#include <memory.h>
#define	YYCONST
#endif

#if !defined(_MSC_VER) && !defined(MINGW) && !defined(__APPLE__)
#include <values.h>
#endif

#if defined(__cplusplus) || defined(__STDC__)

#if defined(__cplusplus) && defined(__EXTERN_C__)
extern "C" {
#endif
#ifndef yyerror
#if defined(__cplusplus)
	void yyerror(YYCONST char *);
#endif
#endif
#ifndef yylex
	int yylex(void);
#endif
	int yyparse(void);
#if defined(__cplusplus) && defined(__EXTERN_C__)
}
#endif

#endif

#define yyclearin yychar = -1
#define yyerrok yyerrflag = 0
extern int yychar;
extern int yyerrflag;
YYSTYPE yylval;
YYSTYPE yyval;
typedef int yytabelem;
#ifndef YYMAXDEPTH
#define YYMAXDEPTH 150
#endif
#if YYMAXDEPTH > 0
int yy_yys[YYMAXDEPTH], *yys = yy_yys;
YYSTYPE yy_yyv[YYMAXDEPTH], *yyv = yy_yyv;
#else	/* user does initial allocation */
int *yys;
YYSTYPE *yyv;
#endif
static int yymaxdepth = YYMAXDEPTH;
# define YYERRCODE 256

# line 275 "AtomParser.y"


extern "C" void yyerror(const char *s) {
  msgErr << s << sendmsg;
}

// everything comes from a string, so there is no way to
// reset "yyin" (or whatever) to the next input
extern "C" int yywrap(void) {
  return 1;
}

static YYCONST yytabelem yyexca[] ={
-1, 0,
	0, 1,
	-2, 0,
-1, 1,
	0, -1,
	-2, 0,
	};
# define YYNPROD 58
# define YYLAST 365
static YYCONST yytabelem yyact[]={

     4,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    41,    42,    45,    46,    43,    44,
    44,     4,    41,    42,    45,    46,    43,    44,    45,    46,
    43,    44,    74,    26,    25,    25,    65,    64,    63,    32,
    33,    38,    37,    75,    36,    91,    82,    79,    78,   102,
    77,    76,    81,    80,    39,    75,     2,    62,    61,    71,
    60,    27,    29,    30,     1,     8,    31,     7,    40,     0,
     0,    16,     0,     0,     0,     0,    28,    34,    35,    66,
     0,     0,    72,    73,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,    67,    70,     0,     0,     0,     0,
     0,     0,     0,    68,     0,     0,     0,     0,     0,     0,
     0,     0,    83,    84,    85,    86,    87,    88,    89,    90,
     0,     0,     0,     0,     0,    90,     0,     0,     0,     0,
     0,     0,     0,    94,    95,    96,    97,    98,    99,   100,
    92,     0,     0,    93,     0,     0,     0,     0,     0,   101,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     3,    17,     9,    10,
    11,    12,    13,    14,     0,    24,    18,    15,     6,     0,
     0,     0,     0,     0,     0,    19,    20,    21,    17,     9,
    10,    11,    12,    13,    14,     0,    24,    18,    15,     6,
     0,     0,     0,     0,     0,     0,    19,    20,    21,     0,
    23,    22,     0,     0,     0,     0,     0,     5,     0,     0,
    26,    25,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    23,    22,     0,     0,     0,     0,     0,     5,    47,
    48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
    58,    59,    41,    42,    45,    46,    43,    44,    41,    42,
    45,    46,    43,    44,    41,    42,    45,    46,    43,    44,
    69,     0,     0,     0,     0,     0,     0,     0,    24,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    19,    20,
    21,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,    23,    22 };
static YYCONST yytabelem yypact[]={

   -40,-10000000,  -246,-10000000,   -19,   -19,   -19,-10000000,-10000000,  -236,
  -236,  -236,  -232,  -234,  -235,  -203,  -286,  -239,  -239,-10000000,
-10000000,-10000000,    63,    63,    19,   -19,   -19,    -9,     2,-10000000,
  -246,  -219,-10000000,-10000000,  -220,  -222,  -223,  -216,  -217,  -225,
    63,    63,    63,    63,    63,    63,    63,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
  -239,-10000000,  -227,-10000000,-10000000,-10000000,  -239,-10000000,    63,-10000000,
-10000000,    63,-10000000,  -245,-10000000,-10000000,   -19,   -19,   -19,   -19,
   -19,   -19,   -19,  -278,  -274,  -274,  -285,-10000000,  -285,  -285,
-10000000,  -239,    14,     8,  -246,  -246,  -246,  -246,  -246,  -246,
  -246,-10000000,-10000000 };
static YYCONST yytabelem yypgo[]={

     0,    68,    56,    71,    67,    60,    57,    58,    65,    66,
    64 };
static YYCONST yytabelem yyr1[]={

     0,    10,    10,    10,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     4,     8,     6,     6,     6,     7,     7,     5,     5,     9,
     9,     3,     3,     3,     3,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1 };
static YYCONST yytabelem yyr2[]={

     0,     1,     3,     3,     7,     5,     3,     5,     7,     7,
     3,     3,     9,     9,     9,     9,     9,     9,     9,     7,
     5,     5,     3,     3,     3,     3,     7,     3,     5,     3,
     3,     3,     3,     3,     7,     7,     7,     5,     5,     7,
     7,     7,     7,     3,     9,     3,     3,     3,     3,     3,
     3,     3,     3,     3,     3,     3,     3,     3 };
static YYCONST yytabelem yychk[]={

-10000000,   -10,    -2,   256,    40,   307,   268,    -4,    -8,   258,
   259,   260,   261,   262,   263,   267,    -3,   257,   266,   275,
   276,   277,   301,   300,   265,   280,   279,    -2,    -3,    -2,
    -2,    -9,   275,   276,    -9,    -9,   276,   276,   276,   257,
    -1,   300,   301,   304,   305,   302,   303,   287,   288,   289,
   290,   291,   292,   293,   294,   295,   296,   297,   298,   299,
    -5,    -7,    -6,   277,   276,   275,    -5,    -3,    40,   257,
    -3,    40,    -2,    -2,    41,    41,   270,   270,   270,   270,
   269,   269,   271,    -3,    -3,    -3,    -3,    -3,    -3,    -3,
    -7,   272,    -3,    -3,    -2,    -2,    -2,    -2,    -2,    -2,
    -2,    -6,    41 };
static YYCONST yytabelem yydef[]={

    -2,    -2,     2,     3,     0,     0,     6,    10,    11,     0,
     0,     0,     0,     0,     0,     0,     0,    43,     0,    31,
    32,    33,     0,     0,     0,     0,     0,     0,     0,     5,
     7,     0,    29,    30,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,    45,    46,    47,
    48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
    20,    27,    25,    22,    23,    24,    21,    37,     0,    43,
    38,     0,     8,     9,     4,    34,     0,     0,     0,     0,
     0,     0,     0,    19,    35,    36,    39,    40,    41,    42,
    28,     0,     0,     0,    12,    13,    14,    15,    16,    17,
    18,    26,    44 };
typedef struct
#ifdef __cplusplus
	yytoktype
#endif
{
#ifdef __cplusplus
const
#endif
char *t_name; int t_val; } yytoktype;
#ifndef YYDEBUG
#	define YYDEBUG	0	/* don't allow debugging */
#endif

#if YYDEBUG

yytoktype yytoks[] =
{
	"KEY",	257,
	"WITHIN",	258,
	"EXWITHIN",	259,
	"PBWITHIN",	260,
	"WITHINBONDS",	261,
	"MAXRINGSIZE",	262,
	"RINGSIZE",	263,
	"WHERE",	264,
	"FUNC",	265,
	"STRFCTN",	266,
	"SAME",	267,
	"SINGLE",	268,
	"FROM",	269,
	"OF",	270,
	"AS",	271,
	"THROUGH",	272,
	"PARSEERROR",	273,
	"RANGE",	274,
	"FLOATVAL",	275,
	"INTVAL",	276,
	"STRWORD",	277,
	"COMPARE",	278,
	"OR",	279,
	"AND",	280,
	"LT",	281,
	"LE",	282,
	"EQ",	283,
	"GE",	284,
	"GT",	285,
	"NE",	286,
	"NLT",	287,
	"NLE",	288,
	"NEQ",	289,
	"NGE",	290,
	"NGT",	291,
	"NNE",	292,
	"SLT",	293,
	"SLE",	294,
	"SEQ",	295,
	"SGE",	296,
	"SGT",	297,
	"SNE",	298,
	"MATCH",	299,
	"ADD",	300,
	"SUB",	301,
	"MULT",	302,
	"DIV",	303,
	"MOD",	304,
	"EXP",	305,
	"nonassoc",	306,
	"NOT",	307,
	"UMINUS",	308,
	"-unknown-",	-1	/* ends search */
};

#ifdef __cplusplus
const
#endif
char * yyreds[] =
{
	"-no such reduction-",
	"selection_list : /* empty */",
	"selection_list : selection",
	"selection_list : error",
	"selection : '(' selection ')'",
	"selection : NOT selection",
	"selection : SINGLE",
	"selection : SINGLE selection",
	"selection : selection AND selection",
	"selection : selection OR selection",
	"selection : keyword_list",
	"selection : strfctn_list",
	"selection : WITHIN number OF selection",
	"selection : EXWITHIN number OF selection",
	"selection : PBWITHIN number OF selection",
	"selection : WITHINBONDS INTVAL OF selection",
	"selection : MAXRINGSIZE INTVAL FROM selection",
	"selection : RINGSIZE INTVAL FROM selection",
	"selection : SAME KEY AS selection",
	"selection : expression compare expression",
	"keyword_list : KEY string_list",
	"strfctn_list : STRFCTN string_list",
	"word : STRWORD",
	"word : INTVAL",
	"word : FLOATVAL",
	"string_list_ele : word",
	"string_list_ele : word THROUGH word",
	"string_list : string_list_ele",
	"string_list : string_list string_list_ele",
	"number : FLOATVAL",
	"number : INTVAL",
	"expression : FLOATVAL",
	"expression : INTVAL",
	"expression : STRWORD",
	"expression : '(' expression ')'",
	"expression : expression ADD expression",
	"expression : expression SUB expression",
	"expression : SUB expression",
	"expression : ADD expression",
	"expression : expression MOD expression",
	"expression : expression EXP expression",
	"expression : expression MULT expression",
	"expression : expression DIV expression",
	"expression : KEY",
	"expression : FUNC '(' expression ')'",
	"compare : NLT",
	"compare : NLE",
	"compare : NEQ",
	"compare : NGE",
	"compare : NGT",
	"compare : NNE",
	"compare : SLT",
	"compare : SLE",
	"compare : SEQ",
	"compare : SGE",
	"compare : SGT",
	"compare : SNE",
	"compare : MATCH",
};
#endif /* YYDEBUG */
# line	1 "/usr/ccs/bin/yaccpar"
/*
 * Copyright (c) 1993 by Sun Microsystems, Inc.
 */

#pragma ident	"@(#)yaccpar	6.16	99/01/20 SMI"

/*
** Skeleton parser driver for yacc output
*/

/*
** yacc user known macros and defines
*/
#define YYERROR		goto yyerrlab
#define YYACCEPT	return(0)
#define YYABORT		return(1)
#define YYBACKUP( newtoken, newvalue )\
{\
	if ( yychar >= 0 || ( yyr2[ yytmp ] >> 1 ) != 1 )\
	{\
		yyerror( "syntax error - cannot backup" );\
		goto yyerrlab;\
	}\
	yychar = newtoken;\
	yystate = *yyps;\
	yylval = newvalue;\
	goto yynewstate;\
}
#define YYRECOVERING()	(!!yyerrflag)
#define YYNEW(type)	malloc(sizeof(type) * yynewmax)
#define YYCOPY(to, from, type) \
	(type *) memcpy(to, (char *) from, yymaxdepth * sizeof (type))
#define YYENLARGE( from, type) \
	(type *) realloc((char *) from, yynewmax * sizeof(type))
#ifndef YYDEBUG
#	define YYDEBUG	1	/* make debugging available */
#endif

/*
** user known globals
*/
int yydebug;			/* set to 1 to get debugging */

/*
** driver internal defines
*/
#define YYFLAG		(-10000000)

/*
** global variables used by the parser
*/
YYSTYPE *yypv;			/* top of value stack */
int *yyps;			/* top of state stack */

int yystate;			/* current state */
int yytmp;			/* extra var (lasts between blocks) */

int yynerrs;			/* number of errors */
int yyerrflag;			/* error recovery flag */
int yychar;			/* current input token number */



#ifdef YYNMBCHARS
#define YYLEX()		yycvtok(yylex())
/*
** yycvtok - return a token if i is a wchar_t value that exceeds 255.
**	If i<255, i itself is the token.  If i>255 but the neither 
**	of the 30th or 31st bit is on, i is already a token.
*/
#if defined(__STDC__) || defined(__cplusplus)
int yycvtok(int i)
#else
int yycvtok(i) int i;
#endif
{
	int first = 0;
	int last = YYNMBCHARS - 1;
	int mid;
	wchar_t j;

	if(i&0x60000000){/*Must convert to a token. */
		if( yymbchars[last].character < i ){
			return i;/*Giving up*/
		}
		while ((last>=first)&&(first>=0)) {/*Binary search loop*/
			mid = (first+last)/2;
			j = yymbchars[mid].character;
			if( j==i ){/*Found*/ 
				return yymbchars[mid].tvalue;
			}else if( j<i ){
				first = mid + 1;
			}else{
				last = mid -1;
			}
		}
		/*No entry in the table.*/
		return i;/* Giving up.*/
	}else{/* i is already a token. */
		return i;
	}
}
#else/*!YYNMBCHARS*/
#define YYLEX()		yylex()
#endif/*!YYNMBCHARS*/

/*
** yyparse - return 0 if worked, 1 if syntax error not recovered from
*/
#if defined(__STDC__) || defined(__cplusplus)
int yyparse(void)
#else
int yyparse()
#endif
{
	register YYSTYPE *yypvt = 0;	/* top of value stack for $vars */

#if defined(__cplusplus) || defined(lint)
/*
	hacks to please C++ and lint - goto's inside
	switch should never be executed
*/
	static int __yaccpar_lint_hack__ = 0;
	switch (__yaccpar_lint_hack__)
	{
		case 1: goto yyerrlab;
		case 2: goto yynewstate;
	}
#endif

	/*
	** Initialize externals - yyparse may be called more than once
	*/
	yypv = &yyv[-1];
	yyps = &yys[-1];
	yystate = 0;
	yytmp = 0;
	yynerrs = 0;
	yyerrflag = 0;
	yychar = -1;

#if YYMAXDEPTH <= 0
	if (yymaxdepth <= 0)
	{
		if ((yymaxdepth = YYEXPAND(0)) <= 0)
		{
			yyerror("yacc initialization error");
			YYABORT;
		}
	}
#endif

	{
		register YYSTYPE *yy_pv;	/* top of value stack */
		register int *yy_ps;		/* top of state stack */
		register int yy_state;		/* current state */
		register int  yy_n;		/* internal state number info */
	goto yystack;	/* moved from 6 lines above to here to please C++ */

		/*
		** get globals into registers.
		** branch to here only if YYBACKUP was called.
		*/
	yynewstate:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;
		goto yy_newstate;

		/*
		** get globals into registers.
		** either we just started, or we just finished a reduction
		*/
	yystack:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;

		/*
		** top of for (;;) loop while no reductions done
		*/
	yy_stack:
		/*
		** put a state and value onto the stacks
		*/
#if YYDEBUG
		/*
		** if debugging, look up token value in list of value vs.
		** name pairs.  0 and negative (-1) are special values.
		** Note: linear search is used since time is not a real
		** consideration while debugging.
		*/
		if ( yydebug )
		{
			register int yy_i;

			printf( "State %d, token ", yy_state );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ++yy_ps >= &yys[ yymaxdepth ] )	/* room on stack? */
		{
			/*
			** reallocate and recover.  Note that pointers
			** have to be reset, or bad things will happen
			*/
			long yyps_index = (yy_ps - yys);
			long yypv_index = (yy_pv - yyv);
			long yypvt_index = (yypvt - yyv);
			int yynewmax;
#ifdef YYEXPAND
			yynewmax = YYEXPAND(yymaxdepth);
#else
			yynewmax = 2 * yymaxdepth;	/* double table size */
			if (yymaxdepth == YYMAXDEPTH)	/* first time growth */
			{
				char *newyys = (char *)YYNEW(int);
				char *newyyv = (char *)YYNEW(YYSTYPE);
				if (newyys != 0 && newyyv != 0)
				{
					yys = YYCOPY(newyys, yys, int);
					yyv = YYCOPY(newyyv, yyv, YYSTYPE);
				}
				else
					yynewmax = 0;	/* failed */
			}
			else				/* not first time */
			{
				yys = YYENLARGE(yys, int);
				yyv = YYENLARGE(yyv, YYSTYPE);
				if (yys == 0 || yyv == 0)
					yynewmax = 0;	/* failed */
			}
#endif
			if (yynewmax <= yymaxdepth)	/* tables not expanded */
			{
				yyerror( "yacc stack overflow" );
				YYABORT;
			}
			yymaxdepth = yynewmax;

			yy_ps = yys + yyps_index;
			yy_pv = yyv + yypv_index;
			yypvt = yyv + yypvt_index;
		}
		*yy_ps = yy_state;
		*++yy_pv = yyval;

		/*
		** we have a new state - find out what to do
		*/
	yy_newstate:
		if ( ( yy_n = yypact[ yy_state ] ) <= YYFLAG )
			goto yydefault;		/* simple state */
#if YYDEBUG
		/*
		** if debugging, need to mark whether new token grabbed
		*/
		yytmp = yychar < 0;
#endif
		if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
			yychar = 0;		/* reached EOF */
#if YYDEBUG
		if ( yydebug && yytmp )
		{
			register int yy_i;

			printf( "Received token " );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ( ( yy_n += yychar ) < 0 ) || ( yy_n >= YYLAST ) )
			goto yydefault;
		if ( yychk[ yy_n = yyact[ yy_n ] ] == yychar )	/*valid shift*/
		{
			yychar = -1;
			yyval = yylval;
			yy_state = yy_n;
			if ( yyerrflag > 0 )
				yyerrflag--;
			goto yy_stack;
		}

	yydefault:
		if ( ( yy_n = yydef[ yy_state ] ) == -2 )
		{
#if YYDEBUG
			yytmp = yychar < 0;
#endif
			if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
				yychar = 0;		/* reached EOF */
#if YYDEBUG
			if ( yydebug && yytmp )
			{
				register int yy_i;

				printf( "Received token " );
				if ( yychar == 0 )
					printf( "end-of-file\n" );
				else if ( yychar < 0 )
					printf( "-none-\n" );
				else
				{
					for ( yy_i = 0;
						yytoks[yy_i].t_val >= 0;
						yy_i++ )
					{
						if ( yytoks[yy_i].t_val
							== yychar )
						{
							break;
						}
					}
					printf( "%s\n", yytoks[yy_i].t_name );
				}
			}
#endif /* YYDEBUG */
			/*
			** look through exception table
			*/
			{
				register YYCONST int *yyxi = yyexca;

				while ( ( *yyxi != -1 ) ||
					( yyxi[1] != yy_state ) )
				{
					yyxi += 2;
				}
				while ( ( *(yyxi += 2) >= 0 ) &&
					( *yyxi != yychar ) )
					;
				if ( ( yy_n = yyxi[1] ) < 0 )
					YYACCEPT;
			}
		}

		/*
		** check for syntax error
		*/
		if ( yy_n == 0 )	/* have an error */
		{
			/* no worry about speed here! */
			switch ( yyerrflag )
			{
			case 0:		/* new error */
				yyerror( "syntax error" );
				goto skip_init;
			yyerrlab:
				/*
				** get globals into registers.
				** we have a user generated syntax type error
				*/
				yy_pv = yypv;
				yy_ps = yyps;
				yy_state = yystate;
			skip_init:
				yynerrs++;
				/* FALLTHRU */
			case 1:
			case 2:		/* incompletely recovered error */
					/* try again... */
				yyerrflag = 3;
				/*
				** find state where "error" is a legal
				** shift action
				*/
				while ( yy_ps >= yys )
				{
					yy_n = yypact[ *yy_ps ] + YYERRCODE;
					if ( yy_n >= 0 && yy_n < YYLAST &&
						yychk[yyact[yy_n]] == YYERRCODE)					{
						/*
						** simulate shift of "error"
						*/
						yy_state = yyact[ yy_n ];
						goto yy_stack;
					}
					/*
					** current state has no shift on
					** "error", pop stack
					*/
#if YYDEBUG
#	define _POP_ "Error recovery pops state %d, uncovers state %d\n"
					if ( yydebug )
						printf( _POP_, *yy_ps,
							yy_ps[-1] );
#	undef _POP_
#endif
					yy_ps--;
					yy_pv--;
				}
				/*
				** there is no state on stack with "error" as
				** a valid shift.  give up.
				*/
				YYABORT;
			case 3:		/* no shift yet; eat a token */
#if YYDEBUG
				/*
				** if debugging, look up token in list of
				** pairs.  0 and negative shouldn't occur,
				** but since timing doesn't matter when
				** debugging, it doesn't hurt to leave the
				** tests here.
				*/
				if ( yydebug )
				{
					register int yy_i;

					printf( "Error recovery discards " );
					if ( yychar == 0 )
						printf( "token end-of-file\n" );
					else if ( yychar < 0 )
						printf( "token -none-\n" );
					else
					{
						for ( yy_i = 0;
							yytoks[yy_i].t_val >= 0;
							yy_i++ )
						{
							if ( yytoks[yy_i].t_val
								== yychar )
							{
								break;
							}
						}
						printf( "token %s\n",
							yytoks[yy_i].t_name );
					}
				}
#endif /* YYDEBUG */
				if ( yychar == 0 )	/* reached EOF. quit */
					YYABORT;
				yychar = -1;
				goto yy_newstate;
			}
		}/* end if ( yy_n == 0 ) */
		/*
		** reduction by production yy_n
		** put stack tops, etc. so things right after switch
		*/
#if YYDEBUG
		/*
		** if debugging, print the string that is the user's
		** specification of the reduction which is just about
		** to be done.
		*/
		if ( yydebug )
			printf( "Reduce by (%d) \"%s\"\n",
				yy_n, yyreds[ yy_n ] );
#endif
		yytmp = yy_n;			/* value to switch over */
		yypvt = yy_pv;			/* $vars top of value stack */
		/*
		** Look in goto table for next state
		** Sorry about using yy_state here as temporary
		** register variable, but why not, if it works...
		** If yyr2[ yy_n ] doesn't have the low order bit
		** set, then there is no action to be done for
		** this reduction.  So, no saving & unsaving of
		** registers done.  The only difference between the
		** code just after the if and the body of the if is
		** the goto yy_stack in the body.  This way the test
		** can be made before the choice of what to do is needed.
		*/
		{
			/* length of production doubled with extra bit */
			register int yy_len = yyr2[ yy_n ];

			if ( !( yy_len & 01 ) )
			{
				yy_len >>= 1;
				yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
				yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
					*( yy_ps -= yy_len ) + 1;
				if ( yy_state >= YYLAST ||
					yychk[ yy_state =
					yyact[ yy_state ] ] != -yy_n )
				{
					yy_state = yyact[ yypgo[ yy_n ] ];
				}
				goto yy_stack;
			}
			yy_len >>= 1;
			yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
			yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
				*( yy_ps -= yy_len ) + 1;
			if ( yy_state >= YYLAST ||
				yychk[ yy_state = yyact[ yy_state ] ] != -yy_n )
			{
				yy_state = yyact[ yypgo[ yy_n ] ];
			}
		}
					/* save until reenter driver code */
		yystate = yy_state;
		yyps = yy_ps;
		yypv = yy_pv;
	}
	/*
	** code supplied by user is placed in this switch
	*/
	switch( yytmp )
	{
		
case 1:
# line 76 "AtomParser.y"
{// printf("Blank line.\n");
					  atomparser_result =  NULL;
					} break;
case 2:
# line 79 "AtomParser.y"
{ //printf("Parsed a line\n");
					  if (*atomparser_yystring != 0) {
    msgErr << "Selection terminated too early" << sendmsg;
    if (yypvt[-0].node) delete yypvt[-0].node;
    yypvt[-0].node = NULL;
					  }
					  atomparser_result = yypvt[-0].node;
					} break;
case 3:
# line 87 "AtomParser.y"
{ //printf("Error occured\n");
					  atomparser_result = NULL;
//					  yyerrok;
					} break;
case 4:
# line 93 "AtomParser.y"
{ // printf("Parens\n");
					  yyval.node = yypvt[-1].node;
					} break;
case 5:
# line 96 "AtomParser.y"
{ yyval.node = new atomparser_node(NOT);
					  yyval.node->left = yypvt[-0].node;
					} break;
case 6:
# line 99 "AtomParser.y"
{ yyval.node = yypvt[-0].node; } break;
case 7:
# line 100 "AtomParser.y"
{ yyval.node = new atomparser_node(AND);
					  yyval.node->left = yypvt[-1].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 8:
# line 104 "AtomParser.y"
{ //printf("AND\n");
					  yyval.node = new atomparser_node(AND);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 9:
# line 109 "AtomParser.y"
{ //printf("OR\n");
					  yyval.node = new atomparser_node(OR);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 10:
# line 114 "AtomParser.y"
{ yyval.node = yypvt[-0].node; } break;
case 11:
# line 115 "AtomParser.y"
{ yyval.node = yypvt[-0].node; } break;
case 12:
# line 116 "AtomParser.y"
{ yyval.node = new atomparser_node(WITHIN);
					  yyval.node->left = yypvt[-0].node;
					  yyval.node->dval = yypvt[-2].dval;
					} break;
case 13:
# line 120 "AtomParser.y"
{ yyval.node = new atomparser_node(EXWITHIN);
					  yyval.node->left = yypvt[-0].node;
					  yyval.node->dval = yypvt[-2].dval;
					} break;
case 14:
# line 124 "AtomParser.y"
{ yyval.node = new atomparser_node(PBWITHIN);
					  yyval.node->left = yypvt[-0].node;
					  yyval.node->dval = yypvt[-2].dval;
					} break;
case 15:
# line 128 "AtomParser.y"
{ yyval.node = new atomparser_node(WITHINBONDS);
					  yyval.node->left = yypvt[-0].node;
					  yyval.node->ival = yypvt[-2].ival;
					} break;
case 16:
# line 132 "AtomParser.y"
{ yyval.node = new atomparser_node(MAXRINGSIZE);
					  yyval.node->left = yypvt[-0].node;
					  yyval.node->ival = yypvt[-2].ival;
					} break;
case 17:
# line 136 "AtomParser.y"
{ yyval.node = new atomparser_node(RINGSIZE);
					  yyval.node->left = yypvt[-0].node;
					  yyval.node->ival = yypvt[-2].ival;
					} break;
case 18:
# line 140 "AtomParser.y"
{ yyval.node = yypvt[-2].node;
					  yyval.node->node_type = SAME;
					  yyval.node->left = yypvt[-0].node;
					} break;
case 19:
# line 144 "AtomParser.y"
{yyval.node = new atomparser_node(COMPARE);
					  yyval.node -> ival = yypvt[-1].ival;
					  yyval.node -> left = yypvt[-2].node;
					  yyval.node -> right = yypvt[-0].node;
					} break;
case 20:
# line 151 "AtomParser.y"
{
					  yypvt[-1].node -> left = yypvt[-0].node;
					  yypvt[-0].node -> right = NULL;
					  yyval.node = yypvt[-1].node;
					} break;
case 21:
# line 158 "AtomParser.y"
{  yypvt[-1].node -> left = yypvt[-0].node;
					   yypvt[-0].node -> right = NULL;
					   yyval.node = yypvt[-1].node;
					} break;
case 22:
# line 164 "AtomParser.y"
{ 
					  yyval.node = yypvt[-0].node;
					  //printf("Single\n");
					} break;
case 23:
# line 168 "AtomParser.y"
{ yyval.node = new atomparser_node(STRWORD);
					  //printf("converted int\n");
					  char s[100];
					  sprintf(s, "%ld", (long) yypvt[-0].ival);
					  yyval.node -> sele.s = s;
					  yyval.node -> sele.st = RAW_STRING;
					} break;
case 24:
# line 175 "AtomParser.y"
{ yyval.node = new atomparser_node(STRWORD);
					  char s[100];
					  sprintf(s, "%f", (double) yypvt[-0].dval);
					  yyval.node -> sele.s = s;
					  yyval.node -> sele.st = RAW_STRING;
					} break;
case 25:
# line 183 "AtomParser.y"
{ 
					  yypvt[-0].node -> right = yypvt[-0].node;
   					  yyval.node = yypvt[-0].node; 
					} break;
case 26:
# line 187 "AtomParser.y"
{ yypvt[-2].node -> right = yypvt[-0].node;
					  yypvt[-2].node -> left = yypvt[-0].node;
					  yypvt[-2].node -> extra_type = 1;
					  yyval.node = yypvt[-2].node;
					  //printf("Using through\n");
					} break;
case 27:
# line 196 "AtomParser.y"
{ yyval.node = yypvt[-0].node; } break;
case 28:
# line 197 "AtomParser.y"
{ /* copy the new word on the list */
               /* like a linked list, with head's right pointed to the end */
	       /* element and head's left pointed to the second element    */
				          yypvt[-1].node -> right -> left = yypvt[-0].node;
					  yypvt[-1].node -> right = yypvt[-0].node -> right;
					  yypvt[-0].node -> right = NULL;
					  // printf("Returning\n");
					  yyval.node = yypvt[-1].node;
				       } break;
case 29:
# line 208 "AtomParser.y"
{ yyval.dval = yypvt[-0].dval;// printf("## %lf\n", yyval.dval);
					} break;
case 30:
# line 210 "AtomParser.y"
{ yyval.dval = (double) yypvt[-0].ival; 
					  // printf("# %lf\n", yyval.dval);
					} break;
case 31:
# line 215 "AtomParser.y"
{ yyval.node = new atomparser_node(FLOATVAL);
					  yyval.node->dval = yypvt[-0].dval; 
					} break;
case 32:
# line 218 "AtomParser.y"
{ yyval.node = new atomparser_node(INTVAL);
					  yyval.node->ival = yypvt[-0].ival; 
					} break;
case 33:
# line 221 "AtomParser.y"
{ yyval.node = yypvt[-0].node; 
					} break;
case 34:
# line 223 "AtomParser.y"
{ yyval.node = yypvt[-1].node; } break;
case 35:
# line 224 "AtomParser.y"
{ yyval.node = new atomparser_node(ADD);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 36:
# line 228 "AtomParser.y"
{ yyval.node = new atomparser_node(SUB);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 37:
# line 232 "AtomParser.y"
{ yyval.node = new atomparser_node(UMINUS);
					  yyval.node->left = yypvt[-0].node;
					} break;
case 38:
# line 235 "AtomParser.y"
{ yyval.node = yypvt[-0].node;
					} break;
case 39:
# line 237 "AtomParser.y"
{ yyval.node = new atomparser_node(MOD);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 40:
# line 241 "AtomParser.y"
{ yyval.node = new atomparser_node(EXP);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 41:
# line 245 "AtomParser.y"
{ yyval.node = new atomparser_node(MULT);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 42:
# line 249 "AtomParser.y"
{ yyval.node = new atomparser_node(DIV);
					  yyval.node->left = yypvt[-2].node;
					  yyval.node->right = yypvt[-0].node;
					} break;
case 43:
# line 253 "AtomParser.y"
{ yyval.node = yypvt[-0].node; } break;
case 44:
# line 254 "AtomParser.y"
{ yypvt[-3].node->left = yypvt[-1].node;
					  yyval.node = yypvt[-3].node;
					} break;
case 45:
# line 259 "AtomParser.y"
{ yyval.ival = NLT; } break;
case 46:
# line 260 "AtomParser.y"
{ yyval.ival = NLE; } break;
case 47:
# line 261 "AtomParser.y"
{ yyval.ival = NEQ; } break;
case 48:
# line 262 "AtomParser.y"
{ yyval.ival = NGE; } break;
case 49:
# line 263 "AtomParser.y"
{ yyval.ival = NGT; } break;
case 50:
# line 264 "AtomParser.y"
{ yyval.ival = NNE; } break;
case 51:
# line 265 "AtomParser.y"
{ yyval.ival = SLT; } break;
case 52:
# line 266 "AtomParser.y"
{ yyval.ival = SLE; } break;
case 53:
# line 267 "AtomParser.y"
{ yyval.ival = SEQ; } break;
case 54:
# line 268 "AtomParser.y"
{ yyval.ival = SGE; } break;
case 55:
# line 269 "AtomParser.y"
{ yyval.ival = SGT; } break;
case 56:
# line 270 "AtomParser.y"
{ yyval.ival = SNE; } break;
case 57:
# line 271 "AtomParser.y"
{ yyval.ival = MATCH; } break;
# line	531 "/usr/ccs/bin/yaccpar"
	}
	goto yystack;		/* reset registers in driver code */
}

