
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
	int ival;
	double dval;
	atomparser_node *node;
} YYSTYPE;
extern YYSTYPE yylval;
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
