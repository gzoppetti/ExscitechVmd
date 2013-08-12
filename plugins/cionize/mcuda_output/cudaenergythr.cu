typedef long unsigned int size_t;
typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
struct cudaenergythr_cu_264
{
int __val[2];
};

typedef struct cudaenergythr_cu_264 __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;
typedef int __daddr_t;
typedef long int __swblk_t;
typedef int __key_t;
typedef int __clockid_t;
typedef int __timer_t;
typedef long int __blksize_t;
typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;
typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;
typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;
typedef long int __ssize_t;
typedef __off64_t __loff_t;
typedef __quad_t * __qaddr_t;
typedef char * __caddr_t;
typedef long int __intptr_t;
typedef unsigned int __socklen_t;
typedef struct _IO_FILE FILE;
typedef struct _IO_FILE __FILE;
typedef int wchar_t;
typedef unsigned int wint_t;
union cudaenergythr_cu_570
{
wint_t __wch;
char __wchb[4];
};

struct cudaenergythr_cu_562
{
int __count;
union cudaenergythr_cu_570 __value;
};

typedef struct cudaenergythr_cu_562 __mbstate_t;
struct cudaenergythr_cu_596
{
__off_t __pos;
__mbstate_t __state;
};

typedef struct cudaenergythr_cu_596 _G_fpos_t;
struct cudaenergythr_cu_615
{
__off64_t __pos;
__mbstate_t __state;
};

typedef struct cudaenergythr_cu_615 _G_fpos64_t;
enum cudaenergythr_cu_632 { __GCONV_OK = 0, __GCONV_NOCONV, __GCONV_NODB, __GCONV_NOMEM, __GCONV_EMPTY_INPUT, __GCONV_FULL_OUTPUT, __GCONV_ILLEGAL_INPUT, __GCONV_INCOMPLETE_INPUT, __GCONV_ILLEGAL_DESCRIPTOR, __GCONV_INTERNAL_ERROR };
enum cudaenergythr_cu_668 { __GCONV_IS_LAST = 1, __GCONV_IGNORE_ERRORS = 2 };
struct __gconv_step;

struct __gconv_step_data;

struct __gconv_loaded_object;

struct __gconv_trans_data;

typedef int (* __gconv_fct)(struct __gconv_step * , struct __gconv_step_data * , const unsigned char *  * , const unsigned char * , unsigned char *  * , size_t * , int , int );
typedef wint_t (* __gconv_btowc_fct)(struct __gconv_step * , unsigned char );
typedef int (* __gconv_init_fct)(struct __gconv_step * );
typedef void (* __gconv_end_fct)(struct __gconv_step * );
typedef int (* __gconv_trans_fct)(struct __gconv_step * , struct __gconv_step_data * , void * , const unsigned char * , const unsigned char *  * , const unsigned char * , unsigned char *  * , size_t * );
typedef int (* __gconv_trans_context_fct)(void * , const unsigned char * , const unsigned char * , unsigned char * , unsigned char * );
typedef int (* __gconv_trans_query_fct)(const char * , const char *  *  * , size_t * );
typedef int (* __gconv_trans_init_fct)(void *  * , const char * );
typedef void (* __gconv_trans_end_fct)(void * );
struct __gconv_trans_data
{
__gconv_trans_fct __trans_fct;
__gconv_trans_context_fct __trans_context_fct;
__gconv_trans_end_fct __trans_end_fct;
void * __data;
struct __gconv_trans_data * __next;
};

struct __gconv_trans_data;

struct __gconv_step
{
struct __gconv_loaded_object * __shlib_handle;
const char * __modname;
int __counter;
char * __from_name;
char * __to_name;
__gconv_fct __fct;
__gconv_btowc_fct __btowc_fct;
__gconv_init_fct __init_fct;
__gconv_end_fct __end_fct;
int __min_needed_from;
int __max_needed_from;
int __min_needed_to;
int __max_needed_to;
int __stateful;
void * __data;
};

struct __gconv_loaded_object;

struct __gconv_step_data
{
unsigned char * __outbuf;
unsigned char * __outbufend;
int __flags;
int __invocation_counter;
int __internal_use;
__mbstate_t * __statep;
__mbstate_t __state;
struct __gconv_trans_data * __trans;
};

struct __gconv_trans_data;

struct __gconv_info
{
size_t __nsteps;
struct __gconv_step * __steps;
struct __gconv_step_data __data[];
};

typedef struct __gconv_info * __gconv_t;
struct cudaenergythr_cu_1218
{
struct __gconv_info __cd;
struct __gconv_step_data __data;
};

union cudaenergythr_cu_1208
{
struct __gconv_info __cd;
struct cudaenergythr_cu_1218 __combined;
};

typedef union cudaenergythr_cu_1208 _G_iconv_t;
typedef int _G_int16_t;
typedef int _G_int32_t;
typedef unsigned int _G_uint16_t;
typedef unsigned int _G_uint32_t;
typedef __builtin_va_list __gnuc_va_list;
struct _IO_jump_t;

struct _IO_FILE;

typedef void _IO_lock_t;
struct _IO_marker
{
struct _IO_marker * _next;
struct _IO_FILE * _sbuf;
int _pos;
};

struct _IO_FILE;

enum __codecvt_result { __codecvt_ok, __codecvt_partial, __codecvt_error, __codecvt_noconv };
struct _IO_FILE
{
int _flags;
char * _IO_read_ptr;
char * _IO_read_end;
char * _IO_read_base;
char * _IO_write_base;
char * _IO_write_ptr;
char * _IO_write_end;
char * _IO_buf_base;
char * _IO_buf_end;
char * _IO_save_base;
char * _IO_backup_base;
char * _IO_save_end;
struct _IO_marker * _markers;
struct _IO_FILE * _chain;
int _fileno;
int _flags2;
__off_t _old_offset;
unsigned short _cur_column;
signed char _vtable_offset;
char _shortbuf[1];
_IO_lock_t * _lock;
__off64_t _offset;
void * __pad1;
void * __pad2;
void * __pad3;
void * __pad4;
size_t __pad5;
int _mode;
char _unused2[(((15*sizeof (int))-(4*sizeof (void * )))-sizeof (size_t))];
};

struct _IO_FILE;

typedef struct _IO_FILE _IO_FILE;
struct _IO_FILE_plus;

extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
typedef __ssize_t __io_read_fn(void * __cookie, char * __buf, size_t __nbytes);
typedef __ssize_t __io_write_fn(void * __cookie, const char * __buf, size_t __n);
typedef int __io_seek_fn(void * __cookie, __off64_t * __pos, int __w);
typedef int __io_close_fn(void * __cookie);
extern int __underflow(_IO_FILE * );
extern int __uflow(_IO_FILE * );
extern int __overflow(_IO_FILE * , int );
extern wint_t __wunderflow(_IO_FILE * );
extern wint_t __wuflow(_IO_FILE * );
extern wint_t __woverflow(_IO_FILE * , wint_t );
extern int _IO_getc(_IO_FILE * __fp);
extern int _IO_putc(int __c, _IO_FILE * __fp);
extern int _IO_feof(_IO_FILE * __fp);
extern int _IO_ferror(_IO_FILE * __fp);
extern int _IO_peekc_locked(_IO_FILE * __fp);
extern void _IO_flockfile(_IO_FILE * );
extern void _IO_funlockfile(_IO_FILE * );
extern int _IO_ftrylockfile(_IO_FILE * );
extern int _IO_vfscanf(_IO_FILE * , const char * , __gnuc_va_list , int * );
extern int _IO_vfprintf(_IO_FILE * , const char * , __gnuc_va_list );
extern __ssize_t _IO_padn(_IO_FILE * , int , __ssize_t );
extern size_t _IO_sgetn(_IO_FILE * , void * , size_t );
extern __off64_t _IO_seekoff(_IO_FILE * , __off64_t , int , int );
extern __off64_t _IO_seekpos(_IO_FILE * , __off64_t , int );
extern void _IO_free_backup_area(_IO_FILE * );
typedef _G_fpos_t fpos_t;
extern struct _IO_FILE * stdin;
extern struct _IO_FILE * stdout;
extern struct _IO_FILE * stderr;
extern int remove(const char * __filename);
extern int rename(const char * __old, const char * __new);
extern FILE * tmpfile(void );
extern char * tmpnam(char * __s);
extern char * tmpnam_r(char * __s);
extern char * tempnam(const char * __dir, const char * __pfx);
extern int fclose(FILE * __stream);
extern int fflush(FILE * __stream);
extern int fflush_unlocked(FILE * __stream);
extern FILE * fopen(const char * __filename, const char * __modes);
extern FILE * freopen(const char * __filename, const char * __modes, FILE * __stream);
extern FILE * fdopen(int __fd, const char * __modes);
extern void setbuf(FILE * __stream, char * __buf);
extern int setvbuf(FILE * __stream, char * __buf, int __modes, size_t __n);
extern void setbuffer(FILE * __stream, char * __buf, size_t __size);
extern void setlinebuf(FILE * __stream);
extern int fprintf(FILE * __stream, const char * __format,  ...);
extern int printf(const char * __format,  ...);
extern int sprintf(char * __s, const char * __format,  ...);
extern int vfprintf(FILE * __s, const char * __format, __gnuc_va_list __arg);
extern int vprintf(const char * __format, __gnuc_va_list __arg);
extern int vsprintf(char * __s, const char * __format, __gnuc_va_list __arg);
extern int snprintf(char * __s, size_t __maxlen, const char * __format,  ...);
extern int vsnprintf(char * __s, size_t __maxlen, const char * __format, __gnuc_va_list __arg);
extern int fscanf(FILE * __stream, const char * __format,  ...);
extern int scanf(const char * __format,  ...);
extern int sscanf(const char * __s, const char * __format,  ...);
extern int fgetc(FILE * __stream);
extern int getc(FILE * __stream);
extern int getchar(void );
extern int getc_unlocked(FILE * __stream);
extern int getchar_unlocked(void );
extern int fgetc_unlocked(FILE * __stream);
extern int fputc(int __c, FILE * __stream);
extern int putc(int __c, FILE * __stream);
extern int putchar(int __c);
extern int fputc_unlocked(int __c, FILE * __stream);
extern int putc_unlocked(int __c, FILE * __stream);
extern int putchar_unlocked(int __c);
extern int getw(FILE * __stream);
extern int putw(int __w, FILE * __stream);
extern char * fgets(char * __s, int __n, FILE * __stream);
extern char * gets(char * __s);
extern int fputs(const char * __s, FILE * __stream);
extern int puts(const char * __s);
extern int ungetc(int __c, FILE * __stream);
extern size_t fread(void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern size_t fwrite(const void * __ptr, size_t __size, size_t __n, FILE * __s);
extern size_t fread_unlocked(void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern size_t fwrite_unlocked(const void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern int fseek(FILE * __stream, long int __off, int __whence);
extern long int ftell(FILE * __stream);
extern void rewind(FILE * __stream);
extern int fgetpos(FILE * __stream, fpos_t * __pos);
extern int fsetpos(FILE * __stream, const fpos_t * __pos);
extern void clearerr(FILE * __stream);
extern int feof(FILE * __stream);
extern int ferror(FILE * __stream);
extern void clearerr_unlocked(FILE * __stream);
extern int feof_unlocked(FILE * __stream);
extern int ferror_unlocked(FILE * __stream);
extern void perror(const char * __s);
extern int sys_nerr;
extern const char *const sys_errlist[];
extern int fileno(FILE * __stream);
extern int fileno_unlocked(FILE * __stream);
extern FILE * popen(const char * __command, const char * __modes);
extern int pclose(FILE * __stream);
extern char * ctermid(char * __s);
extern void flockfile(FILE * __stream);
extern int ftrylockfile(FILE * __stream);
extern void funlockfile(FILE * __stream);
struct cudaenergythr_cu_3918
{
int quot;
int rem;
};

typedef struct cudaenergythr_cu_3918 div_t;
struct cudaenergythr_cu_3937
{
long int quot;
long int rem;
};

typedef struct cudaenergythr_cu_3937 ldiv_t;
extern size_t __ctype_get_mb_cur_max(void );
extern double atof(const char * __nptr);
extern int atoi(const char * __nptr);
extern long int atol(const char * __nptr);
extern long long int atoll(const char * __nptr);
extern double strtod(const char * __nptr, char *  * __endptr);
extern long int strtol(const char * __nptr, char *  * __endptr, int __base);
extern unsigned long int strtoul(const char * __nptr, char *  * __endptr, int __base);
extern long long int strtoq(const char * __nptr, char *  * __endptr, int __base);
extern unsigned long long int strtouq(const char * __nptr, char *  * __endptr, int __base);
extern long long int strtoll(const char * __nptr, char *  * __endptr, int __base);
extern unsigned long long int strtoull(const char * __nptr, char *  * __endptr, int __base);
extern double __strtod_internal(const char * __nptr, char *  * __endptr, int __group);
extern float __strtof_internal(const char * __nptr, char *  * __endptr, int __group);
extern long double __strtold_internal(const char * __nptr, char *  * __endptr, int __group);
extern long int __strtol_internal(const char * __nptr, char *  * __endptr, int __base, int __group);
extern unsigned long int __strtoul_internal(const char * __nptr, char *  * __endptr, int __base, int __group);
extern long long int __strtoll_internal(const char * __nptr, char *  * __endptr, int __base, int __group);
extern unsigned long long int __strtoull_internal(const char * __nptr, char *  * __endptr, int __base, int __group);
extern char * l64a(long int __n);
extern long int a64l(const char * __s);
typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;
typedef __loff_t loff_t;
typedef __ino_t ino_t;
typedef __dev_t dev_t;
typedef __gid_t gid_t;
typedef __mode_t mode_t;
typedef __nlink_t nlink_t;
typedef __uid_t uid_t;
typedef __off_t off_t;
typedef __pid_t pid_t;
typedef __id_t id_t;
typedef __ssize_t ssize_t;
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;
typedef __key_t key_t;
typedef __time_t time_t;
typedef __clockid_t clockid_t;
typedef __timer_t timer_t;
typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
typedef int int8_t;
typedef int int16_t;
typedef int int32_t;
typedef int int64_t;
typedef unsigned int u_int8_t;
typedef unsigned int u_int16_t;
typedef unsigned int u_int32_t;
typedef unsigned int u_int64_t;
typedef int register_t;
typedef int __sig_atomic_t;
struct cudaenergythr_cu_5363
{
unsigned long int __val[(1024/(8*sizeof (unsigned long int)))];
};

typedef struct cudaenergythr_cu_5363 __sigset_t;
typedef __sigset_t sigset_t;
struct timespec
{
__time_t tv_sec;
long int tv_nsec;
};

struct timeval
{
__time_t tv_sec;
__suseconds_t tv_usec;
};

typedef __suseconds_t suseconds_t;
typedef long int __fd_mask;
struct cudaenergythr_cu_5460
{
__fd_mask __fds_bits[(1024/(8*sizeof (__fd_mask)))];
};

typedef struct cudaenergythr_cu_5460 fd_set;
typedef __fd_mask fd_mask;
extern int select(int __nfds, fd_set * __readfds, fd_set * __writefds, fd_set * __exceptfds, struct timeval * __timeout);
extern inline unsigned int gnu_dev_major(unsigned long long int __dev);
extern inline unsigned int gnu_dev_minor(unsigned long long int __dev);
extern inline unsigned long long int gnu_dev_makedev(unsigned int __major, unsigned int __minor);
extern inline unsigned int gnu_dev_major(unsigned long long int __dev)
{
return (((__dev>>8)&4095)|(((unsigned int)(__dev>>32))&( ~ 4095)));
}

extern inline unsigned int gnu_dev_minor(unsigned long long int __dev)
{
return ((__dev&255)|(((unsigned int)(__dev>>12))&( ~ 255)));
}

extern inline unsigned long long int gnu_dev_makedev(unsigned int __major, unsigned int __minor)
{
return ((((__minor&255)|((__major&4095)<<8))|(((unsigned long long int)(__minor&( ~ 255)))<<12))|(((unsigned long long int)(__major&( ~ 4095)))<<32));
}

typedef __blkcnt_t blkcnt_t;
typedef __fsblkcnt_t fsblkcnt_t;
typedef __fsfilcnt_t fsfilcnt_t;
struct __sched_param
{
int __sched_priority;
};

struct _pthread_fastlock
{
long int __status;
int __spinlock;
};

typedef struct _pthread_descr_struct * _pthread_descr;
struct __pthread_attr_s
{
int __detachstate;
int __schedpolicy;
struct __sched_param __schedparam;
int __inheritsched;
int __scope;
size_t __guardsize;
int __stackaddr_set;
void * __stackaddr;
size_t __stacksize;
};

typedef struct __pthread_attr_s pthread_attr_t;
typedef long long __pthread_cond_align_t;
struct cudaenergythr_cu_6006
{
struct _pthread_fastlock __c_lock;
_pthread_descr __c_waiting;
char __padding[(((48-sizeof (struct _pthread_fastlock))-sizeof (_pthread_descr))-sizeof (__pthread_cond_align_t))];
__pthread_cond_align_t __align;
};

typedef struct cudaenergythr_cu_6006 pthread_cond_t;
struct cudaenergythr_cu_6063
{
int __dummy;
};

typedef struct cudaenergythr_cu_6063 pthread_condattr_t;
typedef unsigned int pthread_key_t;
struct cudaenergythr_cu_6086
{
int __m_reserved;
int __m_count;
_pthread_descr __m_owner;
int __m_kind;
struct _pthread_fastlock __m_lock;
};

typedef struct cudaenergythr_cu_6086 pthread_mutex_t;
struct cudaenergythr_cu_6122
{
int __mutexkind;
};

typedef struct cudaenergythr_cu_6122 pthread_mutexattr_t;
typedef int pthread_once_t;
typedef unsigned long int pthread_t;
extern long int random(void );
extern void srandom(unsigned int __seed);
extern char * initstate(unsigned int __seed, char * __statebuf, size_t __statelen);
extern char * setstate(char * __statebuf);
struct random_data
{
int32_t * fptr;
int32_t * rptr;
int32_t * state;
int rand_type;
int rand_deg;
int rand_sep;
int32_t * end_ptr;
};

extern int random_r(struct random_data * __buf, int32_t * __result);
extern int srandom_r(unsigned int __seed, struct random_data * __buf);
extern int initstate_r(unsigned int __seed, char * __statebuf, size_t __statelen, struct random_data * __buf);
extern int setstate_r(char * __statebuf, struct random_data * __buf);
extern int rand(void );
extern void srand(unsigned int __seed);
extern int rand_r(unsigned int * __seed);
extern double drand48(void );
extern double erand48(unsigned short int __xsubi[3]);
extern long int lrand48(void );
extern long int nrand48(unsigned short int __xsubi[3]);
extern long int mrand48(void );
extern long int jrand48(unsigned short int __xsubi[3]);
extern void srand48(long int __seedval);
extern unsigned short int * seed48(unsigned short int __seed16v[3]);
extern void lcong48(unsigned short int __param[7]);
struct drand48_data
{
unsigned short int __x[3];
unsigned short int __old_x[3];
unsigned short int __c;
unsigned short int __init;
unsigned long long int __a;
};

extern int drand48_r(struct drand48_data * __buffer, double * __result);
extern int erand48_r(unsigned short int __xsubi[3], struct drand48_data * __buffer, double * __result);
extern int lrand48_r(struct drand48_data * __buffer, long int * __result);
extern int nrand48_r(unsigned short int __xsubi[3], struct drand48_data * __buffer, long int * __result);
extern int mrand48_r(struct drand48_data * __buffer, long int * __result);
extern int jrand48_r(unsigned short int __xsubi[3], struct drand48_data * __buffer, long int * __result);
extern int srand48_r(long int __seedval, struct drand48_data * __buffer);
extern int seed48_r(unsigned short int __seed16v[3], struct drand48_data * __buffer);
extern int lcong48_r(unsigned short int __param[7], struct drand48_data * __buffer);
extern void * malloc(size_t __size);
extern void * calloc(size_t __nmemb, size_t __size);
extern void * realloc(void * __ptr, size_t __size);
extern void free(void * __ptr);
extern void cfree(void * __ptr);
extern void * alloca(size_t __size);
extern void * valloc(size_t __size);
extern void abort(void );
extern int atexit(void (* __func)(void ));
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg);
extern void exit(int __status);
extern char * getenv(const char * __name);
extern char * __secure_getenv(const char * __name);
extern int putenv(char * __string);
extern int setenv(const char * __name, const char * __value, int __replace);
extern int unsetenv(const char * __name);
extern int clearenv(void );
extern char * mktemp(char * __template);
extern int mkstemp(char * __template);
extern char * mkdtemp(char * __template);
extern int system(const char * __command);
extern char * realpath(const char * __name, char * __resolved);
typedef int (* __compar_fn_t)(const void * , const void * );
extern void * bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar);
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar);
extern int abs(int __x);
extern long int labs(long int __x);
extern div_t div(int __numer, int __denom);
extern ldiv_t ldiv(long int __numer, long int __denom);
extern char * ecvt(double __value, int __ndigit, int * __decpt, int * __sign);
extern char * fcvt(double __value, int __ndigit, int * __decpt, int * __sign);
extern char * gcvt(double __value, int __ndigit, char * __buf);
extern char * qecvt(long double __value, int __ndigit, int * __decpt, int * __sign);
extern char * qfcvt(long double __value, int __ndigit, int * __decpt, int * __sign);
extern char * qgcvt(long double __value, int __ndigit, char * __buf);
extern int ecvt_r(double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int fcvt_r(double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int qecvt_r(long double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int qfcvt_r(long double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int mblen(const char * __s, size_t __n);
extern int mbtowc(wchar_t * __pwc, const char * __s, size_t __n);
extern int wctomb(char * __s, wchar_t __wchar);
extern size_t mbstowcs(wchar_t * __pwcs, const char * __s, size_t __n);
extern size_t wcstombs(char * __s, const wchar_t * __pwcs, size_t __n);
extern int rpmatch(const char * __response);
extern int getloadavg(double __loadavg[], int __nelem);
struct sched_param
{
int __sched_priority;
};

extern int clone(int (* __fn)(void * __arg), void * __child_stack, int __flags, void * __arg);
typedef unsigned long int __cpu_mask;
struct cudaenergythr_cu_9214
{
__cpu_mask __bits[(1024/(8*sizeof (__cpu_mask)))];
};

typedef struct cudaenergythr_cu_9214 cpu_set_t;
extern int sched_setparam(__pid_t __pid, const struct sched_param * __param);
extern int sched_getparam(__pid_t __pid, struct sched_param * __param);
extern int sched_setscheduler(__pid_t __pid, int __policy, const struct sched_param * __param);
extern int sched_getscheduler(__pid_t __pid);
extern int sched_yield(void );
extern int sched_get_priority_max(int __algorithm);
extern int sched_get_priority_min(int __algorithm);
extern int sched_rr_get_interval(__pid_t __pid, struct timespec * __t);
extern long int __sysconf(int );
typedef __clock_t clock_t;
struct tm
{
int tm_sec;
int tm_min;
int tm_hour;
int tm_mday;
int tm_mon;
int tm_year;
int tm_wday;
int tm_yday;
int tm_isdst;
long int tm_gmtoff;
const char * tm_zone;
};

struct itimerspec
{
struct timespec it_interval;
struct timespec it_value;
};

struct timespec;

struct sigevent;

extern clock_t clock(void );
extern time_t time(time_t * __timer);
extern double difftime(time_t __time1, time_t __time0);
extern time_t mktime(struct tm * __tp);
extern size_t strftime(char * __s, size_t __maxsize, const char * __format, const struct tm * __tp);
extern struct tm * gmtime(const time_t * __timer);
extern struct tm * localtime(const time_t * __timer);
extern struct tm * gmtime_r(const time_t * __timer, struct tm * __tp);
extern struct tm * localtime_r(const time_t * __timer, struct tm * __tp);
extern char * asctime(const struct tm * __tp);
extern char * ctime(const time_t * __timer);
extern char * asctime_r(const struct tm * __tp, char * __buf);
extern char * ctime_r(const time_t * __timer, char * __buf);
extern char * __tzname[2];
extern int __daylight;
extern long int __timezone;
extern char * tzname[2];
extern void tzset(void );
extern int daylight;
extern long int timezone;
extern int stime(const time_t * __when);
extern time_t timegm(struct tm * __tp);
extern time_t timelocal(struct tm * __tp);
extern int dysize(int __year);
extern int nanosleep(const struct timespec * __requested_time, struct timespec * __remaining);
extern int clock_getres(clockid_t __clock_id, struct timespec * __res);
extern int clock_gettime(clockid_t __clock_id, struct timespec * __tp);
extern int clock_settime(clockid_t __clock_id, const struct timespec * __tp);
extern int timer_create(clockid_t __clock_id, struct sigevent * __evp, timer_t * __timerid);
extern int timer_delete(timer_t __timerid);
extern int timer_settime(timer_t __timerid, int __flags, const struct itimerspec * __value, struct itimerspec * __ovalue);
extern int timer_gettime(timer_t __timerid, struct itimerspec * __value);
extern int timer_getoverrun(timer_t __timerid);
enum cudaenergythr_cu_10398 { PTHREAD_CREATE_JOINABLE, PTHREAD_CREATE_DETACHED };
enum cudaenergythr_cu_10408 { PTHREAD_INHERIT_SCHED, PTHREAD_EXPLICIT_SCHED };
enum cudaenergythr_cu_10418 { PTHREAD_SCOPE_SYSTEM, PTHREAD_SCOPE_PROCESS };
enum cudaenergythr_cu_10428 { PTHREAD_MUTEX_TIMED_NP, PTHREAD_MUTEX_RECURSIVE_NP, PTHREAD_MUTEX_ERRORCHECK_NP, PTHREAD_MUTEX_ADAPTIVE_NP };
enum cudaenergythr_cu_10444 { PTHREAD_PROCESS_PRIVATE, PTHREAD_PROCESS_SHARED };
struct _pthread_cleanup_buffer
{
void (* __routine)(void * );
void * __arg;
int __canceltype;
struct _pthread_cleanup_buffer * __prev;
};

struct _pthread_cleanup_buffer;

enum cudaenergythr_cu_10493 { PTHREAD_CANCEL_ENABLE, PTHREAD_CANCEL_DISABLE };
enum cudaenergythr_cu_10503 { PTHREAD_CANCEL_DEFERRED, PTHREAD_CANCEL_ASYNCHRONOUS };
extern int pthread_create(pthread_t * __threadp, const pthread_attr_t * __attr, void * (* __start_routine)(void * ), void * __arg);
extern pthread_t pthread_self(void );
extern int pthread_equal(pthread_t __thread1, pthread_t __thread2);
extern void pthread_exit(void * __retval);
extern int pthread_join(pthread_t __th, void *  * __thread_return);
extern int pthread_detach(pthread_t __th);
extern int pthread_attr_init(pthread_attr_t * __attr);
extern int pthread_attr_destroy(pthread_attr_t * __attr);
extern int pthread_attr_setdetachstate(pthread_attr_t * __attr, int __detachstate);
extern int pthread_attr_getdetachstate(const pthread_attr_t * __attr, int * __detachstate);
extern int pthread_attr_setschedparam(pthread_attr_t * __attr, const struct sched_param * __param);
extern int pthread_attr_getschedparam(const pthread_attr_t * __attr, struct sched_param * __param);
extern int pthread_attr_setschedpolicy(pthread_attr_t * __attr, int __policy);
extern int pthread_attr_getschedpolicy(const pthread_attr_t * __attr, int * __policy);
extern int pthread_attr_setinheritsched(pthread_attr_t * __attr, int __inherit);
extern int pthread_attr_getinheritsched(const pthread_attr_t * __attr, int * __inherit);
extern int pthread_attr_setscope(pthread_attr_t * __attr, int __scope);
extern int pthread_attr_getscope(const pthread_attr_t * __attr, int * __scope);
extern int pthread_attr_setstackaddr(pthread_attr_t * __attr, void * __stackaddr);
extern int pthread_attr_getstackaddr(const pthread_attr_t * __attr, void *  * __stackaddr);
extern int pthread_attr_setstacksize(pthread_attr_t * __attr, size_t __stacksize);
extern int pthread_attr_getstacksize(const pthread_attr_t * __attr, size_t * __stacksize);
extern int pthread_setschedparam(pthread_t __target_thread, int __policy, const struct sched_param * __param);
extern int pthread_getschedparam(pthread_t __target_thread, int * __policy, struct sched_param * __param);
extern int pthread_mutex_init(pthread_mutex_t * __mutex, const pthread_mutexattr_t * __mutex_attr);
extern int pthread_mutex_destroy(pthread_mutex_t * __mutex);
extern int pthread_mutex_trylock(pthread_mutex_t * __mutex);
extern int pthread_mutex_lock(pthread_mutex_t * __mutex);
extern int pthread_mutex_unlock(pthread_mutex_t * __mutex);
extern int pthread_mutexattr_init(pthread_mutexattr_t * __attr);
extern int pthread_mutexattr_destroy(pthread_mutexattr_t * __attr);
extern int pthread_mutexattr_getpshared(const pthread_mutexattr_t * __attr, int * __pshared);
extern int pthread_mutexattr_setpshared(pthread_mutexattr_t * __attr, int __pshared);
extern int pthread_cond_init(pthread_cond_t * __cond, const pthread_condattr_t * __cond_attr);
extern int pthread_cond_destroy(pthread_cond_t * __cond);
extern int pthread_cond_signal(pthread_cond_t * __cond);
extern int pthread_cond_broadcast(pthread_cond_t * __cond);
extern int pthread_cond_wait(pthread_cond_t * __cond, pthread_mutex_t * __mutex);
extern int pthread_cond_timedwait(pthread_cond_t * __cond, pthread_mutex_t * __mutex, const struct timespec * __abstime);
extern int pthread_condattr_init(pthread_condattr_t * __attr);
extern int pthread_condattr_destroy(pthread_condattr_t * __attr);
extern int pthread_condattr_getpshared(const pthread_condattr_t * __attr, int * __pshared);
extern int pthread_condattr_setpshared(pthread_condattr_t * __attr, int __pshared);
extern int pthread_key_create(pthread_key_t * __key, void (* __destr_function)(void * ));
extern int pthread_key_delete(pthread_key_t __key);
extern int pthread_setspecific(pthread_key_t __key, const void * __pointer);
extern void * pthread_getspecific(pthread_key_t __key);
extern int pthread_once(pthread_once_t * __once_control, void (* __init_routine)(void ));
extern int pthread_setcancelstate(int __state, int * __oldstate);
extern int pthread_setcanceltype(int __type, int * __oldtype);
extern int pthread_cancel(pthread_t __cancelthread);
extern void pthread_testcancel(void );
extern void _pthread_cleanup_push(struct _pthread_cleanup_buffer * __buffer, void (* __routine)(void * ), void * __arg);
extern void _pthread_cleanup_pop(struct _pthread_cleanup_buffer * __buffer, int __execute);
extern int pthread_sigmask(int __how, const __sigset_t * __newmask, __sigset_t * __oldmask);
extern int pthread_kill(pthread_t __threadid, int __signo);
extern int pthread_atfork(void (* __prepare)(void ), void (* __parent)(void ), void (* __child)(void ));
extern void pthread_kill_other_threads_np(void );
struct char1
{
signed char x;
};

struct uchar1
{
unsigned char x;
};

struct char2
{
signed char x, y;
};

struct uchar2
{
unsigned char x, y;
};

struct char3
{
signed char x, y, z;
};

struct uchar3
{
unsigned char x, y, z;
};

struct char4
{
signed char x, y, z, w;
};

struct uchar4
{
unsigned char x, y, z, w;
};

struct short1
{
short x;
};

struct ushort1
{
unsigned short x;
};

struct short2
{
short x, y;
};

struct ushort2
{
unsigned short x, y;
};

struct short3
{
short x, y, z;
};

struct ushort3
{
unsigned short x, y, z;
};

struct short4
{
short x, y, z, w;
};

struct ushort4
{
unsigned short x, y, z, w;
};

struct int1
{
int x;
};

struct uint1
{
unsigned int x;
};

struct int2
{
int x, y;
};

struct uint2
{
unsigned int x, y;
};

struct int3
{
int x, y, z;
};

struct uint3
{
unsigned int x, y, z;
};

struct int4
{
int x, y, z, w;
};

struct uint4
{
unsigned int x, y, z, w;
};

struct long1
{
long x;
};

struct ulong1
{
unsigned long x;
};

struct long2
{
long x, y;
};

struct ulong2
{
unsigned long x, y;
};

struct long3
{
long x, y, z;
};

struct ulong3
{
unsigned long x, y, z;
};

struct long4
{
long x, y, z, w;
};

struct ulong4
{
unsigned long x, y, z, w;
};

struct float1
{
float x;
};

struct float2
{
float x, y;
};

struct float3
{
float x, y, z;
};

struct float4
{
float x, y, z, w;
};

struct double2
{
double x, y;
};

typedef struct char1 char1;
typedef struct uchar1 uchar1;
typedef struct char2 char2;
typedef struct uchar2 uchar2;
typedef struct char3 char3;
typedef struct uchar3 uchar3;
typedef struct char4 char4;
typedef struct uchar4 uchar4;
typedef struct short1 short1;
typedef struct ushort1 ushort1;
typedef struct short2 short2;
typedef struct ushort2 ushort2;
typedef struct short3 short3;
typedef struct ushort3 ushort3;
typedef struct short4 short4;
typedef struct ushort4 ushort4;
typedef struct int1 int1;
typedef struct uint1 uint1;
typedef struct int2 int2;
typedef struct uint2 uint2;
typedef struct int3 int3;
typedef struct uint3 uint3;
typedef struct int4 int4;
typedef struct uint4 uint4;
typedef struct long1 long1;
typedef struct ulong1 ulong1;
typedef struct long2 long2;
typedef struct ulong2 ulong2;
typedef struct long3 long3;
typedef struct ulong3 ulong3;
typedef struct long4 long4;
typedef struct ulong4 ulong4;
typedef struct float1 float1;
typedef struct float2 float2;
typedef struct float3 float3;
typedef struct float4 float4;
typedef struct double2 double2;
typedef struct dim3 dim3;
struct dim3
{
unsigned int x, y, z;
};

void cutCreateTimer(unsigned int * timer);
void cutStartTimer(unsigned int timer);
void cutStopTimer(unsigned int timer);
float cutGetTimerValue(unsigned int timer);
void cutDeleteTimer(unsigned int timer);
void cudaMemcpy(void * dest, void * src, size_t size, int type);
void cudaMalloc(void *  * dest, size_t size);
void cudaFree(void * ptr);
void cudaMemcpyToSymbol(void * dst, void * src, size_t size, int type);
void cudaMemset(void * ptr, int i, size_t size);
void cudaThreadSynchronize(  );
inline int __umul24(int x, int y);
struct cudaenergythr_cu_13305
{
void * data;
int width;
int height;
};

typedef struct cudaenergythr_cu_13305 cpu_texture;
enum __work_assignment { STATIC, DYNAMIC };
typedef void (* __cuda_kernel_function)(const void * , dim3 , dim3 , dim3 );
void __cuda_pthread_pool(int num_threads);
void __cuda_pthread_init(int num_threads, int assignment);
void __cuda_pthread_destruct(void );
void * __thread_pool_loop(void * index);
void __cuda_kernel_launch(__cuda_kernel_function kernel_function, dim3 grid, dim3 block, void *  * params);
void __cuda_kernel_sync(void );
enum cudaRoundMode { cudaRoundNearest, cudaRoundZero, cudaRoundPosInf, cudaRoundMinInf };
typedef long int ptrdiff_t;
enum cudaError { cudaSuccess = 0, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation, cudaErrorInitializationError, cudaErrorLaunchFailure, cudaErrorPriorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration, cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidSymbol, cudaErrorMapBufferObjectFailed, cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidHostPointer, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture, cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, cudaErrorInvalidMemcpyDirection, cudaErrorAddressOfConstant, cudaErrorTextureFetchFailed, cudaErrorTextureNotBound, cudaErrorSynchronizationError, cudaErrorInvalidFilterSetting, cudaErrorInvalidNormSetting, cudaErrorMixedDeviceExecution, cudaErrorCudartUnloading, cudaErrorUnknown, cudaErrorNotYetImplemented, cudaErrorMemoryValueTooLarge, cudaErrorInvalidResourceHandle, cudaErrorNotReady, cudaErrorStartupFailure = 127, cudaErrorApiFailureBase = 10000 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
struct cudaDeviceProp
{
char name[256];
size_t totalGlobalMem;
size_t sharedMemPerBlock;
int regsPerBlock;
int warpSize;
size_t memPitch;
int maxThreadsPerBlock;
int maxThreadsDim[3];
int maxGridSize[3];
size_t totalConstMem;
int major;
int minor;
int clockRate;
size_t textureAlignment;
};

typedef enum cudaError cudaError_t;
typedef int cudaStream_t;
typedef int cudaEvent_t;
struct cudaArray;

enum cudaChannelFormatKind { cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, cudaChannelFormatKindFloat };
struct cudaChannelFormatDesc
{
int x;
int y;
int z;
int w;
enum cudaChannelFormatKind f;
};

enum cudaTextureAddressMode { cudaAddressModeWrap, cudaAddressModeClamp };
enum cudaTextureFilterMode { cudaFilterModePoint, cudaFilterModeLinear };
enum cudaTextureReadMode { cudaReadModeElementType, cudaReadModeNormalizedFloat };
struct textureReference
{
int normalized;
enum cudaTextureFilterMode filterMode;
enum cudaTextureAddressMode addressMode[2];
struct cudaChannelFormatDesc channelDesc;
};

struct cudaChannelFormatDesc;

typedef void * rt_timerhandle;
rt_timerhandle rt_timer_create(void );
void rt_timer_destroy(rt_timerhandle );
void rt_timer_start(rt_timerhandle );
void rt_timer_stop(rt_timerhandle );
float rt_timer_time(rt_timerhandle );
float rt_timer_timenow(rt_timerhandle );
unsigned int rt_rand(unsigned int * );
typedef int rt_thread_t;
typedef int rt_mutex_t;
typedef int rt_barrier_t;
typedef int rt_rwlock_t;
int rt_thread_numprocessors(void );
int rt_thread_setconcurrency(int );
int rt_thread_create(rt_thread_t * , void * routine(void * ), void * );
int rt_thread_join(rt_thread_t , void *  * );
int rt_mutex_init(rt_mutex_t * );
int rt_mutex_lock(rt_mutex_t * );
int rt_mutex_unlock(rt_mutex_t * );
int rt_rwlock_init(rt_rwlock_t * );
int rt_rwlock_readlock(rt_rwlock_t * );
int rt_rwlock_writelock(rt_rwlock_t * );
int rt_rwlock_unlock(rt_rwlock_t * );
rt_barrier_t * rt_thread_barrier_init(int n_clients);
void rt_thread_barrier_destroy(rt_barrier_t * barrier);
int rt_thread_barrier(rt_barrier_t * barrier, int increment);
int calc_grid_energies_cuda_thr(float * atoms, float * grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char * excludepos, int maxnumprocs);
struct cudaenergythr_cu_14200
{
int threadid;
int threadcount;
float * atoms;
float * grideners;
long int numplane;
long int numcol;
long int numpt;
long int natoms;
float gridspacing;
unsigned char * excludepos;
};

typedef struct cudaenergythr_cu_14200 enthrparms;
static void * cudaenergythread(void * );
float4 atominfo[4000];
struct cenergy__MCUDA_kernel_params
{
int numatoms;
float gridspacing;
float * energygrid;
};

typedef struct cenergy__MCUDA_kernel_params cenergy__MCUDA_kernel_params;
void cenergy__MCUDA_kernel(const void *  __params, dim3 blockIdx, dim3 blockDim, dim3 gridDim)
{
dim3 threadIdx;
int __threadIndex;
for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
int numatoms = ((cenergy__MCUDA_kernel_params * )__params)->numatoms;
float gridspacing = ((cenergy__MCUDA_kernel_params * )__params)->gridspacing;
float * energygrid = ((cenergy__MCUDA_kernel_params * )__params)->energygrid;
unsigned int xindex;
unsigned int yindex;
unsigned int outaddr;
float curenergyx1;
float curenergyx2;
float curenergyx3;
float curenergyx4;
float coory;
float coorx1;
float coorx2;
float coorx3;
float coorx4;
float energyvalx1;
float energyvalx2;
float energyvalx3;
float energyvalx4;
int atomid;
xindex=((__umul24(blockIdx.x, blockDim.x)+threadIdx.x)*4);
yindex=(__umul24(blockIdx.y, blockDim.y)+threadIdx.y);
outaddr=(((__umul24(gridDim.x, blockDim.x)*4)*yindex)+xindex);
curenergyx1=energygrid[outaddr];
curenergyx2=energygrid[(outaddr+1)];
curenergyx3=energygrid[(outaddr+2)];
curenergyx4=energygrid[(outaddr+3)];
coory=(gridspacing*yindex);
coorx1=(gridspacing*xindex);
coorx2=(gridspacing*(xindex+1));
coorx3=(gridspacing*(xindex+2));
coorx4=(gridspacing*(xindex+3));
energyvalx1=0.0f;
energyvalx2=0.0f;
energyvalx3=0.0f;
energyvalx4=0.0f;
for (atomid=0; atomid<numatoms; atomid ++ )
{
float dy;
float dysqpdzsq;
float dx1;
float dx2;
float dx3;
float dx4;
dy=(coory-atominfo[atomid].y);
dysqpdzsq=((dy*dy)+atominfo[atomid].z);
dx1=(coorx1-atominfo[atomid].x);
dx2=(coorx2-atominfo[atomid].x);
dx3=(coorx3-atominfo[atomid].x);
dx4=(coorx4-atominfo[atomid].x);
energyvalx1+=(atominfo[atomid].w*(1.0f/sqrtf(((dx1*dx1)+dysqpdzsq))));
energyvalx2+=(atominfo[atomid].w*(1.0f/sqrtf(((dx2*dx2)+dysqpdzsq))));
energyvalx3+=(atominfo[atomid].w*(1.0f/sqrtf(((dx3*dx3)+dysqpdzsq))));
energyvalx4+=(atominfo[atomid].w*(1.0f/sqrtf(((dx4*dx4)+dysqpdzsq))));
}

energygrid[outaddr]=(curenergyx1+energyvalx1);
energygrid[(outaddr+1)]=(curenergyx2+energyvalx2);
energygrid[(outaddr+2)]=(curenergyx3+energyvalx3);
energygrid[(outaddr+3)]=(curenergyx4+energyvalx4);
/*
__MCUDA_THREAD_BODY
*/
}

}

/*
__MCUDA_OUTER_LOOP
*/
}

}

static void cenergy(int numatoms, float gridspacing, float * energygrid, dim3 gridDim, dim3 blockDim)
{
cenergy__MCUDA_kernel_params *  __params;
__params=((cenergy__MCUDA_kernel_params * )malloc(sizeof (struct cenergy__MCUDA_kernel_params)));
__params->numatoms=numatoms;
__params->gridspacing=gridspacing;
__params->energygrid=energygrid;
__cuda_kernel_launch(cenergy__MCUDA_kernel, gridDim, blockDim, ((void *  * )( & __params)));
}

int copyatomstoconstbuf(const float * atoms, int count, float zplane)
{
float atompre[(4*4000)];
int i;
if ((count>4000))
{
printf("Atom count exceeds constant buffer storage capacity\n");
return ( - 1);
}

for (i=0; i<(count*4); i+=4)
{
float dz;
atompre[i]=atoms[i];
atompre[(i+1)]=atoms[(i+1)];
dz=(zplane-atoms[(i+2)]);
atompre[(i+2)]=(dz*dz);
atompre[(i+3)]=atoms[(i+3)];
}

cudaMemcpyToSymbol(atominfo, atompre, ((count*4)*sizeof (float)), 0);
{
cudaError_t err;
if (((err=cudaGetLastError())!=cudaSuccess))
{
printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), 270);
printf("Thread aborting...\n");
return ((void * )0);
}

}

return 0;
}

int calc_grid_energies_cuda_thr(float * atoms, float * grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char * excludepos, int maxnumprocs)
{
int i;
enthrparms * parms;
rt_thread_t * threads;
rt_timerhandle globaltimer;
double totalruntime;
int numprocs;
double atomevalssec;
numprocs=1;
printf("calc_grid_energies_cuda_thr()\n");
printf("  using %d processors\n", numprocs);
threads=((rt_thread_t * )calloc((numprocs*sizeof (rt_thread_t)), 1));
parms=((enthrparms * )malloc((numprocs*sizeof (enthrparms))));
for (i=0; i<numprocs; i ++ )
{
parms[i].threadid=i;
parms[i].threadcount=numprocs;
parms[i].atoms=atoms;
parms[i].grideners=grideners;
parms[i].numplane=numplane;
parms[i].numcol=numcol;
parms[i].numpt=numpt;
parms[i].natoms=natoms;
parms[i].gridspacing=gridspacing;
parms[i].excludepos=excludepos;
}

if ((natoms>1))
{
printf("GPU padded grid size: %d x %d x %d\n", numpt, numcol, numplane);
}

globaltimer=rt_timer_create();
rt_timer_start(globaltimer);
cudaenergythread(((void * )( & parms[0])));
rt_timer_stop(globaltimer);
totalruntime=rt_timer_time(globaltimer);
rt_timer_destroy(globaltimer);
atomevalssec=((((((double)numplane)*numcol)*numpt)*natoms)/(totalruntime*1.0E9));
printf("  %g billion atom evals/second, %g GFLOPS\n", atomevalssec, (atomevalssec*(31.0/4.0)));
free(parms);
free(threads);
return 0;
}

static void * cudaenergythread(void * voidparms)
{
dim3 volsize, Gsz, Bsz;
float * devenergy;
float * hostenergy;
enthrparms * parms;
const float * atoms;
float * grideners;
const long int numplane;
const long int numcol;
const long int numpt;
const long int natoms;
const float gridspacing;
const unsigned char * excludepos;
const int threadid;
const int threadcount;
rt_timerhandle runtimer, copytimer, timer;
float copytotal, runtotal;
int volmemsz;
float lasttime, totaltime;
int iterations;
int k;
devenergy=((void * )0);
hostenergy=((void * )0);
parms=((enthrparms * )voidparms);
atoms=parms->atoms;
grideners=parms->grideners;
numplane=parms->numplane;
numcol=parms->numcol;
numpt=parms->numpt;
natoms=parms->natoms;
gridspacing=parms->gridspacing;
excludepos=parms->excludepos;
threadid=parms->threadid;
threadcount=parms->threadcount;
if ((natoms>1))
{
printf("Thread %d opening CUDA device %d...\n", threadid, threadid);
}

cudaSetDevice(threadid);
{
cudaError_t err;
if (((err=cudaGetLastError())!=cudaSuccess))
{
printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), 411);
printf("Thread aborting...\n");
return ((void * )0);
}

}

volsize.x=((numpt+(16-1))&( ~ (16-1)));
volsize.y=((numcol+(16-1))&( ~ (16-1)));
volsize.z=1;
Bsz.x=(16/4);
Bsz.y=16;
Bsz.z=1;
Gsz.x=(volsize.x/(Bsz.x*4));
Gsz.y=(volsize.y/Bsz.y);
Gsz.z=1;
volmemsz=(((sizeof (float)*volsize.x)*volsize.y)*volsize.z);
copytimer=rt_timer_create();
runtimer=rt_timer_create();
timer=rt_timer_create();
rt_timer_start(timer);
if ((natoms>1))
{
printf("thread %d started...\n", threadid);
}

cudaMalloc(((void *  * )( & devenergy)), volmemsz);
{
cudaError_t err;
if (((err=cudaGetLastError())!=cudaSuccess))
{
printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), 438);
printf("Thread aborting...\n");
return ((void * )0);
}

}

hostenergy=((float * )malloc(volmemsz));
iterations=0;
for (k=threadid; k<numplane; k+=threadcount)
{
int x, y;
int atomstart;
float zplane;
zplane=(k*((float)gridspacing));
for (y=0; y<numcol; y ++ )
{
long eneraddr;
eneraddr=(((k*numcol)*numpt)+(y*numpt));
for (x=0; x<numpt; x ++ )
{
long addr;
addr=(eneraddr+x);
hostenergy[((y*volsize.x)+x)]=grideners[addr];
}

}

cudaMemcpy(devenergy, hostenergy, volmemsz, cudaMemcpyHostToDevice);
{
cudaError_t err;
if (((err=cudaGetLastError())!=cudaSuccess))
{
printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), 461);
printf("Thread aborting...\n");
return ((void * )0);
}

}

lasttime=rt_timer_timenow(timer);
for (atomstart=0; atomstart<natoms; atomstart+=4000)
{
int runatoms;
int atomsremaining;
iterations ++ ;
atomsremaining=(natoms-atomstart);
if ((atomsremaining>4000))
{
runatoms=4000;
}
else
{
runatoms=atomsremaining;
}

rt_timer_start(copytimer);
if (copyatomstoconstbuf((atoms+(4*atomstart)), runatoms, zplane))
{
return ((void * )0);
}

rt_timer_stop(copytimer);
copytotal+=rt_timer_time(copytimer);
rt_timer_start(runtimer);
cenergy(runatoms, gridspacing, devenergy, Gsz, Bsz, 0);
{
cudaError_t err;
if (((err=cudaGetLastError())!=cudaSuccess))
{
printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), 483);
printf("Thread aborting...\n");
return ((void * )0);
}

}

rt_timer_stop(runtimer);
runtotal+=rt_timer_time(runtimer);
}

cudaMemcpy(hostenergy, devenergy, volmemsz, cudaMemcpyDeviceToHost);
{
cudaError_t err;
if (((err=cudaGetLastError())!=cudaSuccess))
{
printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), 490);
printf("Thread aborting...\n");
return ((void * )0);
}

}

for (y=0; y<numcol; y ++ )
{
long eneraddr;
eneraddr=(((k*numcol)*numpt)+(y*numpt));
for (x=0; x<numpt; x ++ )
{
long addr;
addr=(eneraddr+x);
grideners[addr]=hostenergy[((y*volsize.x)+x)];
if ((excludepos[addr]!=0))
{
grideners[addr]=0;
}

}

}

totaltime=rt_timer_timenow(timer);
if ((natoms>1))
{
printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n", threadid, k, numplane, (totaltime-lasttime), totaltime, ((totaltime*numplane)/(k+1)));
}

}

free(hostenergy);
cudaFree(devenergy);
rt_timer_destroy(timer);
rt_timer_destroy(runtimer);
rt_timer_destroy(copytimer);
return ((void * )0);
}

