/**
 @file
 fir~ - the *~ signal operator
 updated 3/22/09 ajm: new API

 @ingroup	examples
 */

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"
#include <array>

// #define WIN_SSE_INTRINSICS

#ifdef WIN_VERSION
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

//Thanks to https://sestevenson.wordpress.com/implementation-of-fir-filtering-in-c-part-1/

//My Defs
// maximum number of inputs that can be handled
// in one function call
#define MAX_INPUT_LEN 512
// maximum length of filter than can be handled
#define MAX_COEFFS_LEN 512
// buffer to hold all of the input samples
#define BUFFER_LEN (MAX_COEFFS_LEN - 1 + MAX_INPUT_LEN)

static t_class *fir_class;

typedef struct _fir
{
    
    t_pxobject	x_obj;
    long m_in;  // proxie id
    void *m_proxy;
    double x_coeffs[ MAX_COEFFS_LEN ];
    long x_coeffs_len;
    double x_coeffs_accum[ BUFFER_LEN ];

} t_fir;


void fir_assist(t_fir *x, void *b, long m, long a, char *s);
void *fir_new(double val);
void fir_dsp64(t_fir *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void fir_list(t_fir *x, t_symbol *msg, long argc, t_atom *argv);
void fir_float(t_fir *x, double f);
void fir_perform64_method(t_fir *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);
void fir_firFloat(t_fir *x, double *coeffs, double *input, double *output, int length, int filterLength );


void ext_main(void *r)
{
	t_class *c;

	c = class_new("fir~", (method)fir_new, (method)dsp_free, sizeof(t_fir), 0L, A_DEFFLOAT, 0);
	class_dspinit(c);

	class_addmethod(c, (method)fir_dsp64, "dsp64", A_CANT, 0);
    class_addmethod(c, (method)fir_float, "float", A_FLOAT, 0);
    class_addmethod(c, (method)fir_list, "list", A_GIMME,0);
	class_addmethod(c, (method)fir_assist, "assist", A_CANT, 0);
	class_setname("*~","fir~"); // because the filename on disk is different from the object name in Max

	class_register(CLASS_BOX, c);
	fir_class = c;
}

void fir_float(t_fir *x, double f)
{
    if(proxy_getinlet((t_object *)x)==1)
    {
        x->x_coeffs[0] = f;
        x->x_coeffs_len = 1;
    }
}

void fir_list(t_fir *x, t_symbol *msg, long argc, t_atom *argv)
{
    if(proxy_getinlet((t_object *)x)==1)
    {
        if(argc > MAX_COEFFS_LEN)
        {
            post("fir~ (%p): Too many coefficants, filter must be shorter than %ld", x, MAX_COEFFS_LEN);
            return;
        }
        
        x->x_coeffs_len = argc;
        for (int i=0; i<argc; i++)
        {
            double value = atom_getfloat(argv+i);
            x->x_coeffs[i] = value;
        }
    }
}

void fir_dsp64(t_fir *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
    dsp_add64(dsp64, (t_object *) x, (t_perfroutine64) fir_perform64_method, 0, 0);
}

void fir_perform64_method(t_fir *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
    t_double *in1 = ins[0];
    t_double *out = outs[0];
    
    if (IS_DENORM_DOUBLE(*in1))
    {
        static int counter = 0;
        post("fir~ (%p): saw denorm (%d)", x, counter++);
    }
    
    fir_firFloat(x, x->x_coeffs, in1, out, sampleframes, x->x_coeffs_len);
}

// the FIR filter function
void fir_firFloat(t_fir *x, double *coeffs, double *input, double *output,
              int length, int filterLength )
{
    
    double acc;     // accumulator for MACs
    double *coeffp; // pointer to coefficients
    double *inputp; // pointer to input samples
    int n;
    int k;
    
    // put the new samples at the high end of the buffer
    memcpy( &x->x_coeffs_accum[filterLength - 1], input,
           length * sizeof(double) );
    
    // apply the filter to each input sample
    for ( n = 0; n < length; n++ ) {
        // calculate output n
        coeffp = coeffs;
        inputp = &x->x_coeffs_accum[filterLength - 1 + n];
        acc = 0;
        for ( k = 0; k < filterLength; k++ ) {
            acc += (*coeffp++) * (*inputp--);
        }
        output[n] = acc;
    }
    // shift input samples back in time for next time
    memmove( &x->x_coeffs_accum[0], &x->x_coeffs_accum[length],
            (filterLength - 1) * sizeof(double) );
}


void fir_assist(t_fir *x, void *b, long m, long a, char *s)
{
	if (m == ASSIST_OUTLET)
    {
		sprintf(s,"(Signal) Convolved Signal");
    }
	else
    {
		switch (a)
        {
		case 0:
			sprintf(s,"(Signal) Signal to be filtered.");
			break;
		case 1:
			sprintf(s,"(List) List of filter coefficients.");
			break;
		}
	}
}

void *fir_new(double val)
{
	t_fir *x = (t_fir *) object_alloc((t_class *) fir_class);
	dsp_setup((t_pxobject *)x,1);
    
    x->m_proxy = proxy_new((t_pxobject *)x, 1, &x->m_in);

    outlet_new((t_pxobject *)x, "signal");
    
    memset( x->x_coeffs_accum, 0, sizeof( x->x_coeffs_accum ) );
    
    x->x_coeffs_len=MAX_COEFFS_LEN;
    
    // bandpass filter centred around 1000 Hz
    // sampling rate = 8000 Hz
    double bandpass_coeffs[ MAX_COEFFS_LEN ] =
    {
        -0.0448093,  0.0322875,   0.0181163,   0.0087615,   0.0056797,
        0.0086685,  0.0148049,   0.0187190,   0.0151019,   0.0027594,
        -0.0132676, -0.0232561,  -0.0187804,   0.0006382,   0.0250536,
        0.0387214,  0.0299817,   0.0002609,  -0.0345546,  -0.0525282,
        -0.0395620,  0.0000246,   0.0440998,   0.0651867,   0.0479110,
        0.0000135, -0.0508558,  -0.0736313,  -0.0529380,  -0.0000709,
        0.0540186,  0.0766746,   0.0540186,  -0.0000709,  -0.0529380,
        -0.0736313, -0.0508558,   0.0000135,   0.0479110,   0.0651867,
        0.0440998,  0.0000246,  -0.0395620,  -0.0525282,  -0.0345546,
        0.0002609,  0.0299817,   0.0387214,   0.0250536,   0.0006382,
        -0.0187804, -0.0232561,  -0.0132676,   0.0027594,   0.0151019,
        0.0187190,  0.0148049,   0.0086685,   0.0056797,   0.0087615,
        0.0181163,  0.0322875,  -0.0448093
    };
    
    memcpy(x->x_coeffs, bandpass_coeffs, MAX_COEFFS_LEN*sizeof(double));

	return (x);
}

