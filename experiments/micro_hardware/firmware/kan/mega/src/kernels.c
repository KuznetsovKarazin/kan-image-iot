/* Frozen KAN arithmetic, with explicit flash readers for the AVR port. */
#include "kernel.h"
#include <math.h>
#include <stddef.h>
#ifdef __AVR__
#include <avr/pgmspace.h>
#ifdef MEGA_TEST_FORCE_FAR
#define FLASH_STORAGE __attribute__((section(".mega_far")))
#else
#define FLASH_STORAGE PROGMEM
#endif
#ifdef MEGA_TEST_BROKEN_NEAR
#define FLASH_FLOAT(a,i) pgm_read_float_near((uint16_t)(pgm_get_far_address(a)+(uint32_t)(i)*4UL))
#define FLASH_INT8(a,i) ((int8_t)pgm_read_byte_near((uint16_t)(pgm_get_far_address(a)+(uint32_t)(i))))
#else
#define FLASH_FLOAT(a,i) pgm_read_float_far(pgm_get_far_address(a)+(uint32_t)(i)*4UL)
#define FLASH_INT8(a,i) ((int8_t)pgm_read_byte_far(pgm_get_far_address(a)+(uint32_t)(i)))
#endif
#else
#define FLASH_STORAGE
#define FLASH_FLOAT(a,i) ((a)[i])
#define FLASH_INT8(a,i) ((a)[i])
#endif

typedef float (*FloatReader)(uint32_t);
typedef int8_t (*Int8Reader)(uint32_t);
typedef struct {
    uint16_t ni,no,knots,degree;
    FloatReader grid,coef,base,spline,mask,ss,sb,ns,nb;
} DirectLayer;
typedef struct {
    uint16_t ni,no,points;
    FloatReader fp;
    Int8Reader q;
    FloatReader qs,lo,hi,ss,sb,ns,nb;
} TableLayer;

#include "model.inc"
#if SELECTED_VARIANT == 0 && defined(OPT_CUBIC)
#include "cubic.inc"
#endif
_Static_assert(sizeof(float)==4,"binary32 required");

#if SELECTED_VARIANT == 0
static void direct_layer(const DirectLayer *l,const float *x,float *y) {
    const uint16_t basis_count=l->knots-l->degree-1;
    for (uint16_t j=0;j<l->no;++j) y[j]=0.0f;
    for (uint16_t i=0;i<l->ni;++i) {
#ifdef OPT_CUBIC
        float basis[12]={0.0f};
        const uint32_t gi=(uint32_t)i*l->knots;
        const float v=x[i];
        int16_t span=-1;
        for (uint16_t t=0;t<l->knots-1;++t)
            if (v>=l->grid(gi+t) && v<l->grid(gi+t+1)) { span=t;break; }
        if (span>=0) {
            const float u=(v-l->grid(gi+(uint16_t)span))*FLASH_FLOAT(cubic_inv_h,(uint16_t)span);
            for (uint16_t q=0;q<4;++q) {
                const int16_t t=span-3+(int16_t)q;
                if (t<0 || t>=8) continue;
                const uint16_t j=((uint16_t)span*4+q)*4;
                basis[t]=((FLASH_FLOAT(cubic_basis_coeff,j+3)*u
                    +FLASH_FLOAT(cubic_basis_coeff,j+2))*u
                    +FLASH_FLOAT(cubic_basis_coeff,j+1))*u
                    +FLASH_FLOAT(cubic_basis_coeff,j);
            }
        }
#else
        float basis[12],next[12];
        const uint32_t gi=(uint32_t)i*l->knots;
        const float v=x[i];
        for (uint16_t t=0;t<l->knots-1;++t)
            basis[t]=(v>=l->grid(gi+t) && v<l->grid(gi+t+1)) ? 1.0f : 0.0f;
        for (uint16_t k=1;k<=l->degree;++k) {
            for (uint16_t t=0;t<l->knots-k-1;++t)
                next[t]=((v-l->grid(gi+t))/(l->grid(gi+t+k)-l->grid(gi+t)))*basis[t]
                       +((l->grid(gi+t+k+1)-v)/(l->grid(gi+t+k+1)-l->grid(gi+t+1)))*basis[t+1];
            for (uint16_t t=0;t<l->knots-k-1;++t) basis[t]=next[t];
        }
#endif
        const float silu=v/(1.0f+expf(-v));
        for (uint16_t j=0;j<l->no;++j) {
            const uint32_t e=(uint32_t)i*l->no+j;
            float spline=0.0f;
            #ifdef OPT_CUBIC
            const uint16_t first=(span>3) ? (uint16_t)(span-3) : 0;
            const uint16_t end=(span<0) ? 0 : ((span>=7) ? 8 : (uint16_t)(span+1));
            for (uint16_t t=first;t<end;++t)
#else
            for (uint16_t t=0;t<basis_count;++t)
#endif
                spline+=basis[t]*l->coef(e*basis_count+t);
            y[j]+=l->mask(e)*(l->base(e)*silu+l->spline(e)*spline);
        }
    }
    for (uint16_t j=0;j<l->no;++j) y[j]=(y[j]*l->ss(j)+l->sb(j))*l->ns(j)+l->nb(j);
}
#endif

#if SELECTED_VARIANT == 4 || SELECTED_VARIANT == 5
static void table_layer(const TableLayer *l,const float *x,float *y) {
    for (uint16_t j=0;j<l->no;++j) y[j]=0.0f;
    for (uint16_t i=0;i<l->ni;++i) {
        const float v=fminf(l->hi(i),fmaxf(l->lo(i),x[i]));
        const float p=((v-l->lo(i))/(l->hi(i)-l->lo(i)))*(l->points-1);
        int16_t k=(int16_t)floorf(p);
        if (k<0) k=0;
        if (k>(int16_t)l->points-2) k=(int16_t)l->points-2;
        const float f=p-k;
        for (uint16_t j=0;j<l->no;++j) {
            const uint32_t index=((uint32_t)i*l->points+(uint16_t)k)*l->no+j;
            float a,b;
            if (l->q) {
                const float scale=l->qs((uint32_t)i*l->no+j);
#ifdef OPT_Q_FACTOR
                const int16_t qa=l->q(index),qb=l->q(index+l->no);
                y[j]+=scale*((float)qa+f*(float)(qb-qa));
                continue;
#else
                a=(float)l->q(index)*scale;
                b=(float)l->q(index+l->no)*scale;
#endif
            } else {
                a=l->fp(index);b=l->fp(index+l->no);
            }
            y[j]+=(1.0f-f)*a+f*b;
        }
    }
    for (uint16_t j=0;j<l->no;++j) y[j]=(y[j]*l->ss(j)+l->sb(j))*l->ns(j)+l->nb(j);
}
#endif

#if SELECTED_VARIANT == 2 || SELECTED_VARIANT == 3
static void whole(const float *x,float *y) {
    y[0]=y[1]=0.0f;
    for (uint16_t i=0;i<MEGA_INPUTS;++i) {
        const float p=(x[i]-WHOLE_LOW)/WHOLE_STEP;
        if (p<=-1.0f || p>=(float)WHOLE_POINTS) continue;
        const int16_t k=(int16_t)floorf(p);
        const float f=p-k;
        for (uint16_t j=0;j<2;++j) {
            float value=0.0f;
            if (k>=0 && k<WHOLE_POINTS) value+=(1.0f-f)*WHOLE_READ(((uint32_t)k*MEGA_INPUTS+i)*2+j);
            if (k+1>=0 && k+1<WHOLE_POINTS) value+=f*WHOLE_READ(((uint32_t)(k+1)*MEGA_INPUTS+i)*2+j);
            y[j]+=value;
        }
    }
    for (uint16_t j=0;j<2;++j) {
        y[j]/=MEGA_INPUTS;
#if SELECTED_VARIANT == 3
        y[j]=MEGA_INPUTS*y[j]-(MEGA_INPUTS-1)*read_f_zero(j);
#endif
    }
}
#endif

int mega_infer(const float *input,float *output) {
    if (!input || !output) return -1;
    for (uint16_t i=0;i<MEGA_INPUTS;++i) if (!isfinite(input[i])) return -2;
#if SELECTED_VARIANT == 2 || SELECTED_VARIANT == 3
    whole(input,output);
#else
    float a[MEGA_INPUTS],b[MEGA_INPUTS];
    for (uint16_t i=0;i<MEGA_INPUTS;++i) a[i]=input[i];
    for (uint16_t l=0;l<MODEL_DEPTH;++l) {
#if SELECTED_VARIANT == 0
        direct_layer(&direct_layers[l],a,b);
        const uint16_t no=direct_layers[l].no;
#else
        table_layer(&table_layers[l],a,b);
        const uint16_t no=table_layers[l].no;
#endif
        for (uint16_t j=0;j<no;++j) a[j]=b[j];
    }
    output[0]=a[0];output[1]=a[1];
#endif
    return isfinite(output[0]) && isfinite(output[1]) ? 0 : -3;
}


int infer_batch(const float *inputs,float *outputs,uint32_t n) {
    for (uint32_t i=0;i<n;++i) {int r=mega_infer(inputs+i*MEGA_INPUTS,outputs+i*2);if(r)return r;}
    return 0;
}
