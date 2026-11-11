/* Same-language KAN/LUT comparisons. No heap allocation; no Python runtime. */
#include "micro_model.h"
#include <math.h>
#include <stddef.h>
#include <string.h>
#ifdef ESP_PLATFORM
#include <esp_heap_caps.h>
#else
#include <stdlib.h>
#endif

typedef struct {
    unsigned ni, no, knots, degree;
    const float *grid, *coef, *base, *spline, *mask;
    const float *ss, *sb, *ns, *nb;
} DirectLayer;

typedef struct {
    unsigned ni, no, points;
    const float *fp;
    const int8_t *q;
    const float *qs, *lo, *hi, *ss, *sb, *ns, *nb;
} TableLayer;

#ifndef MODEL_CASE
#define MODEL_CASE 0
#endif
#if MODEL_CASE == 0
#include "model0.inc"
#elif MODEL_CASE == 1
#include "model1.inc"
#elif MODEL_CASE == 2
#include "model2.inc"
#else
#error Unsupported MODEL_CASE
#endif
#include "cubic.inc"

static void direct_layer(const DirectLayer *l, const float *x, float *y) {
    const unsigned basis_count = l->knots - l->degree - 1;
    for (unsigned j=0;j<l->no;++j) y[j]=0.0f;
    for (unsigned i=0;i<l->ni;++i) {
        float basis[12], next[12];
        const float *g=l->grid+i*l->knots;
        const float v=x[i];
        for (unsigned t=0;t<l->knots-1;++t)
            basis[t]=(v>=g[t] && v<g[t+1]) ? 1.0f : 0.0f;
        /* Export rejects repeated/non-increasing knots. */
        for (unsigned k=1;k<=l->degree;++k) {
            for (unsigned t=0;t<l->knots-k-1;++t)
                next[t]=((v-g[t])/(g[t+k]-g[t]))*basis[t]
                       +((g[t+k+1]-v)/(g[t+k+1]-g[t+1]))*basis[t+1];
            for (unsigned t=0;t<l->knots-k-1;++t) basis[t]=next[t];
        }
        const float silu=v/(1.0f+expf(-v));
        for (unsigned j=0;j<l->no;++j) {
            const unsigned e=i*l->no+j;
            float spline=0.0f;
            for (unsigned t=0;t<basis_count;++t)
                spline+=basis[t]*l->coef[e*basis_count+t];
            y[j]+=l->mask[e]*(l->base[e]*silu+l->spline[e]*spline);
        }
    }
    for (unsigned j=0;j<l->no;++j) y[j]=(y[j]*l->ss[j]+l->sb[j])*l->ns[j]+l->nb[j];
}

static void cubic_layer(const DirectLayer *l, const float *x, float *y) {
    const unsigned basis_count = l->knots - l->degree - 1;
    for (unsigned j=0;j<l->no;++j) y[j]=0.0f;
    for (unsigned i=0;i<l->ni;++i) {
        float basis[12]={0.0f};
        const float *g=l->grid+i*l->knots;
        const float v=x[i];
        int span=-1;
        for (unsigned t=0;t<l->knots-1;++t)
            if (v>=g[t] && v<g[t+1]) {span=(int)t;break;}
        if (span>=0) {
            const float u=(v-g[span])*cubic_inv_h[span];
            for (int q=0;q<4;++q) {
                const int t=span-3+q;
                if (t<0 || t>=8) continue;
                const unsigned j=(span*4+q)*4;
                basis[t]=((cubic_basis_coeff[j+3]*u+cubic_basis_coeff[j+2])*u
                    +cubic_basis_coeff[j+1])*u+cubic_basis_coeff[j];
            }
        }
        const float silu=v/(1.0f+expf(-v));
        for (unsigned j=0;j<l->no;++j) {
            const unsigned e=i*l->no+j;
            float spline=0.0f;
            const unsigned first=(span>3) ? (unsigned)(span-3) : 0;
            const unsigned end=(span<0) ? 0 : ((span>=7) ? 8 : (unsigned)(span+1));
            for (unsigned t=first;t<end;++t)
                spline+=basis[t]*l->coef[e*basis_count+t];
            y[j]+=l->mask[e]*(l->base[e]*silu+l->spline[e]*spline);
        }
    }
    for (unsigned j=0;j<l->no;++j) y[j]=(y[j]*l->ss[j]+l->sb[j])*l->ns[j]+l->nb[j];
}

static __attribute__((noinline)) void table_layer(const TableLayer *l, const float *x, float *y, int factored) {
    for (unsigned j=0;j<l->no;++j) y[j]=0.0f;
    for (unsigned i=0;i<l->ni;++i) {
        const float v=fminf(l->hi[i],fmaxf(l->lo[i],x[i]));
        const float p=((v-l->lo[i])/(l->hi[i]-l->lo[i]))*(l->points-1);
        int k=(int)floorf(p);
        if (k<0) k=0;
        if (k>(int)l->points-2) k=(int)l->points-2;
        const float f=p-k;
        for (unsigned j=0;j<l->no;++j) {
            const unsigned index=(i*l->points+(unsigned)k)*l->no+j;
            float a,b;
            if (l->q) {
                const float scale=l->qs[i*l->no+j];
                if (factored) {
                    const int16_t qa=l->q[index], qb=l->q[index+l->no];
                    y[j]+=scale*((float)qa+f*(float)(qb-qa));
                    continue;
                }
                a=(float)l->q[index]*scale;
                b=(float)l->q[index+l->no]*scale;
            } else {
                a=l->fp[index];
                b=l->fp[index+l->no];
            }
            y[j]+=(1.0f-f)*a+f*b;
        }
    }
    for (unsigned j=0;j<l->no;++j) y[j]=(y[j]*l->ss[j]+l->sb[j])*l->ns[j]+l->nb[j];
}

static void whole(const float *tab, unsigned points, float low, float step,
                  int anchored, const float *x, float *y) {
    y[0]=y[1]=0.0f;
    for (unsigned i=0;i<micro_input_dim;++i) {
        const float p=(x[i]-low)/step;
        /* Historical triangular basis fades to zero one cell outside its grid. */
        if (p<=-1.0f || p>=(float)points) continue;
        const int k=(int)floorf(p);
        const float f=p-k;
        for (unsigned j=0;j<2;++j) {
            float value=0.0f;
            if (k>=0 && k<(int)points) value+=(1.0f-f)*tab[((unsigned)k*micro_input_dim+i)*2+j];
            if (k+1>=0 && k+1<(int)points) value+=f*tab[((unsigned)(k+1)*micro_input_dim+i)*2+j];
            y[j]+=value;
        }
    }
    for (unsigned j=0;j<2;++j) {
        y[j]/=micro_input_dim;
        if (anchored) y[j]=micro_input_dim*y[j]-(micro_input_dim-1)*f_zero[j];
    }
}


static TableLayer dram_layers[2][3];
unsigned micro_dram_bytes=0;
unsigned micro_dram_fp32_bytes=0;
unsigned micro_dram_int8_bytes=0;
int micro_dram_ok=0;
int micro_init_dram(void) {
    if (micro_dram_ok) return 1;
#if defined(ESP_PLATFORM) && !defined(CONFIG_IDF_TARGET_ESP32S3)
    return 0;
#else
    const unsigned bytes=micro_parameter_bytes[4]+micro_parameter_bytes[5];
#ifdef ESP_PLATFORM
    uint8_t *storage=(uint8_t *)heap_caps_malloc(bytes,MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT);
#else
    uint8_t *storage=(uint8_t *)malloc(bytes);
#endif
    if (!storage) return 0;
    uint8_t *p=storage;int equal=1;
    for (unsigned quantized=0;quantized<2;++quantized) {
        uint8_t *begin=p;
        for (unsigned l=0;l<model_depth;++l) {
            const TableLayer *a=&table_variants[quantized][l];
            TableLayer *b=&dram_layers[quantized][l];*b=*a;
            if (quantized) {
                const unsigned qbytes=a->ni*a->no*a->points;
                if (qbytes%sizeof(float)) {free(storage);return 0;}
                memcpy(p,a->q,qbytes);
                if (memcmp(p,a->q,qbytes)) equal=0;
                b->q=(const int8_t *)p;p+=qbytes;
            }
            const float *sources[]={quantized?a->qs:a->fp,a->lo,a->hi,a->ss,a->sb,a->ns,a->nb};
            const unsigned counts[]={a->ni*a->no*(quantized?1:a->points),a->ni,a->ni,a->no,a->no,a->no,a->no};
            const float **dest[]={quantized?&b->qs:&b->fp,&b->lo,&b->hi,&b->ss,&b->sb,&b->ns,&b->nb};
            for (unsigned j=0;j<7;++j) {
                memcpy(p,sources[j],counts[j]*sizeof(float));
                if (memcmp(p,sources[j],counts[j]*sizeof(float))) equal=0;
                *dest[j]=(const float *)p;p+=counts[j]*sizeof(float);
            }
        }
        if ((unsigned)(p-begin)!=micro_parameter_bytes[4+quantized]) {free(storage);return 0;}
    }
    if ((unsigned)((char *)p-(char *)storage)!=bytes || !equal) {
        free(storage);return 0;
    }
    micro_dram_fp32_bytes=micro_parameter_bytes[4];
    micro_dram_int8_bytes=micro_parameter_bytes[5];
    micro_dram_bytes=bytes;micro_dram_ok=1;return 1;
#endif
}

int micro_infer(unsigned variant, const float *input, float *output) {
    if (variant>=MICRO_VARIANTS || !input || !output) return -1;
    for (unsigned i=0;i<micro_input_dim;++i) if (!isfinite(input[i])) return -2;
    if (variant==1) { whole(whole16,16,-2.0f,whole16_step,0,input,output); return 0; }
    if (variant==2) { whole(whole256,256,-2.0f,whole256_step,0,input,output); return 0; }
    if (variant==3) { whole(whole32cal,32,whole32cal_low,whole32cal_step,1,input,output); return 0; }
    float a[MICRO_MAX_INPUTS],b[MICRO_MAX_INPUTS];
    for (unsigned i=0;i<micro_input_dim;++i) a[i]=input[i];
    for (unsigned l=0;l<model_depth;++l) {
        if (variant==0) direct_layer(&direct_layers[l],a,b);
        else if (variant==8) cubic_layer(&direct_layers[l],a,b);
        else if (variant==9) table_layer(&table_variants[1][l],a,b,1);
        else if (variant==10 || variant==11) {
            if (!micro_dram_ok) return -4;
            table_layer(&dram_layers[variant-10][l],a,b,variant==11);
        } else table_layer(&table_variants[variant-4][l],a,b,0);
        for (unsigned j=0;j<direct_layers[l].no;++j) a[j]=b[j];
    }
    output[0]=a[0]; output[1]=a[1];
    return isfinite(output[0]) && isfinite(output[1]) ? 0 : -3;
}

int micro_batch(unsigned v,const float *x,float *y,unsigned n) {
    for (unsigned i=0;i<n;++i) {int r=micro_infer(v,x+i*micro_input_dim,y+2*i);if(r)return r;}return 0;
}
