#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <cstdint>
#include <immintrin.h>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

int* vertex_index = nullptr;
int* out_edge = nullptr;
float* edge_val = nullptr;
int* degree = nullptr;
size_t* offset = nullptr;
float* sqrt_record = nullptr;

vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(char *fname) {
  ifstream infile(fname);

  int source;
  int end;

  infile >> v_num >> e_num;

  while (!infile.eof()) {
    infile >> source >> end;
    if (infile.peek() == EOF) break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

void readFloat(char *fname, float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  FILE *fp = fopen(fname, "rb");
  fread(dst, num * sizeof(float), 1, fp);
  fclose(fp);
}

void initFloat(float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  memset(dst, 0, num * sizeof(float));
}

[[gnu::optimize("O3")]]
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
    float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
    float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
    float(*tmp_W)[out_dim] = (float(*)[out_dim])W;

    float W_T[in_dim*out_dim];

    #pragma omp parallel for
    for(size_t i = 0; i < in_dim*out_dim; ++i) {
        size_t row = i / out_dim;
        size_t colum = i % out_dim;
        W_T[colum*in_dim + row] = W[i];
    }
    float(*tmp_tran_W)[in_dim]=(float(*)[in_dim])W_T;

    #pragma omp parallel for
    for(size_t i = 0;i < v_num;++i)
        for(size_t j = 0;j < out_dim;++j) {
            __m512 c = _mm512_setzero_ps();
            size_t k = 0;
            for(;k+15<in_dim;k+=16) {
                __m512 a = _mm512_loadu_ps((*(tmp_in_X+i)+k));
                __m512 b = _mm512_loadu_ps((*(tmp_tran_W+j)+k));
                c = _mm512_fmadd_ps(a, b, c);
            }
            float cc[16] = {};
            _mm512_storeu_ps(cc, c);
            tmp_out_X[i][j] = cc[0]+cc[1]+cc[2]+cc[3]+c[4]+cc[5]+cc[6]+cc[7]+cc[8]+cc[9]+cc[10]+cc[11]+cc[12]+cc[13]+cc[14]+cc[15];
        }
    int rem = in_dim & (16 - 1);
    if(rem != 0) {
      #pragma omp parallel for
      for(size_t i = 0;i < v_num;++i)
        for(size_t j = 0;j < out_dim;++j)
          for(size_t kk = in_dim-rem; kk < in_dim; ++kk)
            tmp_out_X[i][j] += tmp_in_X[i][kk] * tmp_W[kk][j];
    }
}

[[gnu::optimize("O3")]]
void AX(int dim, float *in_X, float *out_X) {
    float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
    float(*tmp_out_X)[dim] = (float(*)[dim])out_X;

    int max_threads = omp_get_max_threads();
    const int bs = 16;
    int loop_num = dim/bs;
    __m512 dest_arr[max_threads][loop_num];
    __m512 source_arr[max_threads][loop_num];

    #pragma omp parallel for
    for(size_t v = 0; v < v_num; ++v) {
        int threadnow = omp_get_thread_num();
        for(size_t i = 0; i < loop_num; ++i) {
            dest_arr[threadnow][i] = _mm512_loadu_ps(reinterpret_cast<float const *>(&(tmp_out_X[v][i*bs])));;
        }
        int start = vertex_index[v];
        int end = vertex_index[v+1];
        for(size_t j = start; j < end; ++j) 
        {
            int nbr = out_edge[j];
            float weight = edge_val[j];
            __m256 w=_mm256_broadcast_ss(reinterpret_cast<float const *>(&(weight)));
            __m512 w2=_mm512_broadcast_f32x8(w);
            for(size_t i=0;i<loop_num;i+=2){
                __m512 source= _mm512_loadu_ps(reinterpret_cast<float const *>(&(tmp_in_X[nbr][i*bs])));
                dest_arr[threadnow][i] = _mm512_fmadd_ps(source,w2,dest_arr[threadnow][i]);
            }
            for (size_t i = bs*loop_num; i < dim; i++) {
                tmp_out_X[v][i] += tmp_in_X[nbr][i] * weight;
            }
        }
        for(size_t i = 0; i < loop_num; i++) {
            _mm512_storeu_ps(&(tmp_out_X[v][i*bs]), dest_arr[threadnow][i]);
        }
    }
}

[[gnu::optimize("O3")]]
void ReLU(int dim, float *X) {
    const int bs = 8;
    int loop_num = v_num * dim / bs;
    __m256 h= _mm256_setzero_ps();

    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < loop_num; i++){
        __m256 w = _mm256_loadu_ps(reinterpret_cast<float const *>(& X[i*bs]));
        const __m256 comp2 = _mm256_cmp_ps(w, h, _CMP_LT_OQ);
        const __m256 result = _mm256_blendv_ps(w, h, comp2);
        _mm256_storeu_ps(&(X[i*bs]), result);
    }
    for(size_t i = loop_num*bs; i < v_num * dim; i++)
        if (X[i] < 0) X[i] = 0;
}

[[gnu::optimize("O3")]]
void LogSoftmax(int dim, float *X) {
    float(*tmp_X)[dim] = (float(*)[dim])X;

    #pragma omp parallel for
    for (size_t i = 0; i < v_num; i++)
    {
        float max = tmp_X[i][0];
        for (size_t j = 1; j < dim; j++) {
            if (tmp_X[i][j] > max) max = tmp_X[i][j];
        }
        float sum = 0;
        #pragma GCC ivdep
        for (size_t j = 0; j < dim; j++) {
            sum += exp(tmp_X[i][j] - max);
        }
        sum = log(sum);
        for (size_t j = 0; j < dim; j++) {
            tmp_X[i][j] = tmp_X[i][j] - max - sum;
        }
    }
}

[[gnu::optimize("O3")]]
float MaxRowSum(float *X, int dim) {
  float(*tmp_X)[dim] = (float(*)[dim])X;
  float max = -__FLT_MAX__;

  int loop_num = dim/16;

  #pragma omp parallel for schedule(guided) reduction(max:max)
    for (int i = 0; i < v_num; i++) {
        float sum = 0;
        __m512 result =_mm512_setzero_ps();
        for(int j = 0; j < loop_num; j++){
            __m512 a = _mm512_loadu_ps(reinterpret_cast<float const *>(&(tmp_X[i][j*16])));
            result = _mm512_add_ps(result, a);
        }
        sum += ((float *)&result)[0]+((float*)&result)[1]+((float *)&result)[2]+((float*)&result)[3]+((float *)&result)[4]+((float*)&result)[5]+((float *)&result)[6]+((float*)&result)[7]+((float *)&result)[8] + ((float*)&result)[9]+((float *)&result)[10]+((float*)&result)[11]+((float *)&result)[12] + ((float*)&result)[13]+((float *)&result)[14]+((float*)&result)[15];
        for(int j = loop_num * 16; j < dim; j++) {
            sum += tmp_X[i][j];
        }
        if (sum > max) max = sum;
    }
    return max;
}

void freeFloats() {
  free(X0);
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
}

void somePreprocessing() {
  int raw_size = raw_graph.size();

  #pragma omp parallel for schedule(guided)
  for(size_t i = 0; i < raw_graph.size(); i+=2) {
    __sync_add_and_fetch(&vertex_index[raw_graph[i]+1], 1);
  }

  size_t sum = 0;
  vertex_index[0] = 0;
  #pragma omp parallel for schedule(guided)
  for(size_t i = 0; i < v_num; ++i) {
      sqrt_record[i-1] = sqrt(vertex_index[i]);
  }
  
  for(size_t i = 1; i < v_num+1; ++i) {
    vertex_index[i] += vertex_index[i-1];
  }

  #pragma omp parallel for schedule(guided)
  for(size_t i = 0; i < raw_size; i+=2) {
    int src = raw_graph[i];
    int dst = raw_graph[i + 1];
    size_t off = __sync_fetch_and_add(&offset[src], 1);
    off += vertex_index[src];
    out_edge[off] = dst;
    edge_val[off] = 1 / (sqrt_record[src] * sqrt_record[dst]);
  }
}

[[gnu::optimize("O3")]]
void init() {
  vertex_index = (int *) malloc((v_num + 1)*sizeof(int));
  out_edge = (int *) malloc((e_num)*sizeof(int));
  edge_val = (float *) malloc((e_num)*sizeof(float));
  degree = (int *) malloc((v_num)*sizeof(int));
  offset = (size_t *) malloc((v_num)*sizeof(size_t));
  sqrt_record = (float *) malloc((v_num)*sizeof(float));
  
  memset(vertex_index, 0, sizeof(int) * (v_num + 1));
  memset(out_edge, 0, sizeof(int) * e_num);
  memset(edge_val, 0, sizeof(float) * e_num);
  memset(degree, 0, sizeof(int) * v_num);
  memset(offset, 0, sizeof(int) * v_num);
  memset(sqrt_record, 0, sizeof(float) * v_num);
}
int main(int argc, char **argv) {
  F0 = atoi(argv[1]);
  F1 = atoi(argv[2]);
  F2 = atoi(argv[3]);

  readGraph(argv[4]);
  readFloat(argv[5], X0, v_num * F0);
  readFloat(argv[6], W1, F0 * F1);
  readFloat(argv[7], W2, F1 * F2);

  initFloat(X1, v_num * F1);
  initFloat(X1_inter, v_num * F1);
  initFloat(X2, v_num * F2);
  initFloat(X2_inter, v_num * F2);

  //excluded memset time
  init();

  TimePoint start = chrono::steady_clock::now();
  omp_set_num_threads(omp_get_max_threads());
  somePreprocessing();
  XW(F0, F1, X0, X1_inter, W1);
  AX(F1, X1_inter, X1);
  ReLU(F1, X1);
  XW(F1, F2, X1, X2_inter, W2);
  AX(F2, X2_inter, X2);
  LogSoftmax(F2, X2);
  float max_sum = MaxRowSum(X2, F2);
  TimePoint end = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec = end - start;
  double l_timeMs = l_durationSec.count() * 1e3;
  printf("%.8f\n", max_sum);
  printf("%.8lf\n", l_timeMs);
  freeFloats();
}