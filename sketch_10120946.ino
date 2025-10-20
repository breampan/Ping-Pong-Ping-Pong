// =======================================================
//  ESP32-S3 + MAX98357A（I2S_MSB）
//  - 鍵盤 L/R 觸發（USB CDC + UART0）
//  - ESP32 觸控感測 TOUCH_PIN=10（非接觸近場）→ 交替左右 echo
//  - Proper Ping-Pong Echo（雙延遲交叉回授 + 連續 seeding）
//  - 冰晶白亮 LED（Adafruit_NeoPixel GPIO13, 24V WS2811, BRG）
//  *音色/回聲承襲你「有聲音的那版」；LED 更白亮；觸控更靈敏穩定*
// =======================================================

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>
#include <Adafruit_NeoPixel.h>

// -------------------- I2S 腳位（你測過會發聲） --------------------
#define I2S_PORT      I2S_NUM_0
#define I2S_BCLK_PIN  4
#define I2S_LRCLK_PIN 5
#define I2S_DATA_PIN  6

// 同時監聽 USB CDC 與 UART0（確保 l/r 都讀得到）
HardwareSerial U0(0); // TX=GPIO43, RX=GPIO44 (ESP32-S3)

// -------------------- LED（24V WS2811 常見 BRG） --------------------
#define LED_PIN       13
#define NUM_PIXELS    300
#define BRIGHTNESS    180       // 比你上一版再亮一些
Adafruit_NeoPixel pixels(NUM_PIXELS, LED_PIN, NEO_BRG + NEO_KHZ800);

const int   HALF_PIX   = NUM_PIXELS/2;

// 冰晶藍底（更白亮）
const uint8_t  BASE_B        = 100;  // ↑ 底藍提升
const uint8_t  BASE_G        = 12;   // ↑ 淡綠提一點，讓藍更像冰
const uint8_t  BASE_R        = 6;    // ★ 稍加紅，推向冰晶白
const float    BASE_GAMMA    = 2.4f; // 稍降 gamma → 視覺更亮一點

// 脈衝峰值顏色控制
const uint8_t  PEAK_ADD_BASE = 170;  // ↑ 峰值更亮
const float    WHITE_BLEND   = 0.22f; // ↑ 白光混入比例（0.15→0.22）
const float    WAVE_WIDTH    = 12.0f;

// 可選的整條白度推升（0..80），會在每像素最後再加
const uint8_t  WHITE_PUSH    = 18;   // ↑ 冰晶白底光

// -------------------- Audio（承襲你穩定那版的參數） --------------------
static const int   SAMPLE_RATE = 44100;
static const int   FRAMES      = 128;
static const float TAU_F       = 6.28318530718f;

// FM
static const float FREQ_MIN = 80.0f, FREQ_MAX = 320.0f;
static const float FREQ_SLEW_PER_SAMPLE = 0.50f;

enum Stage { IDLE, ATTACK, DECAY, SUSTAIN, RELEASE };
static Stage envStage = IDLE;
static float env=0.0f, aInc=0, dDec=0, rDec=0;
static const float A_SEC=0.003f, D_SEC=0.160f, S_LVL=0.08f, R_SEC=0.120f;
static const int   PLOCK_MS = 180;
static int32_t     noteOnTimeMs = -1;

static float baseFreq=140.0f, freqTarget=140.0f;
static float modRatio=1.0f;
static float modIndexMin=4.0f, modIndexMax=55.0f;   // 和你給的接近
static float currentIndex=4.0f, targetIndex=4.0f, indexLerp=0.0012f;
static float carrierPhase=0.0f, modPhase=0.0f;
static float masterGain = 0.65f;

// Proper Ping-Pong Echo（雙延遲交叉回授 + 連續 seeding）
static const int   MAX_DELAY_MS  = 2000;
static const int   DELAY_LEN_MAX = (int)(SAMPLE_RATE * (MAX_DELAY_MS/1000.0f));
static float*      delayL = nullptr;
static float*      delayR = nullptr;
static int         delayWrite = 0;

static float feedback  = 0.55f;   // 0..0.95
static float wet       = 0.40f;   // 0..0.90

// velocity → echo 時間
static int   ECHO_MIN_MS = 180;
static int   ECHO_MAX_MS = 820;

static double delayMsTarget = 380.0;
static double delayMsF      = 380.0;
static const double DELAY_SLEW_PER_SAMPLE = 0.002;

static const int   SEED_DUR_MS = 250;
static int         seedRemainSamples = 0;
static double      seedPanL = 1.0, seedPanR = 0.0;
static double      seedEnv = 0.0;
static double      seedEnvDecay = 1.0;
static const double SEED_ENV_END = 0.02;

static bool  nextEchoLeft = true;
static float panStrong = 0.85f;

// -------------------- LED：EchoLight --------------------
struct EchoLight {
  uint32_t t0Ms;
  bool     startLeft;
  float    vel;    // 0.1..1.5
  int      echoMs;
  float    startL;
  float    startR;
};
static const int MAX_LIGHTS = 64;
static EchoLight lights[MAX_LIGHTS];
static int       lightCount = 0;

// -------------------- 小工具 --------------------
static inline float clampf(float v,float lo,float hi){ return v<lo?lo:(v>hi?hi:v); }
static inline float fmap(float x,float i0,float i1,float o0,float o1){
  float t=(i1!=i0)?(x-i0)/(i1-i0):0.0f; if(t<0)t=0; if(t>1)t=1; return o0 + t*(o1-o0);
}
inline uint8_t clamp8i(int v){ return v<0?0:(v>255?255:v); }
static inline uint8_t gamma8(uint8_t x, float g){ float n = powf(x/255.0f, g); return (uint8_t)clamp8i((int)(n*255.0f+0.5f)); }
static inline float dconstrain(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }

// -------------------- ADSR --------------------
inline void stepADSR(){
  switch(envStage){
    case IDLE:     env=0; break;
    case ATTACK:   env += aInc; if(env>=1.0f){ env=1.0f; envStage=DECAY; } break;
    case DECAY:    env -= dDec; if(env<=S_LVL){ env=S_LVL; envStage=SUSTAIN; } break;
    case SUSTAIN:  env = S_LVL; break;
    case RELEASE:  env -= rDec; if(env<=0){ env=0; envStage=IDLE; } break;
  }
}
inline void noteOn(){ envStage=ATTACK; noteOnTimeMs = millis(); }
inline void noteOff(){ envStage=RELEASE; noteOnTimeMs = -1; }

// -------------------- Hit（聲音 + LED 事件） --------------------
void addEchoLight(uint32_t nowMs, bool startLeft, float velocity, int echoMs){
  float startL = fmap(velocity, 0.1f, 1.5f, HALF_PIX-1, HALF_PIX * 0.55f);
  float startR = fmap(velocity, 0.1f, 1.5f, HALF_PIX,   HALF_PIX + HALF_PIX * 0.45f);

  EchoLight el{ nowMs, startLeft, velocity, echoMs, startL, startR };
  if (lightCount < MAX_LIGHTS){
    lights[lightCount++] = el;
  }else{
    for(int i=1;i<MAX_LIGHTS;i++) lights[i-1] = lights[i];
    lights[MAX_LIGHTS-1] = el;
  }
}

void hit(float velocity){
  velocity = dconstrain(velocity, 0.1f, 1.5f);

  freqTarget   = fmap(velocity, 0.1f, 1.5f, FREQ_MIN, FREQ_MAX);
  float vel1   = (velocity>1.0f)?1.0f:velocity;
  float idx    = modIndexMin + (modIndexMax - modIndexMin) * vel1;
  currentIndex = idx; targetIndex = modIndexMin;
  masterGain   = 0.60f + 0.20f * vel1;

  double echoMs = fmap(velocity, 0.1f, 1.5f, (float)ECHO_MIN_MS, (float)ECHO_MAX_MS);
  delayMsTarget = echoMs;
  int echoMsInt = (int)round(echoMs);

  nextEchoLeft = !nextEchoLeft;

  seedRemainSamples = (int)(SAMPLE_RATE * (SEED_DUR_MS/1000.0f));
  seedEnv = 1.0;
  if (nextEchoLeft) { seedPanL = 1.0; seedPanR = 0.0; }
  else              { seedPanL = 0.0; seedPanR = 1.0; }

  addEchoLight(millis(), nextEchoLeft, velocity, echoMsInt);
  noteOn();
}

inline void hit_side(float v, bool toLeft){
  nextEchoLeft = toLeft;
  v = dconstrain(v, 0.1f, 1.5f);

  freqTarget   = fmap(v, 0.1f, 1.5f, FREQ_MIN, FREQ_MAX);
  float vel1   = (v>1.0f)?1.0f:v;
  float idx    = modIndexMin + (modIndexMax - modIndexMin) * vel1;
  currentIndex = idx; targetIndex = modIndexMin;
  masterGain   = 0.60f + 0.20f * vel1;

  double echoMs = fmap(v, 0.1f, 1.5f, (float)ECHO_MIN_MS, (float)ECHO_MAX_MS);
  delayMsTarget = echoMs;
  int echoMsInt = (int)round(echoMs);

  seedRemainSamples = (int)(SAMPLE_RATE * (SEED_DUR_MS/1000.0f));
  seedEnv = 1.0;
  if (toLeft) { seedPanL = 1.0; seedPanR = 0.0; } else { seedPanL = 0.0; seedPanR = 1.0; }

  addEchoLight(millis(), toLeft, v, echoMsInt);
  noteOn();
}

// -------------------- I2S 初始化 --------------------
void audio_init(){
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = FRAMES,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_BCLK_PIN,
    .ws_io_num    = I2S_LRCLK_PIN,
    .data_out_num = I2S_DATA_PIN,
    .data_in_num  = I2S_PIN_NO_CHANGE
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);

  aInc = (A_SEC<=0)?1.0f:(1.0f/(A_SEC*SAMPLE_RATE));
  dDec = (D_SEC<=0)?1.0f:((1.0f - S_LVL)/(D_SEC*SAMPLE_RATE));
  rDec = (R_SEC<=0)?1.0f:(S_LVL/(R_SEC*SAMPLE_RATE));

  delayL = (float*)heap_caps_malloc(sizeof(float)*DELAY_LEN_MAX, MALLOC_CAP_8BIT);
  delayR = (float*)heap_caps_malloc(sizeof(float)*DELAY_LEN_MAX, MALLOC_CAP_8BIT);
  if(!delayL || !delayR){ Serial.println("FATAL: no RAM for delay"); while(true){ delay(1000);} }
  for(int i=0;i<DELAY_LEN_MAX;i++){ delayL[i]=0; delayR[i]=0; }
  delayWrite = 0;

  int seedTotal = (int)(SAMPLE_RATE * (SEED_DUR_MS/1000.0f));
  seedEnvDecay  = pow(SEED_ENV_END, 1.0 / max(1, seedTotal));

  Serial.println("[I2S] ready (44.1k, I2S_MSB, proper ping-pong)");
}

// -------------------- 音訊更新 --------------------
void audio_update(){
  static int16_t buf[FRAMES*2];
  size_t wt = 0;

  if(noteOnTimeMs>0 && (millis()-noteOnTimeMs)>=PLOCK_MS &&
      envStage!=RELEASE && envStage!=IDLE) noteOff();

  int di=0;
  for(int n=0;n<FRAMES;n++){
    if(baseFreq<freqTarget) baseFreq = min(freqTarget, baseFreq + FREQ_SLEW_PER_SAMPLE);
    else if(baseFreq>freqTarget) baseFreq = max(freqTarget, baseFreq - FREQ_SLEW_PER_SAMPLE);

    double dErr  = delayMsTarget - delayMsF;
    double step  = (dErr>0?1:-1) * min(fabs(dErr), DELAY_SLEW_PER_SAMPLE);
    delayMsF += step;

    stepADSR();

    if(currentIndex<targetIndex) currentIndex = min(targetIndex, currentIndex + indexLerp);
    else if(currentIndex>targetIndex) currentIndex = max(targetIndex, currentIndex - indexLerp);

    float mod    = sinf(modPhase);
    float delta  = mod * currentIndex;
    float fInst  = dconstrain(baseFreq + delta, FREQ_MIN, FREQ_MAX);
    float fMod   = dconstrain(baseFreq * modRatio, FREQ_MIN, FREQ_MAX);
    modPhase     += TAU_F * (fMod  / SAMPLE_RATE); if(modPhase     > TAU_F) modPhase     -= TAU_F;
    carrierPhase += TAU_F * (fInst / SAMPLE_RATE); if(carrierPhase > TAU_F) carrierPhase -= TAU_F;

    float dry = sinf(carrierPhase) * (masterGain * env);

    int delaySamp = (int)dconstrain((float)round(delayMsF * SAMPLE_RATE / 1000.0f), 1, (float)(DELAY_LEN_MAX-1));
    int readPos   = delayWrite - delaySamp; if(readPos<0) readPos += DELAY_LEN_MAX;

    float yL = delayL[readPos];
    float yR = delayR[readPos];

    float seedL = 0, seedR = 0;
    if(seedRemainSamples>0){
      float s = (float)seedEnv;
      seedL = dry * s * (float)seedPanL;
      seedR = dry * s * (float)seedPanR;
      seedEnv *= (float)seedEnvDecay;
      seedRemainSamples--;
    }

    float fb = feedback;
    delayL[delayWrite] = seedL + yR * fb;   // 交叉回授
    delayR[delayWrite] = seedR + yL * fb;
    delayWrite++; if(delayWrite>=DELAY_LEN_MAX) delayWrite=0;

    float w = wet, dryMix = 1.0f - w;
    float outL = dry * dryMix + yL * w;
    float outR = dry * dryMix + yR * w;

    buf[di++] = (int16_t)constrain((int)(outL * 32767.0f), -32768, 32767);
    buf[di++] = (int16_t)constrain((int)(outR * 32767.0f), -32768, 32767);
  }
  i2s_write(I2S_PORT, buf, sizeof(buf), &wt, 2);
}

// -------------------- LED：重建條帶 --------------------
static unsigned long lastLEDms = 0;
static const uint16_t FRAME_MS = 25;   // ~40fps

void led_update(){
  unsigned long now = millis();
  if (now - lastLEDms < FRAME_MS) return;
  lastLEDms = now;

  // 冰晶白底（R/G/B 都給一點）
  for(int i=0;i<NUM_PIXELS;i++){
    uint8_t r = gamma8(BASE_R, BASE_GAMMA);
    uint8_t g = gamma8(BASE_G, BASE_GAMMA);
    uint8_t b = gamma8(BASE_B, BASE_GAMMA);
    // 推一點白
    r = clamp8i(r + WHITE_PUSH);
    g = clamp8i(g + WHITE_PUSH);
    b = clamp8i(b + WHITE_PUSH);
    pixels.setPixelColor(i, pixels.Color(r, g, b));
  }

  static float addR[NUM_PIXELS];
  static float addG[NUM_PIXELS];
  static float addB[NUM_PIXELS];
  for(int i=0;i<NUM_PIXELS;i++){ addR[i]=0; addG[i]=0; addB[i]=0; }

  const float baseAdd = (float)PEAK_ADD_BASE;
  for(int li=0; li<lightCount; li++){
    const EchoLight &L = lights[li];
    long dt = (long)(now - L.t0Ms);
    if (dt < 0) continue;

    int period = max(1, L.echoMs);
    int kEcho  = (int)floorf((float)dt / (float)period);
    if (kEcho > 24) continue;

    bool leftNow = L.startLeft ? ((kEcho % 2) == 0) : ((kEcho % 2) == 1);
    float tIn    = (float)(dt % period) / (float)period;

    float speed  = fmap(L.vel, 0.1f, 1.5f, 0.7f, 1.6f);
    float tt     = min(1.0f, tIn * speed);

    float pos = leftNow ? (L.startL - (L.startL - 0.0f) * tt)
                        : (L.startR + ((float)NUM_PIXELS-1.0f - L.startR) * tt);

    float echoGain  = powf(feedback, (float)kEcho);
    float vel1      = L.vel>1.0f ? 1.0f : L.vel;
    float lightAmp  = (0.5f + 0.8f*vel1) * echoGain;

    float widthScale= fmap(L.vel, 0.1f, 1.5f, 0.8f, 1.4f);
    float sigma     = WAVE_WIDTH * widthScale;

    int i0 = max(0, (int)floorf(pos - sigma*4));
    int i1 = min(NUM_PIXELS-1, (int)ceilf (pos + sigma*4));
    for(int i=i0; i<=i1; i++){
      float d = fabsf(i - pos);
      float g = expf(-0.5f * (d*d) / (sigma*sigma));  // 0..1
      float add = g * baseAdd * lightAmp;

      float wmix = g * WHITE_BLEND;
      addR[i] += add * (wmix);
      addG[i] += add * (wmix + 0.12f); // G 比例略升，偏冰白
      addB[i] += add;
    }
  }

  for(int i=0;i<NUM_PIXELS;i++){
    int r = clamp8i((int)(addR[i]));
    int g = clamp8i((int)(addG[i]));
    int b = clamp8i((int)(addB[i]));
    // 疊加在底色上（已做 gamma）
    uint32_t cur = pixels.getPixelColor(i);
    uint8_t cr = (cur >> 16) & 0xFF;
    uint8_t cg = (cur >>  8) & 0xFF;
    uint8_t cb = (cur      ) & 0xFF;

    cr = clamp8i(cr + gamma8(r, BASE_GAMMA));
    cg = clamp8i(cg + gamma8(g, BASE_GAMMA));
    cb = clamp8i(cb + gamma8(b, BASE_GAMMA));
    pixels.setPixelColor(i, pixels.Color(cr,cg,cb));
  }
  pixels.show();
}

// =======================================================
// ===============  觸控（TOUCH_PIN=10）  =================
// =======================================================
// 目標：非接觸觸發、穩定不亂跳（自動校正 + IIR 濾波 + 相對門檻 + 不應期）

#define TOUCH_PIN 10          // 你的板上可用的 touch pin（ESP32-S3：T4）
static bool     touch_enabled   = true;

static float    t_base = 0.0f; // 校正得到的基準
static float    t_fast = 0.0f; // IIR
static uint32_t t_lastCalib = 0;

static const int   T_CALIB_SAMPLES = 80;
static const float T_ALPHA         = 0.12f;    // IIR 係數
static const float T_DRIFT         = 0.0008f;  // 基準緩慢追蹤
// ESP32 touchRead()：接近時數值通常「下降」。使用相對門檻：t_fast < t_base * T_THRESH_FRAC → 觸發
static float       T_THRESH_FRAC   = 0.80f;    // ★ 越小越敏感（建議 0.80~0.90）
static const int   T_HOLD_N        = 2;        // 連續 N 次低於門檻才觸發
static const uint32_t T_REFRACT_MS = 700;      // 不應期，避免連發
static uint32_t   t_lastTrigMs     = 0;
static bool       t_gate           = false;
static int        t_hold           = 0;

void touch_calibrate(){
  // 抬手狀態量測
  uint32_t sum = 0;
  for(int i=0;i<T_CALIB_SAMPLES;i++){
    sum += touchRead(TOUCH_PIN);
    delay(4);
  }
  t_base = (float)sum / (float)T_CALIB_SAMPLES;
  t_fast = t_base;
  t_lastCalib = millis();
  // 自動把門檻往中間推一點（抗季節/環境漂移）
  // 你也可以在序列埠用命令微調 T_THRESH_FRAC
  Serial.printf("[TOUCH] base=%.1f  thresh_frac=%.3f\n", t_base, T_THRESH_FRAC);
}

void touch_update(){
  if (!touch_enabled) return;

  uint32_t now = millis();
  uint16_t v = touchRead(TOUCH_PIN);

  // IIR 濾波
  t_fast = (1.0f - T_ALPHA) * t_fast + T_ALPHA * (float)v;

  // 基準慢速漂移（僅在門檻上方時才追）
  float thr = t_base * T_THRESH_FRAC;
  if (t_fast > thr){
    t_base = (1.0f - T_DRIFT) * t_base + T_DRIFT * t_fast;
  }

  // 連續判斷 + 不應期
  bool below = (t_fast > thr);
  if (below) { if (t_hold < 1000) t_hold++; } else { t_hold = 0; t_gate = false; }

  if (!t_gate && t_hold >= T_HOLD_N && (now - t_lastTrigMs) > T_REFRACT_MS){
    // 以接近程度決定 velocity（離門檻越深 → 越大）
    float depth = clampf((thr - t_fast) / max(1.0f, thr), 0.0f, 0.9f);
    float vels  = 0.45f + depth * 0.9f;   // 0.45..1.35
    // 單點 → 左右交替
    hit(vels);
    t_lastTrigMs = now;
    t_gate = true;
  }

  // 每 700ms 自動複檢一次基準（長時間穩定）
  if (now - t_lastCalib > 7000){
    touch_calibrate();
  }
}

// -------------------- 鍵盤讀取（USB CDC + U0） --------------------
void handle_serial_keys(){
  auto read_stream = [&](Stream& s, const char* tag){
    while (s.available()){
      int ch = s.read();
      if (ch=='l' || ch=='L'){ hit_side(0.95f, true );  Serial.printf("[KEY %s] 'L'\n", tag); }
      else if (ch=='r' || ch=='R'){ hit_side(1.00f, false); Serial.printf("[KEY %s] 'R'\n", tag); }
      else if (ch=='t' || ch=='T'){ hit(0.70f);            Serial.printf("[KEY %s] 'T'\n", tag); }
      else if (ch=='c' || ch=='C'){ touch_calibrate();      Serial.printf("[KEY %s] calib\n", tag); }
      else if (ch=='s' || ch=='S'){ T_THRESH_FRAC = clampf(T_THRESH_FRAC-0.02f, 0.70f, 0.95f); Serial.printf("[TOUCH] thresh_frac=%.3f (更敏感)\n", T_THRESH_FRAC); }
      else if (ch=='d' || ch=='D'){ T_THRESH_FRAC = clampf(T_THRESH_FRAC+0.02f, 0.70f, 0.95f); Serial.printf("[TOUCH] thresh_frac=%.3f (較鈍)\n",   T_THRESH_FRAC); }
    }
  };
  read_stream(Serial, "USB");
  read_stream(U0,     "U0");
}

// -------------------- Arduino 週期 --------------------
void setup(){
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis()-t0 < 2000) { delay(10); }

  // UART0 也開著，避免某些狀態下 USB CDC 讀不到鍵
  U0.begin(115200, SERIAL_8N1, 44, 43);

  // I2S/Audio
  audio_init();

  // LED
  pixels.begin();
  pixels.setBrightness(BRIGHTNESS);
  pixels.show();

  // Seed 指數衰減
  int seedTotal = (int)(SAMPLE_RATE * (SEED_DUR_MS/1000.0f));
  seedEnvDecay  = pow(SEED_ENV_END, 1.0 / max(1, seedTotal));

  // TOUCH 初始化（抬手）：
  touch_calibrate();

  Serial.println("FM + Proper PingPong Echo + Touch(10) + IceBlue LED (L/R + Touch)");
  // 開機提示：輕觸發一次（左）
  hit_side(0.7f, true);
}

void loop(){
  audio_update();
  handle_serial_keys();
  touch_update();
  led_update();
}