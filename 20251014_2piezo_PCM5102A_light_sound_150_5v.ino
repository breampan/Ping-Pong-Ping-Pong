// =======================================================
//  ESP32-S3 + PCM5102A (I2S) + Dual Piezo(ADC) + NeoPixel(WS2812)
//  FM Pluck + Proper Ping-Pong Echo (250ms seeding)
//  + Echo-driven Icy-Blue Lights
//  + Left piezo(GPIO15) -> 左聲道起始；Right piezo(GPIO17) -> 右聲道起始
//  + 各自獨立 baseline/平滑/不應期，允許同時疊加
//  + 串列監看台印出兩路 piezo 數值與觸發
// =======================================================

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>
#include <Adafruit_NeoPixel.h>

// -------------------- I2S 腳位 (PCM5102A) --------------------
#define I2S_PORT      I2S_NUM_0
#define I2S_BCLK_PIN  4     // PCM5102A BCK
#define I2S_LRCLK_PIN 5     // PCM5102A LRCK
#define I2S_DATA_PIN  6     // ★ 改回 6（你的既有連線）

// -------------------- NeoPixel (5V WS2812) -------------------
#define LED_PIN         13
#define NUM_PIXELS      150
#define LED_ORDER       NEO_GRB
#define LED_KHZ         NEO_KHZ800
#define LED_BRIGHTNESS  180
Adafruit_NeoPixel pixels(NUM_PIXELS, LED_PIN, LED_ORDER + LED_KHZ);

static const uint8_t BASE_B = 80;
static const uint8_t BASE_G = 5;
static const float   BASE_GAMMA = 2.5f;
static const uint8_t PEAK_ADD_BASE = 170;
static const float   WHITE_BLEND = 0.18f;
static const float   WAVE_WIDTH = 12.0f;
static const int     HALF_PIX = NUM_PIXELS/2;

// -------------------- Piezo (ADC) 兩路 -----------------------
#define PIEZO_L_PIN 15   // 左：固定從左聲道起始
#define PIEZO_R_PIN 17   // 右：固定從右聲道起始

static const int RAW_MAX = 4095;
static const int TRIG_ON = 120;   // 觸發門檻（會再以 baseline 修正）
static const int FRAME_MS = 20;   // LED 更新幀距

// 單一路 piezo 狀態
struct PiezoState {
  int   pin;
  // 移動平均
  static const int MA_N = 10;
  int   maBuf[MA_N];
  int   maIdx = 0;
  long  maSum = 0;
  int   baseline = -1;
  // 觸發節流
  uint32_t lastHitMs = 0;
  const int refractoryMs = 100;
  // debug 列印節流
  uint32_t lastPrintMs = 0;
  const int printEveryMs = 60;
};
PiezoState PZL{PIEZO_L_PIN};
PiezoState PZR{PIEZO_R_PIN};

// -------------------- Audio 參數 ------------------------------
static const int   SAMPLE_RATE = 44100;
static const int   FRAMES      = 128;
static const float TAU_F       = 6.28318530718f;

// FM（左/右可做微差異，右邊加一點亮度）
static const float FREQ_MIN = 80.0f, FREQ_MAX = 320.0f;
static const float FREQ_SLEW_PER_SAMPLE = 0.50f;
static float baseFreq=140.0f, freqTarget=140.0f;
static float modRatio=1.0f;
static float modIndexMin=4.0f, modIndexMax=55.0f;
static float currentIndex=4.0f, targetIndex=4.0f, indexLerp=0.0012f;
static float carrierPhase=0.0f, modPhase=0.0f;
static float masterGain = 0.65f;

// 右邊 piezo 的音色微調（想維持一樣就都設 1.0 / 0.0）
static const float RIGHT_MOD_RATIO  = 1.07f;  // 右邊調制比微增
static const float RIGHT_INDEX_BOOST= 2.0f;   // 右邊 index 起始微增（絕對值）

// ADSR
enum Stage { IDLE, ATTACK, DECAY, SUSTAIN, RELEASE };
static Stage envStage = IDLE;
static float env=0.0f, aInc=0, dDec=0, rDec=0;
static const float A_SEC=0.003f, D_SEC=0.160f, S_LVL=0.08f, R_SEC=0.120f;
static const int   PLOCK_MS = 180;
static int32_t     noteOnTimeMs = -1;

// Proper Ping-Pong Echo（雙延遲交叉 + 連續 seeding）
static const int   MAX_DELAY_MS  = 2000;
static const int   DELAY_LEN_MAX = (int)(SAMPLE_RATE * (MAX_DELAY_MS/1000.0f));
static float*      delayL = nullptr;
static float*      delayR = nullptr;
static int         delayWrite = 0;

static float feedback  = 0.55f;   // 0..0.95
static float wet       = 0.40f;   // 0..0.90
static int   ECHO_MIN_MS = 180;
static int   ECHO_MAX_MS = 820;

static double delayMsTarget = 380.0;         // 每次 hit 設
static double delayMsF      = 380.0;         // 平滑追蹤
static const double DELAY_SLEW_PER_SAMPLE = 0.002;

static const int   SEED_DUR_MS = 250;        // 連續 seeding 時間
static int         seedRemainSamples = 0;
static double      seedPanL = 1.0, seedPanR = 0.0;
static double      seedEnv = 0.0;
static double      seedEnvDecay = 1.0;
static const double SEED_ENV_END = 0.02;

static float panStrong = 0.85f;  // 未使用，但保留以便後續擴充

// -------------------- LED EchoLight --------------------------
struct EchoLight {
  uint32_t t0Ms;
  bool     startLeft;
  float    vel;
  int      echoMs;
  float    startL;
  float    startR;
};
static const int MAX_LIGHTS = 64;
static EchoLight lights[MAX_LIGHTS];
static int       lightCount = 0;

// -------------------- 小工具 -------------------------------
static inline float clampf(float v,float lo,float hi){ return v<lo?lo:(v>hi?hi:v); }
static inline float fmap(float x,float i0,float i1,float o0,float o1){
  float t=(i1!=i0)?(x-i0)/(i1-i0):0.0f; if(t<0)t=0; if(t>1)t=1; return o0 + t*(o1-o0);
}
static inline uint8_t clamp8(int v){ return v<0?0:(v>255?255:v); }
static inline uint8_t gammaApply8(uint8_t x, float g){ float n = powf(x/255.0f, g); return (uint8_t)clamp8((int)(n*255.0f+0.5f)); }
static inline float dconstrain(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }

// -------------------- ADSR -------------------------------
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

// -------------------- Hit（聲音+燈） -----------------------
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

// 核心觸發（toLeft = true/false 固定第一拍邊）
void hit_core(float velocity, bool toLeft, bool isRightVoice){
  // ---- velocity 安全映射 ----
  float safe = powf(clampf(velocity,0.1f,1.5f), 0.6f); // 拉平
  float vel01 = fmap(safe, 0.1f, 1.5f, 0.1f, 1.0f);

  // FM：右邊可做微差異
  float thisModRatio  = isRightVoice ? (modRatio * RIGHT_MOD_RATIO) : modRatio;
  float idxStart      = (modIndexMin + (modIndexMax - modIndexMin) * vel01)
                        + (isRightVoice ? RIGHT_INDEX_BOOST : 0.0f);

  freqTarget   = fmap(vel01, 0.1f, 1.0f, FREQ_MIN, FREQ_MAX);
  currentIndex = idxStart;
  targetIndex  = modIndexMin;
  masterGain   = 0.60f + 0.20f * vel01;

  // echo 時間（與 velocity 相關）
  double echoMs = fmap(vel01, 0.1f, 1.0f, (float)ECHO_MIN_MS, (float)ECHO_MAX_MS);
  echoMs = max(150.0, echoMs);
  delayMsTarget = echoMs;
  int echoMsInt = (int)round(echoMs);

  // 250ms 連續 seeding：固定聲像到首拍那側
  seedRemainSamples = (int)(SAMPLE_RATE * (SEED_DUR_MS/1000.0f));
  seedEnv = 1.0;
  if (toLeft) { seedPanL = 1.0; seedPanR = 0.0; }
  else        { seedPanL = 0.0; seedPanR = 1.0; }

  // LED 事件
  addEchoLight(millis(), toLeft, vel01, echoMsInt);

  // 音量起動
  noteOn();

  // 將本次的調制比套用（把 modRatio 作為「即時」控制）
  modRatio = thisModRatio;
}

inline void hit_left (float v){ hit_core(v, true , false); } // 左 piezo → 左聲道開始
inline void hit_right(float v){ hit_core(v, false, true ); } // 右 piezo → 右聲道開始（微亮音色）

// -------------------- I2S 初始化 --------------------------
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

  // delay 緩衝配置（優先 PSRAM）
  delayL = (float*)heap_caps_malloc(sizeof(float)*DELAY_LEN_MAX, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  delayR = (float*)heap_caps_malloc(sizeof(float)*DELAY_LEN_MAX, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if(!delayL || !delayR){
    if(!delayL) delayL = (float*)heap_caps_malloc(sizeof(float)*DELAY_LEN_MAX, MALLOC_CAP_8BIT);
    if(!delayR) delayR = (float*)heap_caps_malloc(sizeof(float)*DELAY_LEN_MAX, MALLOC_CAP_8BIT);
  }
  if(!delayL || !delayR){ Serial.println("FATAL: no RAM for delay"); while(true){ delay(1000);} }
  for(int i=0;i<DELAY_LEN_MAX;i++){ delayL[i]=0; delayR[i]=0; }
  delayWrite = 0;

  int seedTotal = (int)(SAMPLE_RATE * (SEED_DUR_MS/1000.0f));
  seedEnvDecay  = pow(SEED_ENV_END, 1.0 / max(1, seedTotal));

  Serial.println("[I2S] ready (44.1k, I2S_MSB, proper ping-pong, PCM5102A)");
}

// -------------------- 音訊更新（含 limiter/clipper） -----
void audio_update(){
  static int16_t buf[FRAMES*2];
  size_t wt = 0;

  if(noteOnTimeMs>0 && (millis()-noteOnTimeMs)>=PLOCK_MS &&
      envStage!=RELEASE && envStage!=IDLE) noteOff();

  int di=0;
  for(int n=0;n<FRAMES;n++){
    // 平滑
    if(baseFreq<freqTarget) baseFreq = min(freqTarget, baseFreq + FREQ_SLEW_PER_SAMPLE);
    else if(baseFreq>freqTarget) baseFreq = max(freqTarget, baseFreq - FREQ_SLEW_PER_SAMPLE);

    double dErr  = delayMsTarget - delayMsF;
    double step  = (dErr>0?1:-1) * min(fabs(dErr), DELAY_SLEW_PER_SAMPLE);
    delayMsF += step;

    stepADSR();

    // FM
    if(currentIndex<targetIndex) currentIndex = min(targetIndex, currentIndex + indexLerp);
    else if(currentIndex>targetIndex) currentIndex = max(targetIndex, currentIndex - indexLerp);

    float mod    = sinf(modPhase);
    float delta  = mod * currentIndex;
    float fInst  = dconstrain(baseFreq + delta, FREQ_MIN, FREQ_MAX);
    float fMod   = dconstrain(baseFreq * modRatio, FREQ_MIN, FREQ_MAX);
    modPhase     += TAU_F * (fMod  / SAMPLE_RATE); if(modPhase     > TAU_F) modPhase     -= TAU_F;
    carrierPhase += TAU_F * (fInst / SAMPLE_RATE); if(carrierPhase > TAU_F) carrierPhase -= TAU_F;

    float dry = sinf(carrierPhase) * (masterGain * env);

    // Ping-Pong 讀/寫
    int delaySamp = (int)dconstrain((float)round(delayMsF * SAMPLE_RATE / 1000.0f), 1, (float)(DELAY_LEN_MAX-1));
    int readPos   = delayWrite - delaySamp; if(readPos<0) readPos += DELAY_LEN_MAX;
    float yL = delayL[readPos];
    float yR = delayR[readPos];

    // 連續 seeding 到單邊
    float seedL = 0, seedR = 0;
    if(seedRemainSamples>0){
      float s = (float)seedEnv;
      seedL = dry * s * (float)seedPanL;
      seedR = dry * s * (float)seedPanR;
      seedEnv *= (float)seedEnvDecay;
      seedRemainSamples--;
    }

    // Feedback Limiter（動態抑制過強回授）
    float fbDyn = feedback;
    float peakY = max(fabsf(yL), fabsf(yR));
    if (peakY > 0.9f)      fbDyn *= 0.55f;
    else if (peakY > 0.8f) fbDyn *= 0.70f;
    else if (peakY > 0.7f) fbDyn *= 0.85f;

    delayL[delayWrite] = seedL + yR * fbDyn;
    delayR[delayWrite] = seedR + yL * fbDyn;
    delayWrite++; if(delayWrite>=DELAY_LEN_MAX) delayWrite=0;

    // 最終混音
    float w = wet, dryMix = 1.0f - w;
    float mixL = dry * dryMix + yL * w;
    float mixR = dry * dryMix + yR * w;

    // Soft clip（避免爆音）
    auto softclip = [](float x){ return tanhf(x * 1.5f); };
    float outL = softclip(mixL) * 0.95f;
    float outR = softclip(mixR) * 0.95f;

    buf[di++] = (int16_t)constrain((int)(outL * 32767.0f), -32768, 32767);
    buf[di++] = (int16_t)constrain((int)(outR * 32767.0f), -32768, 32767);
  }
  i2s_write(I2S_PORT, buf, sizeof(buf), &wt, 2 /*ticks*/);
}

// -------------------- LED 更新 ----------------------------
static uint32_t lastLEDms = 0;
void led_update(){
  uint32_t now = millis();
  if (now - lastLEDms < FRAME_MS) return;
  lastLEDms = now;

  // 底色
  for(int i=0;i<NUM_PIXELS;i++){
    pixels.setPixelColor(i, pixels.Color(0, BASE_G, BASE_B));
  }

  // 疊加所有 EchoLight
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

    float speed  = fmap(L.vel, 0.1f, 1.0f, 0.7f, 1.6f);
    float tt     = min(1.0f, tIn * speed);

    float pos = leftNow ? (L.startL - (L.startL - 0.0f) * tt)
                        : (L.startR + ((float)NUM_PIXELS-1.0f - L.startR) * tt);

    float echoGain  = powf(feedback, (float)kEcho);
    float lightAmp  = (0.5f + 0.8f*L.vel) * echoGain;

    float widthScale= fmap(L.vel, 0.1f, 1.0f, 0.8f, 1.4f);
    float sigma     = WAVE_WIDTH * widthScale;

    int i0 = max(0, (int)floorf(pos - sigma*4));
    int i1 = min(NUM_PIXELS-1, (int)ceilf (pos + sigma*4));
    for(int i=i0; i<=i1; i++){
      float d = fabsf(i - pos);
      float g = expf(-0.5f * (d*d) / (sigma*sigma));
      float add = g * baseAdd * lightAmp;

      float wmix = g * WHITE_BLEND;
      addR[i] += add * (wmix);
      addG[i] += add * (wmix + 0.10f);
      addB[i] += add;
    }
  }

  // 合成 + gamma
  for(int i=0;i<NUM_PIXELS;i++){
    int r = clamp8((int)(addR[i]));
    int g = clamp8((int)(BASE_G + addG[i]));
    int b = clamp8((int)(BASE_B + addB[i]));
    r = gammaApply8(r, BASE_GAMMA);
    g = gammaApply8(g, BASE_GAMMA);
    b = gammaApply8(b, BASE_GAMMA);
    pixels.setPixelColor(i, pixels.Color(r,g,b));
  }
  pixels.show();
}

// -------------------- Piezo：單一路處理 --------------------
void piezo_init_one(PiezoState& P){
  // baseline
  int sum=0;
  for(int i=0;i<100;i++){ sum += analogRead(P.pin); delay(1); }
  P.baseline = sum/100;
  P.maSum = 0;
  for(int i=0;i<PiezoState::MA_N;i++){ P.maBuf[i]=P.baseline; P.maSum += P.baseline; }
  P.maIdx = 0;
  Serial.printf("[Piezo pin %d] baseline=%d\n", P.pin, P.baseline);
}

void piezo_update_one(PiezoState& P, bool toLeft){
  // 移動平均
  P.maSum -= P.maBuf[P.maIdx];
  int rawNow = analogRead(P.pin);
  P.maBuf[P.maIdx] = rawNow;
  P.maSum += P.maBuf[P.maIdx];
  P.maIdx = (P.maIdx + 1) % PiezoState::MA_N;
  int raw = (int)(P.maSum / PiezoState::MA_N);
  raw = max(raw, P.baseline);

  // 節流列印
  if(millis()-P.lastPrintMs > (uint32_t)P.printEveryMs){
    Serial.printf("piezo%s raw=%d\n", toLeft?"L":"R", raw);
    P.lastPrintMs = millis();
  }

  // 觸發
  const int sensitivity = 5;
  uint32_t now = millis();
  if(raw >= TRIG_ON && raw > (P.baseline + sensitivity) && (now-P.lastHitMs)>(uint32_t)P.refractoryMs){
    P.lastHitMs = now;
    float vel = fmap((float)raw, (float)TRIG_ON, (float)RAW_MAX, 0.1f, 1.2f);
    if(toLeft) hit_left(vel); else hit_right(vel);
    Serial.printf("TRIG %s raw=%d vel=%.2f\n", toLeft?"L":"R", raw, vel);
  }
}

// -------------------- 鍵盤 (可選) -------------------------
void handle_serial_keys(){
  while (Serial.available()){
    int ch = Serial.read();
    if (ch=='l' || ch=='L'){ hit_left(0.95f);  Serial.println("[KEY] L"); }
    else if (ch=='r' || ch=='R'){ hit_right(1.00f); Serial.println("[KEY] R"); }
    else if (ch=='t' || ch=='T'){ hit_left(0.70f);  Serial.println("[KEY] T"); }
  }
}

// -------------------- Arduino 週期 ------------------------
void setup(){
  Serial.begin(115200);
  delay(80);

  // ADC
  analogReadResolution(12);
  analogSetPinAttenuation(PIEZO_L_PIN, ADC_11db);
  analogSetPinAttenuation(PIEZO_R_PIN, ADC_11db);

  // I2S
  audio_init();

  // LED
  pixels.begin();
  pixels.setBrightness(LED_BRIGHTNESS);
  pixels.clear(); pixels.show();

  // seed 衰減（再次保險）
  int seedTotal = (int)(SAMPLE_RATE * (SEED_DUR_MS/1000.0f));
  seedEnvDecay  = pow(SEED_ENV_END, 1.0 / max(1, seedTotal));

  // 基線校準
  piezo_init_one(PZL);
  piezo_init_one(PZR);

  Serial.println("PCM5102A + FM + Proper PingPong + SoftLimiter + IceBlue LED + Dual Piezo(15=L,17=R)");
  // 開機輕觸發（左）
  hit_left(0.6f);
}

void loop(){
  audio_update();
  piezo_update_one(PZL, /*toLeft=*/true);
  piezo_update_one(PZR, /*toLeft=*/false);
  handle_serial_keys();
  led_update();
}